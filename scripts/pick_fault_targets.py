"""Pick where to plant each fault (F1-F4) in each study task. Deterministic, no model calls.

    python scripts/pick_fault_targets.py                     # writes data/fault_targets.json
    python scripts/pick_fault_targets.py --check runs/colab_a100/baseline_v2/qwen3-8b/clean-D0

For each task we replay its gold actions on a fresh ToolEnv, so every read output is the real one the
agents would see, and pick targets from the gold write the faults should break. W is the first gold
write that takes item IDs (every study task has one); R is the gold `get_order_details` call for W's order.

  F1 corrupted value  A non-ID value the gold write depends on, flipped to a wrong value of the same type.
                      Exchange / modify (W has new_item_ids): the new item's `available` flag in its
                      get_product_details output, True -> False, so the right variant looks unavailable.
                      Return / cancel: the order `status` in R ("delivered" <-> "pending"), which decides
                      which write is allowed. Prices are not used: no gold write takes a price argument,
                      so a wrong price would act like the F3 control (it changes no write).
  F2 wrong valid ID   W's first item_id in R's `items`, swapped for another variant of the same product:
                      a real item ID in the database, not in the order, not in any gold argument.
  F3 control          R's shipping `address.address2`, which no gold action uses.
  F4 false claim      In the researcher -> executor handoff (hop 1): W's first new_item_id (else its first
                      item_id), replaced by another variant of the same product.

Each tool-level entry has the agreed fields (tool, occurrence, path, true_value, wrong_value) plus `args`,
the arguments of the targeted gold call. ToolEnv matches faults by occurrence only (the n-th successful
call to `tool`), so `args` lets the runner check it hit the intended call; `--check` reports how often the
n-th call in clean runs had those arguments.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from handoff.env import ALL_TOOLS, ToolEnv
from handoff.tasks import Ticket, load_tickets

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "fault_targets.json"
N_TASKS = 10


def replay_reads(tk: Ticket) -> list[dict]:
    """Gold actions in order, each with its real output (reads come before writes in every gold list)."""
    te = ToolEnv()
    calls, seen = [], {}
    for a in tk.expected_actions:
        out = te.call("gold", a["name"], dict(a["arguments"]), ALL_TOOLS)
        seen[a["name"]] = seen.get(a["name"], 0) + (0 if te.log[-1].error else 1)
        calls.append({"tool": a["name"], "args": a["arguments"], "output": out,
                      "error": te.log[-1].error, "is_write": te.log[-1].is_write,
                      "occurrence": seen[a["name"]]})
    return calls


def _gold_strings(tk: Ticket) -> set[str]:
    """Every scalar value in every gold action's arguments, as strings."""
    out: set[str] = set()

    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        else:
            out.add(str(v))

    for a in tk.expected_actions:
        walk(a["arguments"])
    return out


def _first(calls, tool, pred=lambda c: True) -> dict:
    return next(c for c in calls if c["tool"] == tool and not c["error"] and pred(c))


def _other_variant(product_id: str, avoid: set[str]) -> str:
    """Smallest-ID variant of the product that is not in `avoid` (deterministic, exists in the DB)."""
    variants = json.loads(ToolEnv().call("gold", "get_product_details", {"product_id": product_id},
                                         ALL_TOOLS))["variants"]
    return sorted(v for v in variants if v not in avoid)[0]


def _entry(call: dict, path: list, true_value, wrong_value) -> dict:
    return {"tool": call["tool"], "occurrence": call["occurrence"], "path": path,
            "true_value": true_value, "wrong_value": wrong_value, "args": call["args"]}


def _suite(true: str) -> str:
    """A different address2 of the same shape: 'Suite 135' -> 'Suite 935'."""
    m = re.search(r"\d+", true or "")
    if not m:
        return "Suite 100"
    digits = m.group()
    new = str((int(digits[0]) + 8) % 10) + digits[1:]
    return true[: m.start()] + new + true[m.end():]


def pick(tk: Ticket) -> dict:
    calls = replay_reads(tk)
    gold = _gold_strings(tk)
    w = next(a for a in tk.expected_actions if a["arguments"].get("item_ids"))
    wa = w["arguments"]
    r = _first(calls, "get_order_details", lambda c: c["args"].get("order_id") == wa["order_id"])
    order = json.loads(r["output"])
    in_order = {it["item_id"] for it in order["items"]}
    targets: dict[str, dict] = {}

    # F1: a decision-relevant non-ID value
    if wa.get("new_item_ids"):
        new = wa["new_item_ids"][0]
        p = _first(calls, "get_product_details", lambda c: new in json.loads(c["output"])["variants"])
        avail = json.loads(p["output"])["variants"][new]["available"]
        targets["F1"] = _entry(p, ["variants", new, "available"], avail, not avail)
    else:
        status = order["status"]
        targets["F1"] = _entry(r, ["status"], status, "pending" if status == "delivered" else "delivered")

    # F2: a used ID swapped for another valid one
    old = wa["item_ids"][0]
    i = next(k for k, it in enumerate(order["items"]) if it["item_id"] == old)
    pid = order["items"][i]["product_id"]
    f2 = _other_variant(pid, {old} | in_order | gold)
    targets["F2"] = _entry(r, ["items", i, "item_id"], old, f2)

    # F3: control, a field no gold action uses
    a2 = order["address"]["address2"]
    targets["F3"] = _entry(r, ["address", "address2"], a2, _suite(a2))

    # F4: a value the researcher hands the executor
    true4 = (wa.get("new_item_ids") or wa["item_ids"])[0]
    pid4 = next((c for c in calls if c["tool"] == "get_product_details" and not c["error"]
                 and true4 in json.loads(c["output"])["variants"]), None)
    pid4 = json.loads(pid4["output"])["product_id"] if pid4 else pid
    targets["F4"] = {"hop": 1, "true_value": true4,
                     "false_value": _other_variant(pid4, {true4, f2} | in_order | gold)}
    return targets


def pick_all(n: int = N_TASKS) -> dict:
    return {tk.task_id: pick(tk) for tk in load_tickets()[:n]}


def check(targets: dict, folder: Path) -> None:
    """Against saved clean runs: does the n-th call hit the targeted arguments, and does the
    researcher's handoff contain F4's true value (else the F4 fork has nothing to rewrite)?"""
    print(f"{'task':>4}  {'F1 hits':<8} {'F2 hits':<8} {'F3 hits':<8} F4 value in handoff")
    for tid, t in targets.items():
        fp = folder / f"task{tid}__trial0.json"
        if not fp.exists():
            print(f"{tid:>4}  (no clean record)")
            continue
        st = json.loads(fp.read_text())["state"]
        row = []
        for f in ("F1", "F2", "F3"):
            e = t[f]
            ok = [c for c in st["tool_log"] if c["tool"] == e["tool"] and not c["error"]]
            hit = len(ok) >= e["occurrence"] and ok[e["occurrence"] - 1]["args"] == e["args"]
            row.append("yes" if hit else ("no call" if len(ok) < e["occurrence"] else "other"))
        h = next((x["content"] for x in st.get("handoffs", []) if x["from"] == "researcher"), None)
        f4 = "n/a (no researcher handoff)" if h is None else ("yes" if t["F4"]["true_value"] in h else "no")
        print(f"{tid:>4}  {row[0]:<8} {row[1]:<8} {row[2]:<8} {f4}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", type=Path, help="folder of clean records (task<ID>__trial0.json)")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    targets = pick_all()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(targets, indent=2) + "\n")
    print(f"wrote {a.out} ({len(targets)} tasks)")
    if a.check:
        check(targets, a.check)


if __name__ == "__main__":
    main()
