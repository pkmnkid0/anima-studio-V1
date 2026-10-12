"""One training run, in its own process.

Running the trainer inside the server process meant (1) Stop had to wait for
the trainer to notice, (2) a failed run left global state behind
("TokenizeStrategy strategy is already set" on the next try) and (3) GPU memory
was never fully released. A separate process fixes all three: Stop kills it
immediately and every run starts completely clean.

usage: python -m native_backend.train_worker <payload.json> <anima|sdxl>
"""
from __future__ import annotations

import json
import sys
import traceback


def main() -> int:
    payload = json.load(open(sys.argv[1], encoding="utf-8"))
    kind = sys.argv[2]

    def on_progress(**kw) -> None:
        print("@@PROGRESS " + json.dumps(kw, default=str), flush=True)

    try:
        if kind == "anima":
            from . import anima_trainer as t
        elif kind == "selftest":      # used by the tests: fake slow run with tqdm-style output
            import time
            for i in range(1, 200):
                sys.stdout.write(f"\rsteps:  {i}%|##| {i}/200 [00:01<00:10,  1.0it/s, avr_loss=0.{i:03d}]")
                sys.stdout.flush()
                time.sleep(0.1)
            return 0
        elif kind == "sdxl":
            from . import sdxl_trainer as t
        else:
            print(f"Unknown trainer '{kind}'", flush=True)
            return 2
        t.run(payload["args"], payload["dataset"], on_progress=on_progress, should_stop=lambda: False)
        return 0
    except BaseException as exc:  # noqa: BLE001 - reported to the parent through the exit code + log
        traceback.print_exc()
        print(f"@@ERROR {type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
