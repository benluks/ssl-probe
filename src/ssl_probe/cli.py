from __future__ import annotations

import runpy
import sys

COMMANDS = {
    ("train",): "ssl_probe.commands.train",
    ("correlate",): "ssl_probe.commands.correlate",
    ("precompute", "opensmile"): "ssl_probe.commands.precompute_opensmile",
    ("precompute", "knnvc"): "ssl_probe.commands.precompute_knnvc",
    ("precompute", "speaker-stats"): "ssl_probe.commands.precompute_speaker_stats",
}


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help"}:
        print("usage: ssl-probe {train,correlate,precompute} ...")
        print("\ncommands:")
        print("  train")
        print("  correlate")
        print("  precompute opensmile")
        print("  precompute knnvc")
        print("  precompute speaker-stats")
        return

    key = (args[0],)
    consumed = 1
    if args[0] == "precompute" and len(args) >= 2:
        key = (args[0], args[1])
        consumed = 2

    module = COMMANDS.get(key)
    if module is None:
        raise SystemExit(f"Unknown command: {' '.join(args[:consumed])}")

    sys.argv = ["ssl-probe " + " ".join(key), *args[consumed:]]
    runpy.run_module(module, run_name="__main__")
