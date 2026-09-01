from pathlib import Path

import pandas as pd

root = Path("features/opensmile/librispeech")

rows = []

for path in root.rglob("*/*.parquet"):
    df = pd.read_parquet(path)

    jitter = df["jitterLocal_sma3nz"]

    valid = jitter[jitter != 0]
    split = path.parts[-2]

    rows.append(
        {
            "utt_id": path.stem,
            "mean_jitter_all": jitter.mean(),
            "mean_jitter_valid": valid.mean() if len(valid) else float("nan"),
            "n_frames": len(jitter),
            "n_valid": len(valid),
            "valid_fraction": len(valid) / len(jitter),
            "split": split,
        }
    )

results = pd.DataFrame(rows)

print("\nHighest mean jitter (all frames):")
print(
    results.nlargest(10, "mean_jitter_all")[
        ["utt_id", "mean_jitter_all", "n_valid", "valid_fraction", "split"]
    ]
)

print("\nHighest mean jitter (valid frames only):")
print(
    results.nlargest(10, "mean_jitter_valid")[
        ["utt_id", "mean_jitter_valid", "n_valid", "valid_fraction", "split"]
    ]
)
