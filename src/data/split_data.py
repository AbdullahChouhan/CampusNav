"""Create leak-free train/val/test splits grouped by recording_id."""

from pathlib import Path
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold


def main() -> None:
    with open("params.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["seed"])
    data_cfg = cfg["data"]
    df = pd.read_csv(data_cfg["manifest_csv"])

    # Use 5-fold StratifiedGroupKFold: 3 folds train (60%), 1 fold val (20%), 1 fold test (20%)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(sgkf.split(df, df["location_label"], groups=df["recording_id"]))

    test_idx = folds[0][1]
    val_idx = folds[1][1]
    train_idx = [i for i in range(len(df)) if i not in set(test_idx) and i not in set(val_idx)]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Verify zero recording leakage
    assert set(train_df["recording_id"]).isdisjoint(set(val_df["recording_id"]))
    assert set(train_df["recording_id"]).isdisjoint(set(test_df["recording_id"]))
    assert set(val_df["recording_id"]).isdisjoint(set(test_df["recording_id"]))

    splits_dir = Path(data_cfg["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    print(
        f"Splits saved -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}"
    )


if __name__ == "__main__":
    main()
