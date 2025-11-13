import argparse
from pathlib import Path
from typing import List, Dict

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_dataset(path: str) -> tuple[np.ndarray, Dict[str, np.ndarray], List[str], int]:
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    shared_keys = list(data["shared_keys"].tolist())
    K = int(data["K"][0]) if "K" in data else 1
    # Y keys are all remaining arrays not X/shared_keys/K
    Y = {k: data[k] for k in data.files if k not in {"X", "shared_keys", "K"}}
    return X, Y, shared_keys, K


def train(X: np.ndarray, Y: Dict[str, np.ndarray], out_dir: str) -> None:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    # Pack Y columns into a single 2D array with consistent order
    y_keys = sorted(Y.keys())
    Y_mat = np.column_stack([Y[k].reshape(-1) for k in y_keys])

    # Simple scaling + tree booster; multi-output via wrapper
    model = Pipeline([
        ("scale", StandardScaler(with_mean=False)),
        ("reg", MultiOutputRegressor(HistGradientBoostingRegressor())),
    ])

    Xtr, Xva, Ytr, Yva = train_test_split(X, Y_mat, test_size=0.2, random_state=0)
    model.fit(Xtr, Ytr)
    r2 = model.score(Xva, Yva)
    print(f"Validation R^2: {r2:.3f}")

    # Save model and metadata
    joblib.dump({
        "model": model,
        "y_keys": y_keys,
    }, outp / "x_to_y.joblib")
    print(f"Saved model to {outp / 'x_to_y.joblib'}")


def main():
    ap = argparse.ArgumentParser(description="Train f: X→Y from CBS dataset")
    ap.add_argument("--data", type=str, required=True, help="Path to cbs_dataset.npz")
    ap.add_argument("--out", type=str, default="models/x_to_y")
    args = ap.parse_args()

    X, Y, shared_keys, K = load_dataset(args.data)
    print(f"Loaded X shape {X.shape}, Y dims {list(Y.keys())}, K={K}, shared_keys={shared_keys}")
    train(X, Y, args.out)


if __name__ == "__main__":
    main()


