import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.model_selection import train_test_split
from util.helper_functions import pad_with_zeros
import hydra
from omegaconf import DictConfig



@hydra.main(config_path="./../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # -------------------------------------------------------------------------
    # Load paths and parameters from Hydra config
    # -------------------------------------------------------------------------
    data_fp = Path(cfg.dataset.dataset_fp)
    test_dataset_fp = cfg.dataset.test_dataset_fp
    length = cfg.dataset.dim_series
    test_fraction = cfg.training.test_split
    output_dir = Path(cfg.dataset.output_dir)

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    scaler = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0)

    # -------------------------------------------------------------------------
    # Validate file paths
    # -------------------------------------------------------------------------
    if not data_fp.exists():
        raise FileNotFoundError(f"File not found: {data_fp}")

    # -------------------------------------------------------------------------
    # CASE 1 — No separate test dataset provided
    # -------------------------------------------------------------------------
    if test_dataset_fp is None or test_dataset_fp == "":
        print(f"Loading dataset from {data_fp} ...")
        X = np.fromfile(data_fp, dtype=np.float32).reshape(-1, length)
        print(f"Loaded {X.shape[0]} series of length {length}")

        # Normalization
        print("Applying Z-normalization (per series)...")
        X = scaler.fit_transform(X[..., np.newaxis]).squeeze(axis=2)

        # Handle NaNs
        print("Checking for NaNs...")
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            print(f"Found {nan_count} NaN values — replacing with 0.")
            X = np.nan_to_num(X, nan=0.0)
        else:
            print("No NaNs found.")

        # Split
        print(f"Splitting dataset: {100*(1-test_fraction):.1f}% train / {100*test_fraction:.1f}% test ...")
        X_train, X_test = train_test_split(X, test_size=test_fraction, random_state=42, shuffle=True)

        # Normalize again
        X_train = scaler.fit_transform(X_train).squeeze()
        X_test = scaler.fit_transform(X_test).squeeze()

        # Pad
        max_length = max(X_train.shape[1], X_test.shape[1])
        X_train = pad_with_zeros(X_train, max_length)
        X_test = pad_with_zeros(X_test, max_length)

        # Save
        train_fp = Path(cfg.dataset.processed_train_fp)
        test_fp = Path(cfg.dataset.processed_test_fp)

        X_train.astype(np.float32).tofile(train_fp)
        X_test.astype(np.float32).tofile(test_fp)

        print(f"Saved normalized train: {train_fp}")
        print(f"Saved nodmalized test:  {test_fp}")
        print(f"Padded to max length {max_length}")

    # -------------------------------------------------------------------------
    # CASE 2 — Separate test dataset provided
    # -------------------------------------------------------------------------
    else:
        test_dataset_fp = Path(test_dataset_fp)
        if not test_dataset_fp.exists():
            raise FileNotFoundError(f"File not found: {test_dataset_fp}")

        print(f"Loading train dataset from {data_fp}")
        print(f"Loading test dataset from {test_dataset_fp}")

        X_train = np.fromfile(data_fp, dtype=np.float32).reshape(-1, length, 1)
        X_test = np.fromfile(test_dataset_fp, dtype=np.float32).reshape(-1, length, 1)

        X_train, X_test = [np.squeeze(a, axis=2) for a in [X_train, X_test]]

        # Handle NaNs
        print("Checking for NaNs...")
        nans_train = np.isnan(X_train).sum()
        nans_test = np.isnan(X_test).sum()
        print(f"NaNs — train: {nans_train}, test: {nans_test}")
        if nans_train + nans_test > 0:
            print("Replacing NaNs with 0.")
            X_train = np.nan_to_num(X_train, nan=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0)

        # Normalize
        X_train = scaler.fit_transform(X_train).squeeze()
        X_test = scaler.fit_transform(X_test).squeeze()

        # Pad
        max_length = max(X_train.shape[1], X_test.shape[1])
        X_train = pad_with_zeros(X_train, max_length)
        X_test = pad_with_zeros(X_test, max_length)

        # Save
        train_fp = Path(cfg.dataset.processed_train_fp)
        test_fp = Path(cfg.dataset.processed_test_fp)

        X_train.astype(np.float32).tofile(train_fp)
        X_test.astype(np.float32).tofile(test_fp)

        print(f"Saved normalized train: {train_fp}")
        print(f"Saved normalized test:  {test_fp}")
        print(f"Padded to max length {max_length}")

    print("Done! Processed data saved to:")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
