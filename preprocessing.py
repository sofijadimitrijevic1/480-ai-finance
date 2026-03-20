"""Fraud detection preprocessing pipeline for the CSC 480 project.

This module loads the Kaggle transactions-fraud-datasets dataset, performs
basic exploration, applies feature engineering, handles missing values and
outliers, optionally applies SMOTE to the training split only, and saves the
processed train/test arrays for downstream models.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

try:
    import kagglehub
except ImportError as exc:
    raise ImportError(
        "kagglehub is required. Install dependencies with: pip install -r requirements.txt"
    ) from exc

warnings.filterwarnings("ignore")


class FraudDataPreprocessor:
    """Preprocessing pipeline for the fraud detection dataset."""

    ID_LIKE = {"id", "client_id", "card_id", "merchant_id"}
    DATASET_ID = "computingvictor/transactions-fraud-datasets"
    DATASET_FOLDER_NAME = "transactions-fraud-datasets"

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.transactions: Optional[pd.DataFrame] = None
        self.users: Optional[pd.DataFrame] = None
        self.labels: Optional[Dict[str, str]] = None
        self.cards: Optional[pd.DataFrame] = None
        self.mcc_codes: Optional[dict] = None
        self.scalers: Dict[str, object] = {}

    @classmethod
    def resolve_dataset_dir(cls, data_dir: Optional[Path] = None) -> Path:
        """Use the given directory or download the dataset with kagglehub."""
        if data_dir is not None and Path(data_dir).exists():
            return Path(data_dir).resolve()

        return Path(kagglehub.dataset_download(cls.DATASET_ID)).resolve()

    def load_data(self) -> None:
        """Load transactions, users, labels, cards, and MCC metadata when available."""
        print("=" * 60)
        print("LOADING DATA...")
        print("=" * 60)

        tx_path = self.data_dir / "transactions_data.csv"
        users_path = self.data_dir / "users_data.csv"
        labels_path = self.data_dir / "train_fraud_labels.json"
        cards_path = self.data_dir / "cards_data.csv"
        mcc_path = self.data_dir / "mcc_codes.json"

        self.transactions = pd.read_csv(tx_path)
        print(f"✓ Transactions loaded: {self.transactions.shape}")

        for col in self.ID_LIKE:
            if col in self.transactions.columns:
                self.transactions[col] = self.transactions[col].astype("string")

        self.users = pd.read_csv(users_path)
        print(f"✓ Users loaded: {self.users.shape}")

        with open(labels_path, "r", encoding="utf-8") as f:
            labels_data = json.load(f)

        self.labels = {str(k): v for k, v in labels_data.get("target", labels_data).items()}
        print(f"✓ Labels loaded: {len(self.labels)} transactions")

        fraud_count = sum(1 for v in self.labels.values() if v == "Yes")
        print(f"  Fraud cases: {fraud_count}")
        print(f"  Legitimate cases: {len(self.labels) - fraud_count}")

        if len(self.labels) < len(self.transactions):
            print("Filtering transactions to labeled training rows only...")
            labeled_mask = self.transactions["id"].astype(str).isin(self.labels.keys())
            self.transactions = self.transactions[labeled_mask].reset_index(drop=True)
            print(f"✓ Filtered transactions: {self.transactions.shape}")

        if cards_path.exists():
            self.cards = pd.read_csv(cards_path)
            print(f"✓ Cards loaded: {self.cards.shape}")

        if mcc_path.exists():
            with open(mcc_path, "r", encoding="utf-8") as f:
                self.mcc_codes = json.load(f)
            print(f"✓ MCC codes loaded: {len(self.mcc_codes)} codes")

    def explore_missing_values(self) -> None:
        print("\n" + "=" * 60)
        print("MISSING VALUES ANALYSIS")
        print("=" * 60)

        for name, df in [("Transactions", self.transactions), ("Users", self.users)]:
            if df is None:
                continue

            missing = df.isnull().sum()
            missing_pct = (missing / len(df)) * 100
            if missing.sum() == 0:
                print(f"\n{name}: No missing values")
                continue

            print(f"\n{name}:")
            for col in missing[missing > 0].index:
                print(f"  {col}: {missing[col]} ({missing_pct[col]:.2f}%)")

    def explore_outliers(self) -> None:
        print("\n" + "=" * 60)
        print("OUTLIER ANALYSIS (IQR)")
        print("=" * 60)

        if self.transactions is None:
            return

        numeric_cols = [
            c for c in self.transactions.select_dtypes(include=[np.number]).columns
            if c not in self.ID_LIKE
        ]

        for col in numeric_cols:
            q1 = self.transactions[col].quantile(0.25)
            q3 = self.transactions[col].quantile(0.75)
            iqr = q3 - q1
            if not np.isfinite(iqr) or iqr == 0:
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = ((self.transactions[col] < lower) | (self.transactions[col] > upper)).sum()

            if outliers > 0:
                pct = 100 * outliers / len(self.transactions)
                print(f"{col}: {outliers} ({pct:.2f}%)")

    def explore_class_imbalance(self) -> None:
        print("\n" + "=" * 60)
        print("CLASS IMBALANCE ANALYSIS")
        print("=" * 60)

        if self.labels is None:
            print("No labels loaded.")
            return

        labels = list(self.labels.values())
        fraud_count = sum(1 for label in labels if label == "Yes")
        normal_count = len(labels) - fraud_count

        print(f"Normal transactions: {normal_count} ({100 * normal_count / len(labels):.2f}%)")
        print(f"Fraud transactions: {fraud_count} ({100 * fraud_count / len(labels):.2f}%)")
        if fraud_count > 0:
            print(f"Imbalance ratio: {normal_count / fraud_count:.1f}:1")

    def handle_missing_values(self) -> None:
        print("\n" + "=" * 60)
        print("HANDLING MISSING VALUES")
        print("=" * 60)

        if self.transactions is None or self.users is None:
            return

        tx_copy = self.transactions.copy()
        numeric_cols = [
            c for c in tx_copy.select_dtypes(include=[np.number]).columns
            if c not in self.ID_LIKE
        ]
        if numeric_cols:
            tx_copy[numeric_cols] = SimpleImputer(strategy="median").fit_transform(tx_copy[numeric_cols])

        categorical_cols = tx_copy.select_dtypes(include=["object", "string"]).columns
        for col in categorical_cols:
            tx_copy[col] = tx_copy[col].fillna("Unknown")

        self.transactions = tx_copy
        print("✓ Transaction missing values handled")

        users_copy = self.users.copy()
        numeric_cols_u = users_copy.select_dtypes(include=[np.number]).columns
        if len(numeric_cols_u) > 0:
            users_copy[numeric_cols_u] = SimpleImputer(strategy="median").fit_transform(users_copy[numeric_cols_u])

        self.users = users_copy
        print("✓ User missing values handled")

    def handle_outliers(self, method: str = "clip", threshold: float = 1.5) -> None:
        print("\n" + "=" * 60)
        print(f"HANDLING OUTLIERS ({method.upper()})")
        print("=" * 60)

        if self.transactions is None:
            return

        tx_copy = self.transactions.copy()
        numeric_cols = [
            c for c in tx_copy.select_dtypes(include=[np.number]).columns
            if c not in self.ID_LIKE
        ]

        if method == "clip":
            for col in numeric_cols:
                q1 = tx_copy[col].quantile(0.25)
                q3 = tx_copy[col].quantile(0.75)
                iqr = q3 - q1
                if not np.isfinite(iqr) or iqr == 0:
                    continue
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                tx_copy[col] = tx_copy[col].clip(lower, upper)
            print(f"✓ Outliers clipped with threshold={threshold}")

        elif method == "remove":
            original_rows = len(tx_copy)
            for col in numeric_cols:
                q1 = tx_copy[col].quantile(0.25)
                q3 = tx_copy[col].quantile(0.75)
                iqr = q3 - q1
                if not np.isfinite(iqr) or iqr == 0:
                    continue
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                tx_copy = tx_copy[(tx_copy[col] >= lower) & (tx_copy[col] <= upper)]
            print(f"✓ Removed {original_rows - len(tx_copy)} rows containing outliers")

        self.transactions = tx_copy

    def feature_engineering(self) -> None:
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING")
        print("=" * 60)

        if self.transactions is None:
            return

        df = self.transactions.copy()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["hour"] = df["date"].dt.hour
        df["day"] = df["date"].dt.day
        df["month"] = df["date"].dt.month
        df["day_of_week"] = df["date"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        amount_raw = df["amount"].astype(str).str.replace(r"[$,]", "", regex=True)
        df["amount"] = pd.to_numeric(amount_raw, errors="coerce")
        df["amount"] = df["amount"].fillna(df["amount"].median())
        df["log_amount"] = np.sign(df["amount"]) * np.log1p(np.abs(df["amount"]))
        df["amount_squared"] = df["amount"] ** 2

        df["is_online"] = (df["use_chip"] == "Online Transaction").astype(int)
        df["is_online_merchant"] = df["merchant_city"].isna().astype(int)

        user_tx_count = df.groupby("client_id").size().reset_index(name="user_tx_count")
        user_avg_amount = df.groupby("client_id")["amount"].mean().reset_index(name="user_avg_amount")
        user_std_amount = df.groupby("client_id")["amount"].std().reset_index(name="user_std_amount")
        user_std_amount["user_std_amount"] = user_std_amount["user_std_amount"].fillna(0)

        df = df.merge(user_tx_count, on="client_id", how="left")
        df = df.merge(user_avg_amount, on="client_id", how="left")
        df = df.merge(user_std_amount, on="client_id", how="left")

        self.transactions = df
        print("✓ Feature engineering complete")

    def normalize_and_scale(self, method: str = "standard") -> None:
        print("\n" + "=" * 60)
        print(f"NORMALIZATION AND SCALING ({method.upper()})")
        print("=" * 60)

        if self.transactions is None:
            return

        df = self.transactions.copy()
        exclude_cols = {
            "id",
            "client_id",
            "card_id",
            "merchant_id",
            "date",
            "use_chip",
            "merchant_city",
            "merchant_state",
            "errors",
        }

        numeric_cols = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in exclude_cols
        ]

        scaler = StandardScaler() if method == "standard" else MinMaxScaler()

        if numeric_cols:
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            if df[numeric_cols].isna().any().any():
                df[numeric_cols] = SimpleImputer(strategy="median").fit_transform(df[numeric_cols])
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        self.transactions = df
        self.scalers[method] = scaler
        self.scalers["numeric_cols"] = numeric_cols
        print(f"✓ Scaled {len(numeric_cols)} numeric features")

    def encode_categorical(self) -> None:
        print("\n" + "=" * 60)
        print("CATEGORICAL ENCODING")
        print("=" * 60)

        if self.transactions is None:
            return

        df = self.transactions.copy()
        categorical_cols = df.select_dtypes(include=["object", "string"]).columns
        encode_cols = [col for col in categorical_cols if col not in ["id", "date", "errors"]]

        for col in encode_cols:
            if df[col].nunique(dropna=False) <= 10:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
                df.drop(columns=[col], inplace=True)
                print(f"  ✓ One-hot encoded: {col}")

        for col in ["merchant_city", "merchant_state"]:
            if col in df.columns and str(df[col].dtype) in ("object", "string"):
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                print(f"  ✓ Label encoded: {col}")

        self.transactions = df
        print("✓ Categorical encoding complete")

    def apply_smote_train_only(
        self,
        random_state: int = 42,
        test_size: float = 0.2,
        downsample_majority: bool = True,
        majority_to_minority_ratio: float = 10.0,
        max_train_rows: Optional[int] = 1_000_000,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        print("\n" + "=" * 60)
        print("TRAIN/TEST SPLIT + SMOTE (TRAIN ONLY)")
        print("=" * 60)

        if self.transactions is None or self.labels is None:
            raise ValueError("Transactions or labels must be loaded first.")

        df = self.transactions.copy()
        df["id"] = df["id"].astype(str).str.replace(r"\.0$", "", regex=True)
        df["fraud_label_raw"] = df["id"].map(self.labels).fillna("No")
        df["fraud_label"] = (df["fraud_label_raw"] == "Yes").astype(int)

        X = df.drop(["id", "fraud_label", "fraud_label_raw", "date", "errors"], axis=1, errors="ignore")
        y = df["fraud_label"].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        if max_train_rows is not None and len(y_train) > max_train_rows:
            train_df = X_train.copy()
            train_df["fraud_label"] = y_train

            fraud_df = train_df[train_df["fraud_label"] == 1]
            normal_df = train_df[train_df["fraud_label"] == 0]

            remaining = max_train_rows - len(fraud_df)
            normal_sample = normal_df.sample(
                n=min(len(normal_df), max(remaining, 0)),
                random_state=random_state,
            )
            train_df = pd.concat([fraud_df, normal_sample], axis=0).sample(frac=1, random_state=random_state)

            y_train = train_df["fraud_label"].to_numpy()
            X_train = train_df.drop(columns=["fraud_label"])

        if downsample_majority:
            train_df = X_train.copy()
            train_df["fraud_label"] = y_train

            fraud_df = train_df[train_df["fraud_label"] == 1]
            normal_df = train_df[train_df["fraud_label"] == 0]

            max_normals = int(majority_to_minority_ratio * len(fraud_df))
            if len(normal_df) > max_normals:
                normal_df = normal_df.sample(n=max_normals, random_state=random_state)
                train_df = pd.concat([fraud_df, normal_df], axis=0).sample(frac=1, random_state=random_state)

                y_train = train_df["fraud_label"].to_numpy()
                X_train = train_df.drop(columns=["fraud_label"])

        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        print("✓ SMOTE applied to training data only")
        return X_train_res.to_numpy(), y_train_res, X_test.to_numpy(), y_test

    def preprocess_all(
        self,
        handle_outliers: bool = True,
        apply_smote_flag: bool = True,
        scaling_method: str = "standard",
        test_size: float = 0.2,
        downsample_majority: bool = True,
        majority_to_minority_ratio: float = 10.0,
        max_train_rows: Optional[int] = 1_000_000,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        print("\n" + "=" * 60)
        print("STARTING PREPROCESSING PIPELINE")
        print("=" * 60)

        self.load_data()
        self.explore_missing_values()
        self.explore_outliers()
        self.explore_class_imbalance()

        self.handle_missing_values()
        if handle_outliers:
            self.handle_outliers(method="clip", threshold=1.5)
        self.feature_engineering()
        self.normalize_and_scale(method=scaling_method)
        self.encode_categorical()

        if apply_smote_flag:
            X_train, y_train, X_test, y_test = self.apply_smote_train_only(
                random_state=42,
                test_size=test_size,
                downsample_majority=downsample_majority,
                majority_to_minority_ratio=majority_to_minority_ratio,
                max_train_rows=max_train_rows,
            )
        else:
            if self.transactions is None or self.labels is None:
                raise ValueError("Transactions or labels must be loaded first.")

            df = self.transactions.copy()
            df["id"] = df["id"].astype(str).str.replace(r"\.0$", "", regex=True)
            df["fraud_label_raw"] = df["id"].map(self.labels).fillna("No")
            df["fraud_label"] = (df["fraud_label_raw"] == "Yes").astype(int)

            X = df.drop(["id", "fraud_label", "fraud_label_raw", "date", "errors"], axis=1, errors="ignore")
            y = df["fraud_label"].to_numpy()

            X_train_df, X_test_df, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            X_train = X_train_df.to_numpy()
            X_test = X_test_df.to_numpy()

        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE")
        print("=" * 60)
        print(f"Train shape: {X_train.shape}, {y_train.shape}")
        print(f"Test shape: {X_test.shape}, {y_test.shape}")

        return X_train, y_train, X_test, y_test

    def save_processed_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        output_dir: Path,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "X_train.npy", X_train)
        np.save(output_dir / "y_train.npy", y_train)
        np.save(output_dir / "X_test.npy", X_test)
        np.save(output_dir / "y_test.npy", y_test)

        print(f"✓ Saved processed arrays to {output_dir}")

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        report = [
            "=" * 70,
            "FRAUD DETECTION PREPROCESSING REPORT",
            "=" * 70,
        ]

        if self.transactions is not None:
            report.append(f"Transactions rows: {len(self.transactions)}")
            report.append(f"Features after processing: {self.transactions.shape[1]}")
        if self.users is not None:
            report.append(f"Users rows: {len(self.users)}")
        report.append(f"Scaling artifacts: {list(self.scalers.keys())}")
        report.append("=" * 70)

        report_str = "\n".join(report)

        if output_path is not None:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_str)

        return report_str


def main():
    data_dir = FraudDataPreprocessor.resolve_dataset_dir()
    output_dir = Path("processed_data")

    preprocessor = FraudDataPreprocessor(data_dir)
    X_train, y_train, X_test, y_test = preprocessor.preprocess_all(
        handle_outliers=True,
        apply_smote_flag=True,
        scaling_method="standard",
        test_size=0.2,
        downsample_majority=True,
        majority_to_minority_ratio=10.0,
        max_train_rows=1_000_000,
    )

    preprocessor.save_processed_data(X_train, y_train, X_test, y_test, output_dir)
    report = preprocessor.generate_report(output_dir / "preprocessing_report.txt")
    print("\n" + report)


if __name__ == "__main__":
    main()
