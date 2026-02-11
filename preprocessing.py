"""
Comprehensive Data Preprocessing Pipeline for Fraud Detection
Handles: Missing Values, Outliers, Normalization, Feature Engineering, Class Imbalance (SMOTE)
Compatible with: LSTM, CNN, RNN, Random Forest models

Key updates:
- Lock ID-like columns as strings early so they never get mutated
- Exclude ID-like columns from missing-value imputation and outlier handling
- Robust amount parsing + signed log transform (prevents inf/-inf)
- Finite cleanup + imputation before scaling (prevents scaler crashes)
- Label mapping fix + normalization of floaty IDs (strip trailing .0)
- Train/test split first; SMOTE applied ONLY on training set
- Optional majority downsampling + optional train-size cap for laptop feasibility
"""
#!/bin/bash

import json
import warnings
import subprocess
import zipfile
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd

# Preprocessing & ML libraries
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")


class FraudDataPreprocessor:
    """Complete preprocessing pipeline for fraud detection dataset"""

    ID_LIKE = {"id", "client_id", "card_id", "merchant_id"}
    DATASET_ZIP_URL = "https://www.kaggle.com/api/v1/datasets/download/computingvictor/transactions-fraud-datasets"

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.transactions: Optional[pd.DataFrame] = None
        self.users: Optional[pd.DataFrame] = None
        self.labels: Optional[Dict[str, str]] = None
        self.cards: Optional[pd.DataFrame] = None
        self.mcc_codes: Optional[dict] = None
        self.scalers: Dict[str, object] = {}

    @staticmethod
    def download_dataset_via_curl(download_dir: Optional[Path] = None) -> Path:
        """
        Download fraud detection dataset using curl and extract it.
        
        Args:
            download_dir: Directory to download to. Defaults to ~/Downloads
            
        Returns:
            Path to the extracted dataset directory
        """
        if download_dir is None:
            download_dir = Path.home() / "Downloads"
        
        download_dir.mkdir(parents=True, exist_ok=True)
        zip_path = download_dir / "transactions-fraud-datasets.zip"
        extract_dir = download_dir / "transactions-fraud-datasets"
        
        # Only download if not already present
        if extract_dir.exists():
            print(f"✓ Dataset already extracted at: {extract_dir}")
            return extract_dir
        
        # Download using curl
        print("=" * 60)
        print("DOWNLOADING DATA VIA CURL...")
        print("=" * 60)
        print(f"Target: {zip_path}")
        
        try:
            cmd = [
                "curl", "-L", "-o", str(zip_path),
                FraudDataPreprocessor.DATASET_ZIP_URL
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise Exception(f"curl failed: {result.stderr}")
            
            if not zip_path.exists():
                raise FileNotFoundError(f"Download did not create {zip_path}")
            
            print(f"✓ Download successful ({zip_path.stat().st_size / (1024**2):.1f} MB)")
            
            # Extract the zip
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            print(f"✓ Extraction complete: {extract_dir}")
            return extract_dir
            
        except subprocess.TimeoutExpired:
            raise Exception("Download timed out (5 minutes exceeded)")
        except Exception as e:
            raise Exception(f"Failed to download dataset: {str(e)}")

    def load_data(self) -> None:
        print("=" * 60)
        print("LOADING DATA...")
        print("=" * 60)

        tx_path = self.data_dir / "transactions_data.csv"
        self.transactions = pd.read_csv(tx_path)
        print(f"✓ Transactions loaded: {self.transactions.shape}")

        # Lock ID-like columns so preprocessing never mutates them
        for col in self.ID_LIKE:
            if col in self.transactions.columns:
                self.transactions[col] = self.transactions[col].astype("string")

        users_path = self.data_dir / "users_data.csv"
        self.users = pd.read_csv(users_path)
        print(f"✓ Users loaded: {self.users.shape}")

        labels_path = self.data_dir / "train_fraud_labels.json"
        with open(labels_path, "r") as f:
            labels_data = json.load(f)
        
        # Extract the "target" dictionary (structure: {"target": {tx_id: "Yes"/"No", ...}})
        self.labels = labels_data.get("target", labels_data)
        print(f"✓ Labels loaded: {len(self.labels)} transactions")
        
        # Count fraud vs legitimate
        fraud_count = sum(1 for v in self.labels.values() if v == "Yes")
        print(f"   - Fraud cases: {fraud_count}")
        print(f"   - Legitimate cases: {len(self.labels) - fraud_count}")

        # Normalize label keys to strings
        self.labels = {str(k): v for k, v in self.labels.items()}

        # Filter transactions to ONLY those with labels (training set)
        # Unlabeled transactions are reserved for prediction/test set
        if self.transactions is not None and len(self.labels) < len(self.transactions):
            print(f"⚠️  Dataset split:")
            print(f"   - Labeled transactions (training): {len(self.labels)}")
            print(f"   - Unlabeled transactions (prediction): {len(self.transactions) - len(self.labels)}")
            print(f"   Filtering to labeled transactions only...")
            
            tx_ids_str = self.transactions["id"].astype(str)
            labeled_mask = tx_ids_str.isin(self.labels.keys())
            self.transactions = self.transactions[labeled_mask].reset_index(drop=True)
            print(f"   ✓ Filtered to {len(self.transactions)} labeled transactions")

        cards_path = self.data_dir / "cards_data.csv"
        if cards_path.exists():
            self.cards = pd.read_csv(cards_path)
            print(f"✓ Cards data loaded: {self.cards.shape}")

        mcc_path = self.data_dir / "mcc_codes.json"
        if mcc_path.exists():
            with open(mcc_path, "r") as f:
                self.mcc_codes = json.load(f)
            print(f"✓ MCC codes loaded: {len(self.mcc_codes)} codes")

        # Optional debug: verify there is some overlap
        if self.transactions is not None and self.labels is not None:
            tx_ids_1k = set(self.transactions["id"].astype(str).head(1000))
            any_overlap = any(k in tx_ids_1k for k in list(self.labels.keys())[:5000])
            print(f"✓ Label/tx overlap in first 1k tx? {any_overlap}")

    def explore_missing_values(self) -> None:
        print("\n" + "=" * 60)
        print("MISSING VALUES ANALYSIS")
        print("=" * 60)

        for name, df in [("Transactions", self.transactions), ("Users", self.users)]:
            if df is None:
                continue
            missing = df.isnull().sum()
            missing_pct = (missing / len(df)) * 100
            if missing.sum() > 0:
                print(f"\n{name}:")
                for col in missing[missing > 0].index:
                    print(f"  {col}: {missing[col]} ({missing_pct[col]:.2f}%)")
            else:
                print(f"\n{name}: No missing values ✓")

    def explore_outliers(self) -> None:
        """Explore outliers using IQR method (excluding ID-like columns)"""
        print("\n" + "=" * 60)
        print("OUTLIERS ANALYSIS (IQR Method)")
        print("=" * 60)

        if self.transactions is None:
            return

        df = self.transactions
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in self.ID_LIKE
        ]

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            # skip degenerate columns
            if not np.isfinite(IQR) or IQR == 0:
                continue

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                pct = (outliers / len(df)) * 100
                print(f"{col}: {outliers} outliers ({pct:.2f}%)")

    def explore_class_imbalance(self) -> None:
        print("\n" + "=" * 60)
        print("CLASS IMBALANCE ANALYSIS")
        print("=" * 60)

        if self.labels is None:
            print("No labels loaded.")
            return

        fraud_labels = list(self.labels.values())
        fraud_count = sum(1 for label in fraud_labels if label == "Yes")
        normal_count = len(fraud_labels) - fraud_count

        print(f"Normal transactions: {normal_count} ({normal_count/len(fraud_labels)*100:.2f}%)")
        print(f"Fraudulent transactions: {fraud_count} ({fraud_count/len(fraud_labels)*100:.2f}%)")
        if fraud_count > 0:
            print(f"Imbalance ratio: {normal_count/fraud_count:.1f}:1")
        else:
            print("Warning: No fraudulent transactions found in labels")

    def handle_missing_values(self) -> None:
        print("\n" + "=" * 60)
        print("HANDLING MISSING VALUES")
        print("=" * 60)

        if self.transactions is None or self.users is None:
            return

        tx_copy = self.transactions.copy()

        # numeric impute excluding IDs
        numeric_cols = [
            c for c in tx_copy.select_dtypes(include=[np.number]).columns
            if c not in self.ID_LIKE
        ]
        if numeric_cols:
            imputer_num = SimpleImputer(strategy="median")
            tx_copy[numeric_cols] = imputer_num.fit_transform(tx_copy[numeric_cols])

        # categorical fill (include string dtype too)
        categorical_cols = tx_copy.select_dtypes(include=["object", "string"]).columns
        for col in categorical_cols:
            if tx_copy[col].isnull().sum() > 0:
                tx_copy[col] = tx_copy[col].fillna("Unknown")

        self.transactions = tx_copy
        print("✓ Transaction missing values handled")

        users_copy = self.users.copy()
        numeric_cols_u = users_copy.select_dtypes(include=[np.number]).columns
        if len(numeric_cols_u) > 0:
            imputer_u = SimpleImputer(strategy="median")
            users_copy[numeric_cols_u] = imputer_u.fit_transform(users_copy[numeric_cols_u])

        self.users = users_copy
        print("✓ User missing values handled")

    def handle_outliers(self, method: str = "clip", threshold: float = 1.5) -> None:
        print("\n" + "=" * 60)
        print(f"HANDLING OUTLIERS ({method.upper()} method)")
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
                Q1 = tx_copy[col].quantile(0.25)
                Q3 = tx_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                if not np.isfinite(IQR) or IQR == 0:
                    continue
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                tx_copy[col] = tx_copy[col].clip(lower_bound, upper_bound)
            print(f"✓ Outliers clipped (threshold={threshold})")

        elif method == "remove":
            original_rows = len(tx_copy)
            for col in numeric_cols:
                Q1 = tx_copy[col].quantile(0.25)
                Q3 = tx_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                if not np.isfinite(IQR) or IQR == 0:
                    continue
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                tx_copy = tx_copy[(tx_copy[col] >= lower_bound) & (tx_copy[col] <= upper_bound)]
            removed = original_rows - len(tx_copy)
            print(f"✓ {removed} rows with outliers removed")

        self.transactions = tx_copy

    def feature_engineering(self) -> None:
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING")
        print("=" * 60)

        if self.transactions is None:
            return

        df = self.transactions.copy()

        # Time-based features
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["hour"] = df["date"].dt.hour
        df["day"] = df["date"].dt.day
        df["month"] = df["date"].dt.month
        df["day_of_week"] = df["date"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        print("  ✓ Time-based features: hour, day, month, day_of_week, is_weekend")

        # Amount features (robust + signed log)
        amount_raw = df["amount"].astype(str).str.replace(r"[$,]", "", regex=True)
        df["amount"] = pd.to_numeric(amount_raw, errors="coerce")
        df["amount"] = df["amount"].fillna(df["amount"].median())
        df["log_amount"] = np.sign(df["amount"]) * np.log1p(np.abs(df["amount"]))
        df["amount_squared"] = df["amount"] ** 2
        print("  ✓ Amount features: log_amount, amount_squared")

        # Transaction type feature
        df["is_online"] = (df["use_chip"] == "Online Transaction").astype(int)
        print("  ✓ Transaction type feature: is_online")

        # Merchant features
        df["is_online_merchant"] = df["merchant_city"].isna().astype(int)
        print("  ✓ Merchant features: is_online_merchant")

        # Velocity features
        user_tx_count = df.groupby("client_id").size().reset_index(name="user_tx_count")
        user_avg_amount = df.groupby("client_id")["amount"].mean().reset_index(name="user_avg_amount")
        user_std_amount = df.groupby("client_id")["amount"].std().reset_index(name="user_std_amount")
        user_std_amount["user_std_amount"] = user_std_amount["user_std_amount"].fillna(0)

        df = df.merge(user_tx_count, on="client_id", how="left")
        df = df.merge(user_avg_amount, on="client_id", how="left")
        df = df.merge(user_std_amount, on="client_id", how="left")
        print("  ✓ Velocity features: user_tx_count, user_avg_amount, user_std_amount")

        self.transactions = df
        print("✓ Feature engineering completed")

    def normalize_and_scale(self, method: str = "standard") -> None:
        print("\n" + "=" * 60)
        print(f"NORMALIZATION & SCALING ({method.upper()} Scaler)")
        print("=" * 60)

        if self.transactions is None:
            return

        df = self.transactions.copy()

        exclude_cols = {
            "id", "client_id", "card_id", "merchant_id",
            "date", "use_chip", "merchant_city", "merchant_state", "errors",
        }
        numeric_cols = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in exclude_cols
        ]

        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        if numeric_cols:
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            if df[numeric_cols].isna().any().any():
                imputer = SimpleImputer(strategy="median")
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        self.scalers[method] = scaler
        self.scalers["numeric_cols"] = numeric_cols
        print(f"✓ Scaled {len(numeric_cols)} numeric features")

        self.transactions = df

    def encode_categorical(self) -> None:
        print("\n" + "=" * 60)
        print("CATEGORICAL ENCODING")
        print("=" * 60)

        if self.transactions is None:
            return

        df = self.transactions.copy()

        categorical_cols = df.select_dtypes(include=["object", "string"]).columns
        encode_cols = [col for col in categorical_cols if col not in ["id", "date", "errors"]]

        # One-hot encode low-cardinality categorical
        for col in encode_cols:
            if df[col].nunique(dropna=False) <= 10:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
                df.drop(col, axis=1, inplace=True)
                print(f"  ✓ One-hot encoded: {col}")

        # Label encode city/state if present and still non-numeric
        for col in ["merchant_city", "merchant_state"]:
            if col in df.columns and df[col].dtype in ("object", "string"):
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                print(f"  ✓ Label encoded: {col}")

        self.transactions = df
        print("✓ Categorical encoding completed")

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
            raise ValueError("Transactions or labels not loaded.")

        df = self.transactions.copy()

        # ---- LABEL MAPPING (FIXED) ----
        df["id"] = df["id"].astype(str).str.replace(r"\.0$", "", regex=True)

        df["fraud_label_raw"] = df["id"].map(self.labels)
        missing_rate = df["fraud_label_raw"].isna().mean()
        if missing_rate > 0:
            print(f"⚠️ Label map missing rate: {missing_rate:.2%} (filling as 'No')")

        df["fraud_label_raw"] = df["fraud_label_raw"].fillna("No")
        df["fraud_label"] = (df["fraud_label_raw"] == "Yes").astype(int)

        n0_all = int((df["fraud_label"] == 0).sum())
        n1_all = int((df["fraud_label"] == 1).sum())
        print("Label counts (all data):")
        print(f"  Normal: {n0_all}")
        print(f"  Fraud:  {n1_all}")

        if n0_all == 0 or n1_all == 0:
            raise ValueError(
                "Need both classes (0 and 1) to apply SMOTE. "
                "Your labels mapping produced only one class."
            )

        X = df.drop(["id", "fraud_label", "fraud_label_raw", "date", "errors"], axis=1, errors="ignore")
        y = df["fraud_label"].to_numpy()

        # Train/test split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print("\nAfter train/test split:")
        print(f"  Train size: {len(y_train)} | Normal: {(y_train==0).sum()} | Fraud: {(y_train==1).sum()}")
        print(f"  Test  size: {len(y_test)}  | Normal: {(y_test==0).sum()}  | Fraud: {(y_test==1).sum()}")

        # Cap train size
        if max_train_rows is not None and len(y_train) > max_train_rows:
            train_df = X_train.copy()
            train_df["fraud_label"] = y_train

            fraud_df = train_df[train_df["fraud_label"] == 1]
            normal_df = train_df[train_df["fraud_label"] == 0]

            remaining = max_train_rows - len(fraud_df)
            if remaining <= 0:
                train_df = fraud_df.sample(n=max_train_rows, random_state=random_state)
            else:
                normal_sample = normal_df.sample(n=min(len(normal_df), remaining), random_state=random_state)
                train_df = pd.concat([fraud_df, normal_sample], axis=0).sample(frac=1, random_state=random_state)

            y_train = train_df["fraud_label"].to_numpy()
            X_train = train_df.drop(columns=["fraud_label"])
            print(f"\n⚠️ Capped train rows to {len(y_train)} (kept all fraud, sampled normals)")

        # Downsample majority before SMOTE
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
                print(f"\n✓ Downsampled train majority to {max_normals} (ratio {majority_to_minority_ratio}:1)")

            print("Train counts before SMOTE:")
            print(f"  Normal: {(y_train==0).sum()}")
            print(f"  Fraud:  {(y_train==1).sum()}")
            print(f"  Ratio:  {(y_train==0).sum()/(y_train==1).sum():.1f}:1")

        # SMOTE on train only
        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        print("\nAfter SMOTE (train only):")
        print(f"  Normal: {(y_train_res==0).sum()}")
        print(f"  Fraud:  {(y_train_res==1).sum()}")
        print(f"  Ratio:  {(y_train_res==0).sum()/(y_train_res==1).sum():.1f}:1")
        print("✓ SMOTE applied successfully (train only)")

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
        print("STARTING COMPLETE PREPROCESSING PIPELINE")
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
                raise ValueError("Transactions or labels not loaded.")

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
        print("PREPROCESSING COMPLETED")
        print("=" * 60)
        print(f"Train X shape: {X_train.shape} | Train y shape: {y_train.shape}")
        print(f"Test  X shape: {X_test.shape}  | Test  y shape: {y_test.shape}")

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

        print(f"\n✓ Processed train/test data saved to {output_dir}")

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        report = []
        report.append("=" * 70)
        report.append("FRAUD DETECTION PREPROCESSING REPORT")
        report.append("=" * 70)
        if self.transactions is not None:
            report.append(f"\nTransactions rows: {len(self.transactions)}")
            report.append(f"Features after processing: {self.transactions.shape[1]}")
        if self.users is not None:
            report.append(f"Users rows: {len(self.users)}")
        report.append(f"Scaling artifacts: {list(self.scalers.keys())}")
        report.append("\n" + "=" * 70)

        report_str = "\n".join(report)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report_str)

        return report_str


def main():
    # Download dataset via curl if not already present
    download_dir = Path.home() / "Downloads"
    data_dir = FraudDataPreprocessor.download_dataset_via_curl(download_dir)
    
    output_dir = Path("/Users/sofijadimitrijevic/csc480/480-ai-finance/processed_data")

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