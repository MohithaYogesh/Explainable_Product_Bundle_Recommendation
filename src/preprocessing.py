"""Data loading and preprocessing utilities for the product bundle recommendation project."""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


def inspect_dataset(df: pd.DataFrame) -> Dict[str, object]:
    """Return the main raw dataset statistics for validation and reporting."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def normalize_description(description: object) -> str:
    """Clean product description text while preserving the original StockCode identity."""
    if pd.isna(description):
        return ""
    text = str(description).strip()
    text = " ".join(text.split())
    return text


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the dataset cleaning rules required for retail purchase analysis."""
    if df is None or df.empty:
        raise ValueError("The dataset is empty or missing.")

    cleaned = df.copy()

    # Remove missing customer IDs or descriptions.
    cleaned = cleaned[cleaned["CustomerID"].notna()].copy()
    cleaned = cleaned[cleaned["Description"].notna()].copy()

    # Remove cancelled invoices and invalid transactions.
    cleaned = cleaned[~cleaned["InvoiceNo"].astype(str).str.startswith("C", na=False)].copy()
    cleaned = cleaned[cleaned["Quantity"] > 0].copy()
    cleaned = cleaned[cleaned["UnitPrice"] > 0].copy()

    # Remove duplicate records.
    cleaned = cleaned.drop_duplicates().copy()

    # Standardize date and description fields.
    cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"], errors="coerce")
    cleaned = cleaned[cleaned["InvoiceDate"].notna()].copy()
    cleaned["Description"] = cleaned["Description"].map(normalize_description)
    cleaned = cleaned[cleaned["Description"].str.strip() != ""].copy()

    # Ensure the product identifier remains the StockCode even when description text is inconsistent.
    cleaned["StockCode"] = cleaned["StockCode"].astype(str).str.strip()
    cleaned["Country"] = cleaned["Country"].astype(str).str.strip()
    cleaned["CustomerID"] = cleaned["CustomerID"].astype(int)

    # Business metric for basket-level analysis.
    cleaned["TotalAmount"] = cleaned["Quantity"] * cleaned["UnitPrice"]

    return cleaned


def get_cleaning_summary(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """Create a before-and-after dataset summary for display in the app and notebook."""
    summary = {
        "metric": [
            "Rows",
            "Unique invoices",
            "Unique customers",
            "Unique products",
            "Missing CustomerID",
            "Missing Description",
            "Cancelled invoices",
            "Negative/zero quantity",
            "Negative/zero unit price",
            "Duplicate rows",
        ],
        "before": [
            len(raw_df),
            raw_df["InvoiceNo"].nunique(),
            raw_df["CustomerID"].nunique(),
            raw_df["StockCode"].nunique(),
            int(raw_df["CustomerID"].isna().sum()),
            int(raw_df["Description"].isna().sum()),
            int(raw_df["InvoiceNo"].astype(str).str.startswith("C").sum()),
            int((raw_df["Quantity"] <= 0).sum()),
            int((raw_df["UnitPrice"] <= 0).sum()),
            int(raw_df.duplicated().sum()),
        ],
        "after": [
            len(cleaned_df),
            cleaned_df["InvoiceNo"].nunique(),
            cleaned_df["CustomerID"].nunique(),
            cleaned_df["StockCode"].nunique(),
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    }
    return pd.DataFrame(summary)


def build_transaction_baskets(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """Group products by invoice to create transaction baskets as the analysis unit."""
    baskets = (
        cleaned_df.groupby("InvoiceNo", as_index=False)
        .agg(
            CustomerID=("CustomerID", "first"),
            InvoiceDate=("InvoiceDate", "min"),
            TotalAmount=("TotalAmount", "sum"),
            Products=("StockCode", lambda values: list(dict.fromkeys(values))),
        )
        .sort_values("InvoiceDate")
        .reset_index(drop=True)
    )
    baskets["ProductCount"] = baskets["Products"].apply(len)
    return baskets


def save_cleaned_dataset(cleaned_df: pd.DataFrame, output_path: str) -> str:
    """Persist the cleaned dataset for downstream model generation and app use."""
    cleaned_df.to_csv(output_path, index=False)
    return output_path
