"""Train and save the product recommendation pipeline artifacts."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.association_rules import generate_apriori_results
from src.customer_segmentation import fit_customer_segmentation
from src.preprocessing import build_transaction_baskets, clean_data, get_cleaning_summary, inspect_dataset
from src.recommendation_engine import build_segment_rules, prepare_product_reference

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "Online Retail.xlsx"
MODEL_DIR = ROOT / "models"


def train_and_save_models() -> dict:
    """Load, preprocess, train, and save the models required by the application."""
    raw_df = pd.read_excel(DATA_PATH)
    dataset_report = inspect_dataset(raw_df)
    cleaned_df = clean_data(raw_df)
    baskets = build_transaction_baskets(cleaned_df)

    apriori_results = generate_apriori_results(baskets, min_support=0.01, min_confidence=0.2)
    segmentation = fit_customer_segmentation(cleaned_df, max_k=10)
    rfm_table = segmentation["rfm"]
    price_lookup = prepare_product_reference(cleaned_df)
    segment_rules = build_segment_rules(cleaned_df, rfm_table, min_support=0.01, min_confidence=0.2)

    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(raw_df, MODEL_DIR / "raw_data.joblib")
    joblib.dump(cleaned_df, MODEL_DIR / "cleaned_data.joblib")
    joblib.dump(baskets, MODEL_DIR / "invoice_baskets.joblib")
    joblib.dump(apriori_results["frequent_itemsets"], MODEL_DIR / "frequent_itemsets.joblib")
    joblib.dump(apriori_results["rules"], MODEL_DIR / "association_rules.joblib")
    joblib.dump(rfm_table, MODEL_DIR / "rfm_table.joblib")
    joblib.dump(segmentation["model"], MODEL_DIR / "kmeans_model.joblib")
    joblib.dump(segmentation["scaler"], MODEL_DIR / "feature_scaler.joblib")
    joblib.dump(price_lookup, MODEL_DIR / "product_prices.joblib")
    joblib.dump(segment_rules, MODEL_DIR / "segment_rules.joblib")
    joblib.dump(
        {
            "dataset_report": dataset_report,
            "cleaning_summary": get_cleaning_summary(raw_df, cleaned_df),
            "cluster_summary": segmentation["cluster_summary"],
            "best_k": segmentation["best_k"],
            "silhouette_score": segmentation["silhouette_score"],
            "apriori_min_support": 0.01,
            "apriori_min_confidence": 0.2,
        },
        MODEL_DIR / "model_metadata.joblib",
    )

    return {
        "dataset_report": dataset_report,
        "cleaned_shape": cleaned_df.shape,
        "frequent_itemsets": apriori_results["frequent_itemsets"],
        "rules": apriori_results["rules"],
        "rfm": rfm_table,
        "cluster_summary": segmentation["cluster_summary"],
        "silhouette_score": segmentation["silhouette_score"],
    }


if __name__ == "__main__":
    train_and_save_models()
    print("Training completed and all model artifacts were saved to the models directory.")
