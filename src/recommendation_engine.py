"""Explainable recommendation engine combining Apriori rules with customer segmentation."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd

from src.association_rules import explode_rule_consequents, get_rules_for_product
from src.preprocessing import build_transaction_baskets


def build_segment_rules(cleaned_df: pd.DataFrame, rfm_table: pd.DataFrame, min_support: float = 0.01, min_confidence: float = 0.2) -> Dict[int, pd.DataFrame]:
    """Generate association rules separately per customer segment when each segment has enough data."""
    from src.association_rules import generate_apriori_results

    segment_rules = {}
    if cleaned_df.empty or rfm_table.empty:
        return segment_rules

    for cluster_id in sorted(rfm_table["Cluster"].unique()):
        cluster_customers = rfm_table[rfm_table["Cluster"] == cluster_id]["CustomerID"].tolist()
        if len(cluster_customers) == 0:
            continue

        segment_transactions = cleaned_df[cleaned_df["CustomerID"].isin(cluster_customers)].copy()
        segment_baskets = build_transaction_baskets(segment_transactions)
        if len(segment_baskets) < 5:
            continue

        rules = generate_apriori_results(segment_baskets, min_support=min_support, min_confidence=min_confidence)
        if not rules["rules"].empty:
            segment_rules[int(cluster_id)] = rules["rules"]

    return segment_rules


def prepare_product_reference(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """Build a product reference table with description and historical average price."""
    product_reference = (
        cleaned_df.groupby("StockCode", as_index=False)
        .agg(
            Description=("Description", "first"),
            AverageUnitPrice=("UnitPrice", "mean"),
            ValidTransactions=("InvoiceNo", "nunique"),
        )
        .sort_values("AverageUnitPrice", ascending=False)
        .reset_index(drop=True)
    )
    return product_reference


def get_customer_cluster(customer_id: Optional[int], rfm_table: pd.DataFrame) -> Optional[int]:
    """Return the cluster of a given customer if present."""
    if customer_id is None or rfm_table.empty:
        return None
    match = rfm_table[rfm_table["CustomerID"] == int(customer_id)]
    if match.empty:
        return None
    return int(match.iloc[0]["Cluster"])


def generate_recommendations(
    selected_product: str,
    rules_df: pd.DataFrame,
    price_lookup: pd.DataFrame,
    customer_id: Optional[int] = None,
    segment_rules: Optional[Dict[int, pd.DataFrame]] = None,
    rfm_table: Optional[pd.DataFrame] = None,
    top_n: int = 5,
) -> pd.DataFrame:
    """Generate explainable recommendations for a selected product, with personalized segment fallback."""
    if selected_product is None or str(selected_product).strip() == "":
        return pd.DataFrame(columns=["SelectedProduct", "RecommendedProduct", "Support", "Confidence", "Lift", "Explanation", "EstimatedPrice", "ProductDescription"])

    active_rules = rules_df
    rule_source = "Global rules"
    if customer_id is not None and segment_rules and rfm_table is not None:
        cluster_id = get_customer_cluster(customer_id, rfm_table)
        if cluster_id is not None and cluster_id in segment_rules:
            segment_candidates = get_rules_for_product(segment_rules[cluster_id], str(selected_product))
            if not segment_candidates.empty:
                active_rules = segment_rules[cluster_id]
                rule_source = f"RFM cluster {cluster_id} rules"

    rule_candidates = get_rules_for_product(active_rules, str(selected_product))
    if rule_candidates.empty:
        return pd.DataFrame(columns=["SelectedProduct", "RecommendedProduct", "Support", "Confidence", "Lift", "Explanation", "EstimatedPrice", "ProductDescription"])

    expanded = explode_rule_consequents(rule_candidates)
    if expanded.empty:
        return pd.DataFrame(columns=["SelectedProduct", "RecommendedProduct", "Support", "Confidence", "Lift", "Explanation", "EstimatedPrice", "ProductDescription"])

    expanded = expanded[expanded["recommended_product"] != str(selected_product)].copy()
    if expanded.empty:
        return pd.DataFrame(columns=["SelectedProduct", "RecommendedProduct", "Support", "Confidence", "Lift", "Explanation", "EstimatedPrice", "ProductDescription"])

    expanded = expanded.sort_values(["lift", "confidence", "support"], ascending=[False, False, False])
    expanded = expanded.drop_duplicates("recommended_product", keep="first").reset_index(drop=True)
    expanded = expanded.head(top_n).copy()

    recommendations = []
    for _, row in expanded.iterrows():
        recommended_product = str(row["recommended_product"])
        product_row = price_lookup[price_lookup["StockCode"] == recommended_product]
        estimated_price = float(product_row["AverageUnitPrice"].iloc[0]) if not product_row.empty else 0.0
        description = product_row["Description"].iloc[0] if not product_row.empty else "Unknown product"

        antecedent = ", ".join(sorted(map(str, row["antecedents"])))
        explanation = f"The rule antecedent {{{antecedent}}} is associated with this product in the historical transactions."
        recommendations.append({
            "SelectedProduct": selected_product,
            "RecommendedProduct": recommended_product,
            "Support": float(row["support"]),
            "Confidence": float(row["confidence"]),
            "Lift": float(row["lift"]),
            "Explanation": explanation,
            "EstimatedPrice": estimated_price,
            "ProductDescription": description,
            "RuleSource": rule_source,
        })

    return pd.DataFrame(recommendations).reset_index(drop=True)


def get_affordable_bundles(
    selected_product: str,
    recommendations: pd.DataFrame,
    price_lookup: pd.DataFrame,
    budget: float,
) -> pd.DataFrame:
    """Filter and rank bundles whose total estimated cost fits within the user budget."""
    if pd.isna(budget) or budget <= 0 or recommendations.empty:
        return pd.DataFrame(columns=["SelectedProduct", "RecommendedProduct", "BundleTotalPrice", "BundleStrength", "EstimatedPrice", "Support", "Confidence", "Lift", "RuleSource", "Explanation"])

    selected_price = price_lookup[price_lookup["StockCode"] == str(selected_product)]["AverageUnitPrice"].tolist()
    selected_price = float(selected_price[0]) if selected_price else 0.0

    bundles = []
    for _, row in recommendations.iterrows():
        recommended_product = str(row["RecommendedProduct"])
        rec_price = float(row["EstimatedPrice"])
        total_price = selected_price + rec_price
        if total_price <= float(budget):
            bundles.append({
                "SelectedProduct": selected_product,
                "RecommendedProduct": recommended_product,
                "BundleTotalPrice": total_price,
                "BundleStrength": float(row["Confidence"]) * float(row["Lift"]),
                "EstimatedPrice": rec_price,
                "Support": float(row["Support"]),
                "Confidence": float(row["Confidence"]),
                "Lift": float(row["Lift"]),
                "RuleSource": row.get("RuleSource", "Global rules"),
                "Explanation": row["Explanation"],
            })

    if not bundles:
        return pd.DataFrame(columns=["SelectedProduct", "RecommendedProduct", "BundleTotalPrice", "BundleStrength", "EstimatedPrice", "Support", "Confidence", "Lift", "RuleSource", "Explanation"])

    result = pd.DataFrame(bundles)
    result = result.sort_values(["BundleStrength", "BundleTotalPrice"], ascending=[False, True]).reset_index(drop=True)
    return result
