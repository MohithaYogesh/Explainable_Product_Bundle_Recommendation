"""Customer segmentation via RFM analysis and K-Means clustering."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def compute_rfm_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create an RFM table using recency, frequency, and monetary values per customer."""
    if df.empty:
        return pd.DataFrame(columns=["CustomerID", "Recency", "Frequency", "Monetary"])

    latest_date = df["InvoiceDate"].max().normalize()
    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda values: (latest_date - values.max().normalize()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalAmount", "sum"),
        )
        .reset_index()
    )
    return rfm


def evaluate_cluster_counts(rfm: pd.DataFrame, max_k: int = 10) -> Dict[str, Dict[int, float]]:
    """Evaluate K values using inertia and silhouette score to support cluster selection."""
    if rfm.empty or len(rfm) < 3:
        return {"inertia": {}, "silhouette": {}}

    max_k = min(max_k, len(rfm) - 1)
    if max_k < 2:
        return {"inertia": {}, "silhouette": {}}

    transformed = rfm[["Recency", "Frequency", "Monetary"]].copy()
    transformed["Frequency"] = np.log1p(transformed["Frequency"])
    transformed["Monetary"] = np.log1p(transformed["Monetary"])

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(transformed)

    inertia = {}
    silhouette = {}
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(x_scaled)
        inertia[k] = model.inertia_
        if len(np.unique(labels)) > 1:
            silhouette[k] = silhouette_score(x_scaled, labels)
        else:
            silhouette[k] = 0.0

    return {"inertia": inertia, "silhouette": silhouette}


def interpret_segment(
    row: pd.Series,
    recency_q25: float,
    recency_q75: float,
    freq_q25: float,
    freq_q75: float,
    mon_q25: float,
    mon_q75: float,
    freq_q50: float,
) -> str:
    """Add segment descriptions based on actual RFM behaviour rather than cluster numbers."""
    recency = row["Recency"]
    frequency = row["Frequency"]
    monetary = row["Monetary"]

    if frequency >= freq_q75 and monetary >= mon_q75:
        if recency <= recency_q25:
            return "Recently Active High-Value Loyal"
        if recency >= recency_q75:
            return "Historically High-Value, Inactive"
        return "High-Value Repeat Buyers"
    if recency >= recency_q75:
        return "Inactive / At-Risk"
    if recency <= recency_q25 and frequency >= freq_q50:
        return "Recently Active Repeat Buyers"
    if monetary <= mon_q25 and frequency <= freq_q25:
        return "Low-Value Infrequent"
    return "Occasional Buyers"


def assign_segment_labels(rfm: pd.DataFrame) -> pd.DataFrame:
    """Assign behavior labels using dataset-level RFM quartiles, including recency."""
    result = rfm.copy()
    if result.empty:
        result["SegmentLabel"] = pd.Series(dtype=str)
        return result

    recency_q25, recency_q75 = result["Recency"].quantile([0.25, 0.75])
    freq_q25, freq_q50, freq_q75 = result["Frequency"].quantile([0.25, 0.5, 0.75])
    mon_q25, mon_q75 = result["Monetary"].quantile([0.25, 0.75])
    result["SegmentLabel"] = result.apply(
        lambda row: interpret_segment(
            row, recency_q25, recency_q75, freq_q25, freq_q75, mon_q25, mon_q75, freq_q50
        ),
        axis=1,
    )
    return result


def fit_customer_segmentation(df: pd.DataFrame, max_k: int = 10) -> Dict[str, object]:
    """Fit K-Means clustering after RFM scaling and return customer assignments and metrics."""
    rfm = compute_rfm_table(df)
    if rfm.empty:
        return {"rfm": rfm, "model": None, "scaler": None, "cluster_summary": pd.DataFrame(), "silhouette_score": 0.0, "best_k": 1}

    rfm["Frequency_log"] = np.log1p(rfm["Frequency"])
    rfm["Monetary_log"] = np.log1p(rfm["Monetary"])
    features = rfm[["Recency", "Frequency_log", "Monetary_log"]]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(features)

    cluster_metrics = evaluate_cluster_counts(rfm, max_k=max_k)
    silhouette_values = cluster_metrics["silhouette"]
    if not silhouette_values:
        best_k = 2
        silhouette_score_value = 0.0
    else:
        best_k = max(silhouette_values, key=silhouette_values.get)
        silhouette_score_value = silhouette_values[best_k]

    model = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    labels = model.fit_predict(x_scaled)
    rfm["Cluster"] = labels

    rfm = assign_segment_labels(rfm)

    cluster_summary = (
        rfm.groupby("Cluster")
        .agg(
            Customers=("CustomerID", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
            Dominant_Segment=("SegmentLabel", lambda values: values.mode().iloc[0] if len(values.mode()) else "Unknown"),
        )
        .reset_index()
    )

    return {
        "rfm": rfm,
        "model": model,
        "scaler": scaler,
        "cluster_summary": cluster_summary,
        "silhouette_score": silhouette_score_value,
        "best_k": best_k,
        "cluster_metrics": cluster_metrics,
    }
