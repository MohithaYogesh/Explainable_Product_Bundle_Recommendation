"""Streamlit app for explainable product bundle recommendations."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.association_rules import get_top_frequent_itemsets
from src.recommendation_engine import generate_recommendations, get_affordable_bundles
from src.customer_segmentation import assign_segment_labels

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"


def product_lookup_frame(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned product lookup with descriptions keyed by StockCode."""
    if cleaned_df.empty:
        return pd.DataFrame(columns=["StockCode", "Description"])
    return cleaned_df[["StockCode", "Description"]].drop_duplicates().sort_values("StockCode").reset_index(drop=True)


def product_display_label(product_code: str, product_lookup: pd.DataFrame) -> str:
    """Create a readable label using the description and StockCode."""
    if product_code is None or str(product_code).strip() == "":
        return "Unknown product"
    matching = product_lookup[product_lookup["StockCode"].astype(str) == str(product_code)]
    if matching.empty:
        return str(product_code)
    description = str(matching["Description"].iloc[0])
    return f"{description} ({product_code})"


def get_rule_backed_products(rules: pd.DataFrame, product_lookup: pd.DataFrame) -> pd.DataFrame:
    """Return products that appear in Apriori antecedents and therefore have rule-backed recommendations."""
    if rules.empty:
        return pd.DataFrame(columns=["StockCode", "Description", "DisplayLabel"])

    products = sorted({item for _, rule in rules.iterrows() for item in list(rule["antecedents"])})
    rule_products = pd.DataFrame({"StockCode": [str(item) for item in products]})
    rule_products = rule_products.merge(product_lookup[["StockCode", "Description"]], on="StockCode", how="left")
    rule_products["DisplayLabel"] = rule_products.apply(
        lambda row: product_display_label(row["StockCode"], product_lookup), axis=1
    )
    return rule_products.sort_values(["Description", "StockCode"], na_position="last").reset_index(drop=True)


def load_artifacts() -> dict:
    """Load precomputed artifacts or train them if they are missing."""
    required_files = [
        MODEL_DIR / "cleaned_data.joblib",
        MODEL_DIR / "invoice_baskets.joblib",
        MODEL_DIR / "frequent_itemsets.joblib",
        MODEL_DIR / "association_rules.joblib",
        MODEL_DIR / "rfm_table.joblib",
        MODEL_DIR / "kmeans_model.joblib",
        MODEL_DIR / "feature_scaler.joblib",
        MODEL_DIR / "product_prices.joblib",
        MODEL_DIR / "segment_rules.joblib",
    ]

    if not all(path.exists() for path in required_files):
        from train_models import train_and_save_models

        train_and_save_models()

    rfm = assign_segment_labels(joblib.load(MODEL_DIR / "rfm_table.joblib"))
    return {
        "cleaned_df": joblib.load(MODEL_DIR / "cleaned_data.joblib"),
        "baskets": joblib.load(MODEL_DIR / "invoice_baskets.joblib"),
        "frequent_itemsets": joblib.load(MODEL_DIR / "frequent_itemsets.joblib"),
        "rules": joblib.load(MODEL_DIR / "association_rules.joblib"),
        "rfm": rfm,
        "kmeans_model": joblib.load(MODEL_DIR / "kmeans_model.joblib"),
        "scaler": joblib.load(MODEL_DIR / "feature_scaler.joblib"),
        "price_lookup": joblib.load(MODEL_DIR / "product_prices.joblib"),
        "segment_rules": joblib.load(MODEL_DIR / "segment_rules.joblib"),
    }


def build_monthly_summary(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """Generate monthly transaction statistics from the actual InvoiceDate column."""
    if cleaned_df.empty or "InvoiceDate" not in cleaned_df.columns:
        return pd.DataFrame(columns=["Month", "Transactions", "Revenue"])

    valid = cleaned_df[["InvoiceDate", "InvoiceNo", "TotalAmount"]].dropna(subset=["InvoiceDate"]).copy()
    if valid.empty:
        return pd.DataFrame(columns=["Month", "Transactions", "Revenue"])

    valid["Month"] = valid["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        valid.groupby("Month", as_index=False)
        .agg(Transactions=("InvoiceNo", "nunique"), Revenue=("TotalAmount", "sum"))
        .sort_values("Month")
        .reset_index(drop=True)
    )

    return monthly


@st.cache_data
def load_dashboard_data() -> dict:
    """Load the dashboard-ready summary data for the page."""
    artifacts = load_artifacts()
    cleaned_df = artifacts["cleaned_df"]
    rules = artifacts["rules"]
    rfm = artifacts["rfm"]

    product_counts = (
        cleaned_df.groupby("StockCode")["InvoiceNo"]
        .nunique()
        .nlargest(10)
        .rename("Transactions containing product")
        .reset_index()
    )
    product_counts = product_counts.merge(
        artifacts["price_lookup"], on="StockCode", how="left"
    )

    trend = build_monthly_summary(cleaned_df)

    country_sales = cleaned_df.groupby("Country", as_index=False)["TotalAmount"].sum().sort_values("TotalAmount", ascending=False)
    country_sales = country_sales.rename(columns={"TotalAmount": "Revenue (£)"})

    cluster_counts = rfm["Cluster"].value_counts().reset_index()
    cluster_counts.columns = ["Cluster", "Customers"]

    return {
        "cleaned_df": cleaned_df,
        "rules": rules,
        "rfm": rfm,
        "product_counts": product_counts,
        "trend": trend,
        "country_sales": country_sales,
        "cluster_counts": cluster_counts,
    }


def build_dashboard() -> None:
    """Render the overview dashboard with KPIs and visuals."""
    data = load_dashboard_data()
    df = data["cleaned_df"]

    st.title("Explainable Product Bundle Recommendation")
    st.subheader("Retail dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total transactions", f"{df['InvoiceNo'].nunique():,}")
    c2.metric("Unique customers", f"{df['CustomerID'].nunique():,}")
    c3.metric("Unique products", f"{df['StockCode'].nunique():,}")
    c4.metric("Total revenue (£)", f"£{df['TotalAmount'].sum():,.2f}")

    st.markdown("### Most popular products")
    st.dataframe(
        data["product_counts"].head(10)[["Description", "StockCode", "Transactions containing product", "AverageUnitPrice"]].rename(columns={"AverageUnitPrice": "Average unit price (£)"}),
        column_config={"Average unit price (£)": st.column_config.NumberColumn(format="£%.2f")},
        use_container_width=True,
    )

    monthly_trend = data["trend"]
    col_a, col_b = st.columns(2)
    with col_a:
        if monthly_trend.empty:
            st.info("No monthly transaction data is available for the current dataset.")
        else:
            fig_trend = px.line(monthly_trend, x="Month", y="Transactions", title="Monthly transaction volume", labels={"Month": "Month", "Transactions": "Transactions"})
            st.plotly_chart(fig_trend, use_container_width=True)
    with col_b:
        if monthly_trend.empty:
            st.info("No monthly revenue data is available for the current dataset.")
        else:
            fig_revenue = px.line(monthly_trend, x="Month", y="Revenue", title="Monthly revenue (£)", labels={"Month": "Month", "Revenue": "Revenue (£)"})
            fig_revenue.update_yaxes(tickprefix="£", tickformat=",.0f")
            st.plotly_chart(fig_revenue, use_container_width=True)

    if data["country_sales"].empty:
        st.info("No country-level sales data is available for the current dataset.")
    else:
        fig_country = px.bar(data["country_sales"], x="Country", y="Revenue (£)", title="Revenue by country (£)", labels={"Country": "Country", "Revenue (£)": "Revenue (£)"})
        fig_country.update_yaxes(tickprefix="£", tickformat=",.0f")
        st.plotly_chart(fig_country, use_container_width=True)


def build_product_analysis() -> None:
    """Render the product association and frequent itemset section."""
    artifacts = load_artifacts()
    cleaned_df = artifacts["cleaned_df"]
    rules = artifacts["rules"]
    frequent_itemsets = artifacts["frequent_itemsets"]

    product_lookup = product_lookup_frame(cleaned_df)
    st.subheader("Product analysis")
    product_code = st.selectbox("Select a product", options=product_lookup["StockCode"].tolist(), index=0)
    selected_desc = product_lookup[product_lookup["StockCode"] == product_code]["Description"].iloc[0]
    st.caption(f"Selected product: {selected_desc} ({product_code})")

    top_itemsets = get_top_frequent_itemsets(frequent_itemsets, top_n=10)
    if not top_itemsets.empty:
        top_itemsets["support_pct"] = top_itemsets["support"] * 100
        top_itemsets["itemset_text"] = top_itemsets["itemsets"].apply(
            lambda values: ", ".join(
                [f"{item} ({product_lookup[product_lookup['StockCode'].astype(str) == str(item)]['Description'].iloc[0] if not product_lookup[product_lookup['StockCode'].astype(str) == str(item)].empty else item})" for item in sorted(map(str, values))]
            )
        )
        st.dataframe(top_itemsets[["support_pct", "itemset_text"]].rename(columns={"support_pct": "Support (%)", "itemset_text": "Frequent itemset (StockCode and description)"}), use_container_width=True)
    else:
        st.info("No frequent itemsets were generated for the current minimum support threshold.")

    if rules.empty:
        st.warning("No association rules were generated for the active configuration.")
        return

    st.caption(f"Generated {len(rules)} association rules using minimum support {0.01} and confidence {0.2}.")
    strongest_rules = rules.head(10).copy()
    strongest_rules["antecedent_text"] = strongest_rules["antecedents"].apply(
        lambda values: ", ".join(
            [f"{item} ({product_lookup[product_lookup['StockCode'].astype(str) == str(item)]['Description'].iloc[0] if not product_lookup[product_lookup['StockCode'].astype(str) == str(item)].empty else item})" for item in list(values)]
        )
    )
    strongest_rules["consequent_text"] = strongest_rules["consequents"].apply(
        lambda values: ", ".join(
            [f"{item} ({product_lookup[product_lookup['StockCode'].astype(str) == str(item)]['Description'].iloc[0] if not product_lookup[product_lookup['StockCode'].astype(str) == str(item)].empty else item})" for item in list(values)]
        )
    )
    strongest_rules["confidence_pct"] = strongest_rules["confidence"] * 100
    strongest_rules["support_pct"] = strongest_rules["support"] * 100
    strongest_rules["lift_value"] = strongest_rules["lift"]
    st.dataframe(
        strongest_rules[["antecedent_text", "consequent_text", "support_pct", "confidence_pct", "lift_value"]].rename(
            columns={
                "antecedent_text": "Antecedent",
                "consequent_text": "Consequent",
                "support_pct": "Support (%)",
                "confidence_pct": "Confidence (%)",
                "lift_value": "Lift",
            }
        ),
        use_container_width=True,
    )


def build_customer_segmentation() -> None:
    """Display RFM segmentation and cluster stats."""
    artifacts = load_artifacts()
    rfm = artifacts["rfm"]
    cluster_summary = rfm.groupby("Cluster").agg(
        Customers=("CustomerID", "count"),
        **{
            "Average recency (days)": ("Recency", "mean"),
            "Average invoices per customer": ("Frequency", "mean"),
            "Average spend (£)": ("Monetary", "mean"),
            "Dominant customer segment": ("SegmentLabel", lambda values: values.mode().iloc[0] if not values.mode().empty else "Unknown"),
        },
    ).reset_index()

    st.subheader("Customer segmentation")
    st.dataframe(
        cluster_summary,
        column_config={
            "Average recency (days)": st.column_config.NumberColumn(format="%.1f days"),
            "Average invoices per customer": st.column_config.NumberColumn(format="%.1f"),
            "Average spend (£)": st.column_config.NumberColumn(format="£%.2f"),
        },
        use_container_width=True,
    )

    fig_cluster = px.bar(cluster_summary, x="Cluster", y="Customers", color="Cluster", title="Customer count by RFM cluster", labels={"Cluster": "RFM cluster", "Customers": "Customers"})
    st.plotly_chart(fig_cluster, use_container_width=True)

    customer_id = st.number_input("Customer lookup", min_value=0, step=1, value=17850)
    customer_record = rfm[rfm["CustomerID"] == int(customer_id)]
    if customer_record.empty:
        st.info("Customer ID not found in the cleaned dataset.")
    else:
        cluster_id = int(customer_record.iloc[0]["Cluster"])
        customer = customer_record.iloc[0]
        st.write({"CustomerID": int(customer_id), "Cluster": cluster_id, "Customer segment": customer["SegmentLabel"]})
        customer_metrics = pd.DataFrame([{
            "Recency (days)": int(customer["Recency"]),
            "Invoice count": int(customer["Frequency"]),
            "Lifetime spend (£)": float(customer["Monetary"]),
        }])
        st.dataframe(
            customer_metrics,
            column_config={"Lifetime spend (£)": st.column_config.NumberColumn(format="£%.2f")},
            use_container_width=True,
        )


def build_recommendation_section() -> None:
    """Render the product bundle recommendation experience and budget filter."""
    artifacts = load_artifacts()
    cleaned_df = artifacts["cleaned_df"]
    rules = artifacts["rules"]
    rfm = artifacts["rfm"]
    price_lookup = artifacts["price_lookup"]
    segment_rules = artifacts["segment_rules"]

    product_lookup = product_lookup_frame(cleaned_df)
    product_code = st.selectbox("Select a product for recommendation", options=product_lookup["StockCode"].tolist(), index=0)
    product_label = product_display_label(product_code, product_lookup)
    customer_id = st.number_input("Customer ID (optional)", min_value=0, step=1, value=17850)

    st.caption(f"Selected product: {product_label}")
    customer_records = rfm[rfm["CustomerID"] == int(customer_id)]
    customer_record = customer_records.iloc[0] if not customer_records.empty else None
    cluster_id = int(customer_record["Cluster"]) if customer_record is not None else None
    if customer_record is not None:
        st.caption(
            f"Customer RFM status: {customer_record['SegmentLabel']} (K-Means cluster {cluster_id}). "
            "The status describes individual recency, frequency, and monetary behavior; it does not change the cluster assignment."
        )
    st.caption(
        f"Recommendations use matching rules for RFM cluster {cluster_id}, falling back to global rules when that cluster has no rule for the selected product."
        if cluster_id is not None else "No matching customer profile; recommendations use global association rules."
    )
    recommendations = generate_recommendations(
        selected_product=product_code,
        rules_df=rules,
        price_lookup=price_lookup,
        customer_id=int(customer_id) if customer_id else None,
        segment_rules=segment_rules,
        rfm_table=rfm,
        top_n=5,
    )

    if recommendations.empty:
        st.warning(
            f"No complementary product recommendations were found for {product_label} at the current support/confidence thresholds. "
            "This product does not appear in any rule antecedent in the generated Apriori model."
        )
        alternative_products = get_rule_backed_products(rules, product_lookup)
        alternative_products = alternative_products[alternative_products["StockCode"].astype(str) != str(product_code)].head(10)
        if not alternative_products.empty:
            st.info("Try one of these products with valid association rules:")
            st.dataframe(alternative_products[["StockCode", "Description"]], use_container_width=True)
        return

    recommendations = recommendations.copy()
    recommendations["SelectedProductDisplay"] = recommendations["SelectedProduct"].apply(
        lambda code: product_display_label(code, product_lookup)
    )
    recommendations["RecommendedProductDisplay"] = recommendations["RecommendedProduct"].apply(
        lambda code: product_display_label(code, product_lookup)
    )

    recommendation_table = recommendations[["SelectedProductDisplay", "RecommendedProductDisplay", "Support", "Confidence", "Lift", "EstimatedPrice", "RuleSource"]].copy()
    recommendation_table["Support"] *= 100
    recommendation_table["Confidence"] *= 100
    recommendation_table = recommendation_table.rename(columns={
        "SelectedProductDisplay": "Selected product",
        "RecommendedProductDisplay": "Recommended product",
        "Support": "Support (%)",
        "Confidence": "Confidence (%)",
        "Lift": "Lift",
        "EstimatedPrice": "Estimated price (£)",
        "RuleSource": "Rule source",
    })
    st.dataframe(
        recommendation_table,
        column_config={
            "Support (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Confidence (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Lift": st.column_config.NumberColumn(format="%.2f"),
            "Estimated price (£)": st.column_config.NumberColumn(format="£%.2f"),
        },
        use_container_width=True,
    )
    st.caption(
        "Support is the share of transactions containing the complete rule; confidence is the share of antecedent transactions that also contain the recommended product; "
        "lift compares that confidence with the product's baseline transaction rate."
    )
    for _, row in recommendations.iterrows():
        st.markdown(
            f"- **{row['RecommendedProductDisplay']}**: {row['Explanation']} "
            f"Support {float(row['Support']):.2%}, confidence {float(row['Confidence']):.2%}, and lift {float(row['Lift']):.2f}. "
            f"Rule source: {row['RuleSource']}. Historical average price: £{float(row['EstimatedPrice']):.2f}."
        )

    st.markdown("### Budget-based bundle recommendations")
    st.caption("Recommendation order: Lift, then Confidence, then Support. For budget filtering, BundleStrength = Confidence × Lift; affordable bundles are ranked by BundleStrength, then lower estimated total price.")
    budget = st.number_input("Maximum spending budget (£)", min_value=0.0, step=10.0, value=50.0)
    selected_price_rows = price_lookup[price_lookup["StockCode"] == str(product_code)]
    selected_price = float(selected_price_rows["AverageUnitPrice"].iloc[0]) if not selected_price_rows.empty else 0.0
    st.caption(
        f"Bundle total (£) includes the selected product {product_label} at its estimated historical average price of £{selected_price:.2f}, "
        "plus the recommended product at the estimated price shown. All amounts are in GBP."
    )
    affordable = get_affordable_bundles(product_code, recommendations, price_lookup, budget)
    if affordable.empty:
        st.info("No affordable bundle is available within the selected budget.")
    else:
        affordable = affordable.copy()
        affordable["RecommendedProductDisplay"] = affordable["RecommendedProduct"].apply(lambda code: product_display_label(code, product_lookup))
        affordable_table = affordable[["RecommendedProductDisplay", "Support", "Confidence", "Lift", "EstimatedPrice", "BundleTotalPrice", "BundleStrength"]].copy()
        affordable_table["Support"] *= 100
        affordable_table["Confidence"] *= 100
        affordable_table = affordable_table.rename(columns={
            "RecommendedProductDisplay": "Recommended product",
            "Support": "Support (%)",
            "Confidence": "Confidence (%)",
            "EstimatedPrice": "Recommended item price (£)",
            "BundleTotalPrice": "Bundle total (£)",
            "BundleStrength": "Bundle strength (confidence × lift)",
        })
        st.dataframe(
            affordable_table,
            column_config={
                "Support (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "Confidence (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "Lift": st.column_config.NumberColumn(format="%.2f"),
                "Recommended item price (£)": st.column_config.NumberColumn(format="£%.2f"),
                "Bundle total (£)": st.column_config.NumberColumn(format="£%.2f"),
                "Bundle strength (confidence × lift)": st.column_config.NumberColumn(format="%.3f"),
            },
            use_container_width=True,
        )


def build_model_evaluation() -> None:
    """Render model evaluation stats for the association and clustering pipeline."""
    artifacts = load_artifacts()
    rules = artifacts["rules"]
    frequent_itemsets = artifacts["frequent_itemsets"]
    rfm = artifacts["rfm"]
    metadata = joblib.load(MODEL_DIR / "model_metadata.joblib")

    st.subheader("Model evaluation")
    st.write({
        "Apriori minimum support": 0.01,
        "Apriori minimum confidence": 0.2,
        "Frequent itemsets": int(len(frequent_itemsets)),
        "Association rules": int(len(rules)),
        "K-Means cluster count": int(rfm["Cluster"].nunique()),
        "Silhouette score": round(float(metadata["silhouette_score"]), 4),
    })

    if not frequent_itemsets.empty:
        itemset_size = frequent_itemsets["itemsets"].apply(len)
        fig_size = px.histogram(itemset_size, nbins=10, title="Frequent itemset sizes")
        st.plotly_chart(fig_size, use_container_width=True)


def main() -> None:
    """Entry point for the Streamlit user interface."""
    load_artifacts()
    st.set_page_config(page_title="Bundle Recommendation", layout="wide")

    build_dashboard()
    st.markdown("---")
    build_product_analysis()
    st.markdown("---")
    build_customer_segmentation()
    st.markdown("---")
    build_recommendation_section()
    st.markdown("---")
    build_model_evaluation()


if __name__ == "__main__":
    main()
