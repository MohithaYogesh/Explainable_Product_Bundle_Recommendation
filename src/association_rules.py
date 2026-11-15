"""Apriori-based frequent itemset and association rule generation."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules as mlxtend_association_rules
from mlxtend.preprocessing import TransactionEncoder


def build_basket_matrix(baskets: pd.DataFrame) -> pd.DataFrame:
    """Convert invoice baskets into a binary one-hot encoded basket matrix."""
    if baskets.empty:
        return pd.DataFrame()

    transactions = baskets["Products"].tolist()
    encoder = TransactionEncoder()
    encoded_array = encoder.fit(transactions).transform(transactions)
    matrix = pd.DataFrame(encoded_array, columns=encoder.columns_)
    matrix.index = baskets["InvoiceNo"]
    return matrix


def generate_apriori_results(
    baskets: pd.DataFrame,
    min_support: float = 0.01,
    min_confidence: float = 0.2,
    max_len: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Generate frequent itemsets and association rules with configurable thresholds."""
    if baskets.empty or "Products" not in baskets.columns:
        empty_itemsets = pd.DataFrame(columns=["support", "itemsets"])
        empty_rules = pd.DataFrame(columns=[
            "antecedents",
            "consequents",
            "antecedent support",
            "consequent support",
            "support",
            "confidence",
            "lift",
            "leverage",
            "conviction",
        ])
        return {"frequent_itemsets": empty_itemsets, "rules": empty_rules}

    basket_matrix = build_basket_matrix(baskets)
    if basket_matrix.empty or basket_matrix.shape[1] == 0:
        empty_itemsets = pd.DataFrame(columns=["support", "itemsets"])
        empty_rules = pd.DataFrame(columns=[
            "antecedents",
            "consequents",
            "antecedent support",
            "consequent support",
            "support",
            "confidence",
            "lift",
            "leverage",
            "conviction",
        ])
        return {"frequent_itemsets": empty_itemsets, "rules": empty_rules}

    frequent_itemsets = apriori(
        basket_matrix,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
    )

    if frequent_itemsets.empty:
        return {"frequent_itemsets": frequent_itemsets, "rules": pd.DataFrame(columns=[
            "antecedents",
            "consequents",
            "antecedent support",
            "consequent support",
            "support",
            "confidence",
            "lift",
            "leverage",
            "conviction",
        ])}

    rules = mlxtend_association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values(["lift", "confidence", "support"], ascending=[False, False, False]).reset_index(drop=True)

    # Keep a plain list representation for downstream recommendation logic.
    rules["antecedent_list"] = rules["antecedents"].apply(list)
    rules["consequent_list"] = rules["consequents"].apply(list)

    return {"frequent_itemsets": frequent_itemsets, "rules": rules}


def get_top_frequent_itemsets(frequent_itemsets: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the strongest frequent itemsets based on support and size."""
    if frequent_itemsets.empty:
        return frequent_itemsets
    filtered = frequent_itemsets[frequent_itemsets["itemsets"].apply(len) >= 2].copy()
    if filtered.empty:
        return filtered
    return filtered.sort_values(["support", "itemsets"], ascending=[False, True]).head(top_n).reset_index(drop=True)


def get_rules_for_product(rules: pd.DataFrame, selected_product: str) -> pd.DataFrame:
    """Return rules where the selected product appears in the antecedent."""
    if rules.empty:
        return rules.copy()

    matches = []
    for _, rule in rules.iterrows():
        antecedents = set(rule["antecedents"])
        consequents = set(rule["consequents"])
        if selected_product in antecedents and selected_product not in consequents:
            matches.append(rule)

    if not matches:
        return pd.DataFrame(columns=rules.columns)

    return pd.DataFrame(matches).reset_index(drop=True)


def explode_rule_consequents(rules: pd.DataFrame) -> pd.DataFrame:
    """Expand one association rule into one row per recommended product."""
    if rules.empty:
        return pd.DataFrame(columns=[
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift",
            "recommended_product",
        ])

    expanded_rows = []
    for _, rule in rules.iterrows():
        for product in list(rule["consequents"]):
            row = rule.copy()
            row["recommended_product"] = product
            expanded_rows.append(row)

    return pd.DataFrame(expanded_rows).reset_index(drop=True)
