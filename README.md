# Explainable Product Bundle Recommendation Using Association Rule Mining and Customer Segmentation

## Project introduction
This project builds an explainable retail recommendation system that identifies complementary products using association rule mining and customer segmentation. It is designed for a small machine learning mini-project with a real transaction dataset, a modular Python architecture, and a Streamlit web interface.

## Problem statement
Online retailers need to recommend products that match customer purchasing behavior while also respecting customer segments and budget constraints. A system based on Apriori rules and customer RFM segmentation provides recommendations that are easier to explain than black-box recommenders.

## Objectives
- Inspect the actual retail dataset structure and validate raw transaction quality.
- Clean the dataset and convert invoice-level purchases into transaction baskets.
- Generate frequent itemsets and association rules using the Apriori algorithm.
- Segment customers using RFM values and K-Means clustering.
- Combine segmentation and association rules to produce explainable recommendations.
- Filter recommendations by a maximum spending budget.
- Deliver the pipeline in a user-friendly Streamlit web application.

## Technologies used
- Python
- Pandas
- NumPy
- scikit-learn
- mlxtend
- Streamlit
- Plotly
- Joblib
- OpenPyXL

## Dataset description
The dataset used is the supplied Excel workbook located at `dataset/Online Retail.xlsx`. The raw file contains the actual invoice-level retail sales records. In the source workbook, columns include `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, and `Country`.

During preprocessing, records with missing `CustomerID` values, missing descriptions, cancelled invoices, non-positive quantities, non-positive unit prices, and duplicate rows are removed. The result is a cleaned sales dataset with a valid transaction basket structure.

## System architecture
- `dataset/` contains the raw Excel workbook.
- `notebooks/` contains an exploratory notebook for analysis.
- `models/` stores trained artifacts such as association rules, customer clusters, and price lookup tables.
- `src/` contains the Python modules for preprocessing, Apriori mining, segmentation, and recommendation logic.
- `app.py` is the main Streamlit front-end.
- `train_models.py` trains the models and saves the artifacts.

## Algorithm descriptions
### Apriori association rule mining
The Apriori algorithm identifies frequent item combinations from transactional baskets. In this system, each invoice is treated as a basket of purchased `StockCode` values. Support, confidence, and lift are computed to evaluate the strength and usefulness of product relationships.

### RFM analysis and K-Means clustering
Customers are segmented using recency, frequency, and monetary values. RFM values are scaled before clustering with K-Means. The model is evaluated using the elbow behavior and silhouette score, and cluster labels are interpreted based on customer behavior rather than on the numeric cluster ID alone.

### Explainable recommendation engine
For a selected product, rules whose antecedent contains that product are used to identify complementary products. If the customer's cluster has a matching rule for that product, recommendations use that cluster's transaction rules; otherwise the engine falls back to the global rules. Recommendation rows are ordered by lift, then confidence, then support, and duplicate recommended products keep their strongest matching rule. The budget table computes `BundleStrength = confidence × lift` and sorts affordable bundles by strength descending, then estimated total price ascending. The customer segment labels use recency, frequency, and monetary quartiles; high historic spend does not imply current activity or loyalty.

### Budget-aware bundles
The recommendation engine estimates historical product prices using the average valid `UnitPrice` for each `StockCode` and filters bundles by the maximum spending limit provided by the user.

## Project folder structure
```text
Explainable_Product_Bundle_Recommendation/
├── dataset/
│   └── Online Retail.xlsx
├── notebooks/
│   └── product_bundle_analysis.ipynb
├── models/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── association_rules.py
│   ├── customer_segmentation.py
│   └── recommendation_engine.py
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation instructions
1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install package dependencies:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

## Execution instructions
### Train the models and save artifacts
```bash
python train_models.py
```

### Launch the Streamlit application
```bash
streamlit run app.py
```

## Expected application functionality
- Dashboard with KPI cards and charts from the cleaned dataset.
- Product analysis with top itemsets and strongest rules.
- Customer segmentation with customer-level lookup and cluster summaries.
- Complementary product recommendation results with explanations and metrics.
- Budget-aware bundle filtering based on historical product prices.
- Model evaluation values for the Apriori and clustering stages.

## Notes
The system uses the supplied dataset only and avoids fabricated recommendation values. Metrics such as support, confidence, and lift are derived directly from the dataset during training.
