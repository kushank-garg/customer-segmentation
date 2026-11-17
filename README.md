
## Try the Project Online

Run this Customer Segmentation project directly in Google Colab without installing Python.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kushank-garg/customer-segmentation/blob/main/customer_segmentation_demo.ipynb)

The project uses K-Means clustering to identify customer segments from sample data.


# Customer Segmentation using K-Means

A beginner-friendly machine learning project that groups customers with similar characteristics using **K-Means clustering**.

> **Dataset note:** `data/customer_data.csv` is synthetic sample data generated for this portfolio project. It is not real customer or company data.

## Objective
Explore customer groups using:
- Age
- Annual income (in thousands)
- Spending score (1–100)
- Approximate annual store visits

The clusters can help illustrate how businesses might think about different marketing approaches. They are exploratory patterns, not verified customer identities.

## Tech stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib.

## Structure
```text
customer-segmentation/
├── data/customer_data.csv
├── src/segment_customers.py
├── outputs/                 # generated when the script runs
├── requirements.txt
└── README.md
```

## Run locally
Python 3.9+ recommended.

```bash
pip install -r requirements.txt
python src/segment_customers.py
```

Generated outputs:
- `elbow_method.png` – inertia for K values from 2 to 8
- `customer_segments.png` – 2D PCA visualisation
- `segment_profiles.csv` – average feature values and counts per cluster
- `segmented_customers.csv` – customer rows with assigned cluster

## Workflow
1. Load the CSV and inspect its shape and missing values.
2. Select numerical features and fill missing values with the median.
3. Standardize features using `StandardScaler`, since their units and ranges differ.
4. Apply K-Means for K=2 to K=8 and plot inertia (Elbow Method).
5. Fit a demonstration model with K=4 and calculate the silhouette score.
6. Summarize each cluster using average feature values.
7. Use PCA to project the scaled features into two dimensions for visualisation.

## Why K-Means?
K-Means is an unsupervised algorithm. It assigns each row to the nearest cluster centroid and repeatedly updates the centroids to reduce within-cluster distances.

## Limitations
- The dataset is synthetic and intended for learning/demo purposes.
- K=4 is selected because the generated data contains four broad behavior patterns. For real data, compare the elbow plot, silhouette scores, business context and validation before choosing K.
- K-Means is sensitive to scaling, outliers and the chosen K.
- PCA's 2D view loses some information from the original features.

## Future improvements
- Compare silhouette scores across several K values.
- Add recency, frequency and monetary-value (RFM) features.
- Compare K-Means with hierarchical clustering or DBSCAN.
- Test on permissioned, anonymised real-world data.
