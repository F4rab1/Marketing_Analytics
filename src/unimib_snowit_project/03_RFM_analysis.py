# %%
from pathlib import Path
import dill
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# parent directory to path to import src module
sys.path.insert(0, str(Path.cwd().parent))

import src.unimib_snowit_project.utils as u

# %% [markdown]
# Loading the data

# %%
DATA_PKL_DIR = 'data_loaded'

USERS_PKL_FILENAME = 'users.pkl'
PROFILES_PKL_FILENAME = 'profiles.pkl'
CARDS_PKL_FILENAME = 'cards.pkl'
ORDERS_PKL_FILENAME = 'orders.pkl'
ORDER_DETAILS_PKL_FILENAME = 'order_details.pkl'

root_dir_path = u.get_root_dir() 

data_pkl_dir_path = root_dir_path / DATA_PKL_DIR
users_pkl_path = data_pkl_dir_path / USERS_PKL_FILENAME
profiles_pkl_path = data_pkl_dir_path / PROFILES_PKL_FILENAME
cards_pkl_path = data_pkl_dir_path / CARDS_PKL_FILENAME
orders_pkl_path = data_pkl_dir_path / ORDERS_PKL_FILENAME
order_details_pkl_path = data_pkl_dir_path / ORDER_DETAILS_PKL_FILENAME

def load_pkl(pkl_path):
    with pkl_path.open('rb') as fh:
        return dill.load(fh)

users_df = load_pkl(users_pkl_path)
profiles_df = load_pkl(profiles_pkl_path)
cards_df = load_pkl(cards_pkl_path)
orders_df = load_pkl(orders_pkl_path)
order_details_df = load_pkl(order_details_pkl_path)


# %% [markdown]
# Building order_kpi from cleaned PKL data

# %%
print("--- Building order_kpi from cleaned PKL data ---")

# 1. Keep only fulfilled items for revenue
order_details_clean = order_details_df[order_details_df["item.status"] == "fulfilled"].copy()

# 2. Parse dates in orders
orders_df["createdAt"] = pd.to_datetime(orders_df["createdAt"])

# 3. Join order_details → orders (to bring user + metadata onto each item)
od_orders = order_details_clean.merge(
    orders_df[["order.uid", "user.uid", "createdAt", "source", "tenant"]],
    on="order.uid",
    how="left"
)

# 4. Basic revenue at item level
od_orders["item_revenue"] = od_orders["item.amount"].astype(float)

# 5. Aggregate to order level (one row per order)
order_kpi = (
    od_orders
    .groupby("order.uid", as_index=False)
    .agg(
        user_uid=("user.uid", "first"),
        order_date=("createdAt", "first"),
        source=("source", "first"),
        tenant=("tenant", "first"),
        order_revenue=("item_revenue", "sum")
    )
)

order_kpi.head()


# %% [markdown]
# RFM MODEL – Deterministic Segmentation
# 
# compute R (Recency), F (Frequency), M (Monetary) per customer using order_kpi.
# 
# Compute R, F, M (Building RFM table from order_kpi)

# %%
print("--- Building RFM table from order_kpi ---")

# Reference date for recency (end of observation window)
snapshot_date = order_kpi["order_date"].max() + pd.Timedelta(days=1)

rfm = (
    order_kpi
    .groupby("user_uid")
    .agg(
        last_order_date=("order_date", "max"),      # most recent purchase
        frequency=("order.uid", "nunique"),         # number of orders
        monetary=("order_revenue", "sum")           # total spending
    )
    .reset_index()
)

# Recency in days
rfm["recency"] = (snapshot_date - rfm["last_order_date"]).dt.days

rfm.head()


# %% [markdown]
# RFM scores (quantile-based 1–5)

# %%
def rfm_score(series, reverse=False):
    # 5 = best, 1 = worst
    quantiles = series.quantile([0.2, 0.4, 0.6, 0.8]).to_dict()
    
    def score(x):
        if reverse:  # for recency (lower is better → higher score)
            if x <= quantiles[0.2]: return 5
            elif x <= quantiles[0.4]: return 4
            elif x <= quantiles[0.6]: return 3
            elif x <= quantiles[0.8]: return 2
            else: return 1
        else:
            if x <= quantiles[0.2]: return 1
            elif x <= quantiles[0.4]: return 2
            elif x <= quantiles[0.6]: return 3
            elif x <= quantiles[0.8]: return 4
            else: return 5
    
    return series.apply(score)

rfm["R_score"] = rfm_score(rfm["recency"], reverse=True)
rfm["F_score"] = rfm_score(rfm["frequency"], reverse=False)
rfm["M_score"] = rfm_score(rfm["monetary"], reverse=False)

rfm["RFM_score"] = rfm["R_score"].astype(str) + rfm["F_score"].astype(str) + rfm["M_score"].astype(str)
rfm.head()


# %% [markdown]
# Define customer segments

# %%
def segment_from_rfm(row):
    r, f, m = row["R_score"], row["F_score"], row["M_score"]
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if r >= 3 and f >= 3:
        return "Loyal"
    if r >= 4 and f <= 2:
        return "New Customers"
    if r <= 2 and f >= 3:
        return "At Risk"
    if r <= 2 and f <= 2 and m <= 2:
        return "Lost"
    return "Others"

rfm["segment"] = rfm.apply(segment_from_rfm, axis=1)
rfm["segment"].value_counts()


# %% [markdown]
# Visualizing Customer Distribution

# %%
# --- Visualization ---
segment_counts = rfm["segment"].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(
    x=segment_counts.index,
    y=segment_counts.values,
    palette="viridis"
)
plt.title("Customer Distribution by RFM Segment", fontsize=16, weight="bold")
plt.xlabel("RFM Segment", fontsize=12)
plt.ylabel("Number of Customers", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


