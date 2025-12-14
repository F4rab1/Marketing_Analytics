# %%

from pathlib import Path
import dill
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Adjust sys.path to include the project root for src module
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))

# Import utils (ensure src/unimib_snowit_project/utils.py exists)
try:
    import src.unimib_snowit_project.utils as u
except ModuleNotFoundError:
    print("Error: src module not found. Ensure src/unimib_snowit_project/utils.py exists in the project root.")
    raise

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report


# CONFIGURATION

DATA_PKL_DIR = project_root / "data_loaded"  # Correct path to parent directory

USERS_PKL_FILENAME = "users.pkl"
PROFILES_PKL_FILENAME = "profiles.pkl"
CARDS_PKL_FILENAME = "cards.pkl"
ORDERS_PKL_FILENAME = "orders.pkl"
ORDER_DETAILS_PKL_FILENAME = "order_details.pkl"

CHURN_PERIOD_DAYS = 90           # inactivity window that defines churn
CHURN_PROB_THRESHOLD = 0.65      # min churn probability to target
VALUE_PERCENTILE = 0.5           # top X% by monetary value (0.5 = top 50%)
OUTPUT_TARGETS_CSV = "campaign_targets.csv"

# HELPERS

def load_pkl(path: Path):
    with path.open("rb") as fh:
        return dill.load(fh)

# ... (rest of the helpers and main function remain unchanged)

def build_order_kpi(orders_df, order_details_df):
    """Builds an order-level KPI table from raw orders and order_details."""
    print("\n--- Building order_kpi ---")
    
    # Keep only fulfilled items
    order_details_clean = order_details_df[order_details_df["item.status"] == "fulfilled"].copy()
    
    #  Parse dates
    orders_df["createdAt"] = pd.to_datetime(orders_df["createdAt"])
    
    # Join items → orders
    od_orders = order_details_clean.merge(
        orders_df[["order.uid", "user.uid", "createdAt", "source", "tenant"]],
        on="order.uid",
        how="left"
    )
    
    #  Item revenue
    od_orders["item_revenue"] = od_orders["item.amount"].astype(float)
    
    #  Aggregate to order level
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
    
    print("order_kpi shape:", order_kpi.shape)
    return order_kpi


def define_churn_label(order_kpi, churn_period_days=90):
    """
    Define churn label using a proper cut-off:
    - T0 = snapshot_date - churn_period_days
    - churned = no order after T0
    """
    print("\n--- Defining churn label with cut-off ---")
    
    snapshot_date = order_kpi["order_date"].max()
    cutoff_date = snapshot_date - pd.Timedelta(days=churn_period_days)
    
    print("Snapshot date:", snapshot_date)
    print("Feature cut-off date (T0):", cutoff_date)
    
    # Orders BEFORE T0 (for feature engineering)
    orders_before = order_kpi[order_kpi["order_date"] < cutoff_date].copy()
    
    # Last order over entire window (for label)
    last_order_overall = (
        order_kpi
        .groupby("user_uid")["order_date"]
        .max()
        .rename("last_order_date")
        .reset_index()
    )
    
    # churned = last_order_date < T0
    last_order_overall["churned"] = (last_order_overall["last_order_date"] < cutoff_date).astype(int)
    
    print("Churn distribution:\n", last_order_overall["churned"].value_counts())
    return orders_before, cutoff_date, last_order_overall


def build_behavior_features(orders_before, cutoff_date):
    """Builds behavioral features using ONLY data before cutoff_date (no leakage)."""
    print("\n--- Building behavioral features (pre-T0 only) ---")
    
    agg_before = (
        orders_before
        .groupby("user_uid")
        .agg(
            first_order_date=("order_date", "min"),
            last_order_before_T0=("order_date", "max"),
            freq_total=("order.uid", "nunique"),
            monetary_total=("order_revenue", "sum")
        )
        .reset_index()
    )
    
    # Tenure
    agg_before["tenure_days"] = (cutoff_date - agg_before["first_order_date"]).dt.days
    
    # Avg order value
    agg_before["avg_order_value"] = agg_before["monetary_total"] / agg_before["freq_total"]
    
    # Orders in last 30 days before T0
    window_start_30 = cutoff_date - pd.Timedelta(days=30)
    orders_last_30 = (
        orders_before[
            (orders_before["order_date"] >= window_start_30) &
            (orders_before["order_date"] < cutoff_date)
        ]
        .groupby("user_uid")["order.uid"]
        .nunique()
        .rename("orders_last_30d")
        .reset_index()
    )
    
    features_behavior = agg_before.merge(orders_last_30, on="user_uid", how="left")
    features_behavior["orders_last_30d"] = features_behavior["orders_last_30d"].fillna(0)
    
    print("Behavioral feature table shape:", features_behavior.shape)
    return features_behavior


def enrich_with_user_profile(features_behavior, users_df, profiles_df, last_order_overall):
    """Merge behavior + churn labels + users + profile info."""
    print("\n--- Enriching with user & profile info ---")
    
    users_ren = users_df.rename(columns={"user.uid": "user_uid"})
    profiles_ren = profiles_df.rename(columns={"user.uid": "user_uid"})
    
    user_profile = (
        profiles_ren
        .groupby("user_uid", as_index=False)
        .agg(
            main_city=("city", "first"),
            sex=("sex", "first"),
            level=("level", "first")
        )
    )
    
    features_df = (
        features_behavior
        .merge(last_order_overall[["user_uid", "churned"]], on="user_uid", how="inner")
        .merge(users_ren, on="user_uid", how="left")
        .merge(user_profile, on="user_uid", how="left")
    )
    
    print("features_df shape:", features_df.shape)
    return features_df


def build_design_matrix(features_df):
    """Build X_final and y with manual preprocessing (no sklearn ColumnTransformer)."""
    print("\n--- Building X_final and y ---")
    
    # Candidate features
    candidate_num = ["freq_total", "monetary_total", "tenure_days", "avg_order_value", "orders_last_30d"]
    candidate_cat = ["source", "language", "sex", "level", "main_city"]
    
    num_features = [c for c in candidate_num if c in features_df.columns]
    cat_features = [c for c in candidate_cat if c in features_df.columns]
    
    print("Numerical features:", num_features)
    print("Categorical features:", cat_features)
    
    # Clean NA types
    features_df_clean = features_df.replace({pd.NA: np.nan})
    
    # Numeric part
    X_num = features_df_clean[num_features].astype("float64")
    X_num_imputed = X_num.fillna(X_num.median())
    
    scaler = StandardScaler()
    X_num_scaled = pd.DataFrame(
        scaler.fit_transform(X_num_imputed),
        columns=num_features,
        index=features_df_clean.index
    )
    
    # Categorical part
    X_cat = features_df_clean[cat_features].astype("string")
    X_cat_imputed = X_cat.fillna("Missing")
    X_cat_dummies = pd.get_dummies(X_cat_imputed, drop_first=False)
    
    # Final matrix
    X_final = pd.concat([X_num_scaled, X_cat_dummies], axis=1)
    
    y = features_df_clean["churned"].astype(int)
    
    print("X_final shape:", X_final.shape)
    print("Target distribution:\n", y.value_counts())
    
    return X_final, y, num_features, cat_features


def train_churn_model(X_final, y):
    """Train a RandomForest churn model and print evaluation on a hold-out set."""
    print("\n--- Train / test split ---")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_final,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    
    print("\n--- Training Random Forest churn model ---")
    model.fit(X_train, y_train)
    
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    
    auc = roc_auc_score(y_test, y_proba)
    print(f"\nROC-AUC: {auc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    
    # Retrain on full data for campaign scoring
    print("\n--- Retraining model on full data for scoring ---")
    model.fit(X_final, y)
    return model


def select_campaign_targets(features_df, model, X_final, prob_threshold=0.65, value_percentile=0.5):
    """
    Score all customers with churn probability, then select campaign targets:
    - churn_proba >= prob_threshold
    - monetary_total in top value_percentile
    """
    print("\n--- Scoring all customers and selecting campaign targets ---")
    
    churn_proba = model.predict_proba(X_final)[:, 1]
    
    scored = features_df.copy()
    scored["churn_proba"] = churn_proba
    
    # High value cutoff based on monetary_total
    monetary_cutoff = scored["monetary_total"].quantile(value_percentile)
    print(f"Monetary cutoff (top {int((1 - value_percentile)*100)}%): {monetary_cutoff:.2f}")
    
    targets = scored[
        (scored["churn_proba"] >= prob_threshold) &
        (scored["monetary_total"] >= monetary_cutoff)
    ].copy()
    
    print("Number of campaign targets:", len(targets))
    
    # Select columns useful for marketing export
    export_cols = [
        "user_uid", "churn_proba", "monetary_total", "freq_total",
        "tenure_days", "orders_last_30d", "main_city", "sex", "level", "source"
    ]
    export_cols = [c for c in export_cols if c in targets.columns]
    
    return targets[export_cols]


# MAIN EXECUTION

def main():
    print("=== RUNNING DATA-DRIVEN CAMPAIGN PIPELINE ===")
    
    #  Load PKL data
    users_df = load_pkl(DATA_PKL_DIR / USERS_PKL_FILENAME)
    profiles_df = load_pkl(DATA_PKL_DIR / PROFILES_PKL_FILENAME)
    cards_df = load_pkl(DATA_PKL_DIR / CARDS_PKL_FILENAME)
    orders_df = load_pkl(DATA_PKL_DIR / ORDERS_PKL_FILENAME)
    order_details_df = load_pkl(DATA_PKL_DIR / ORDER_DETAILS_PKL_FILENAME)
    
    #  Build order_kpi
    order_kpi = build_order_kpi(orders_df, order_details_df)
    
    # Define churn label
    orders_before, cutoff_date, last_order_overall = define_churn_label(
        order_kpi, churn_period_days=CHURN_PERIOD_DAYS
    )
    
    #  Build behavioral features
    features_behavior = build_behavior_features(orders_before, cutoff_date)
    
    #  Enrich with user + profile & churn label
    features_df = enrich_with_user_profile(
        features_behavior, users_df, profiles_df, last_order_overall
    )
    
    #  Build design matrix
    X_final, y, num_features, cat_features = build_design_matrix(features_df)
    
    #  Train churn model
    model = train_churn_model(X_final, y)
    
    #  Select campaign targets
    targets_df = select_campaign_targets(
        features_df,
        model,
        X_final,
        prob_threshold=CHURN_PROB_THRESHOLD,
        value_percentile=VALUE_PERCENTILE
    )
    
    #  Save targets for marketing
    targets_df.to_csv(OUTPUT_TARGETS_CSV, index=False)
    print(f"\n Campaign targets saved to: {OUTPUT_TARGETS_CSV}")
    print("=== PIPELINE COMPLETED ===")


if __name__ == "__main__":
    main()




# %%
