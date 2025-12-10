# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import dill

# --- Configuration & Setup ---

# Set plotting style
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.figsize"] = (14, 7)
plt.rcParams["figure.dpi"] = 100

# Define paths based on your project structure
# NOTE: The cleaning script saves files to 'data_loaded'
PROCESSED_DATA_DIR = Path("../../data_loaded")
OUTPUT_DIR = Path("../../reports/figures/eda")

# Ensure the output directory for plots exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Cleaned data will be loaded from: {PROCESSED_DATA_DIR.resolve()}")
print(f"Plots will be saved to: {OUTPUT_DIR.resolve()}")

# %%
def load_cleaned_datasets(data_directory: Path) -> dict[str, pd.DataFrame]:
    """Loads all .pkl files from the processed data directory with robust error handling."""
    print("\n--- 2. Loading Cleaned Datasets ---")
    files_to_load = [
        "users.pkl",
        "orders.pkl",
        "order_details.pkl",
        "profiles.pkl",
        "reviews.pkl",
    ]
    dataframes = {}
    all_files_found = True
    for filename in files_to_load:
        file_path = data_directory / filename
        if file_path.exists():
            print(f"Loading {filename}...")
            with open(file_path, "rb") as f:
                df_name = filename.split('.')[0]
                dataframes[df_name] = dill.load(f)
        else:
            print(f"❌ ERROR: Cleaned file '{filename}' not found in '{data_directory.resolve()}'")
            all_files_found = False
            
    if all_files_found:
        print("✅ All cleaned datasets loaded successfully.\n")
    else:
        print("\n⚠️ Please run the data cleaning script (01_read_data.py) to generate the missing .pkl files.")
        
    return dataframes

data = load_cleaned_datasets(PROCESSED_DATA_DIR)

# %%
print("\n--- Inspecting the 'users' DataFrame ---")

if 'users' in data:
    users_df = data['users']
    print("Displaying the first 5 rows of the 'users' dataset:")
    print(users_df.head())
else:
    print("⚠️ 'users' DataFrame not found in the loaded data.")

# %%
print("\n--- 3. Analyzing Customer Purchase Frequency ---")

if 'orders' in data and 'users' in data:
    orders_df = data['orders']
    users_df = data['users']

    # Merge orders with user data to get customer information for each order
    customer_orders_df = pd.merge(
        orders_df,
        users_df,
        on="user.uid",
        how="left"
    )

    # Count the number of orders per user
    orders_per_customer = customer_orders_df.dropna(subset=['user.uid'])\
        ['user.uid'].value_counts()

    # Categorize customers
    one_time_buyers = (orders_per_customer == 1).sum()
    repeat_customers = (orders_per_customer > 1).sum()
    total_customers = len(orders_per_customer)

    # --- Print the results ---
    print(f"Total customers with at least one order: {total_customers}")
    print(f"Number of one-time buyers: {one_time_buyers} ({one_time_buyers/total_customers:.1%})")
    print(f"Number of repeat customers: {repeat_customers} ({repeat_customers/total_customers:.1%})")

    # --- Plot the results ---
    plt.figure(figsize=(8, 8))
    plt.pie(
        [one_time_buyers, repeat_customers],
        labels=['One-Time Buyers', 'Repeat Customers'],
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette("viridis", 2)
    )
    plt.title('Proportion of One-Time vs. Repeat Customers', fontsize=16, weight='bold')
    plt.savefig(OUTPUT_DIR / "one_time_vs_repeat_customers.png")
    print(f"\n✅ Plot saved to {OUTPUT_DIR / 'one_time_vs_repeat_customers.png'}")
    plt.show()

else:
    print("⚠️ 'orders' or 'users' DataFrame not found. Please ensure data is loaded correctly.")

# %%
print("\n--- 4. Deeper Geographic Analysis ---")

if 'customer_orders_df' in locals():
    # a. Customer Levels by City
    customers_by_city = customer_orders_df.groupby('city')['user.uid'].nunique().sort_values(ascending=False).head(10)

    print("\nTop 10 Cities by Number of Unique Customers:")
    print(customers_by_city)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=customers_by_city.values, y=customers_by_city.index, hue=customers_by_city.index, palette="viridis", legend=False)
    plt.title('Top 10 Cities by Unique Customers', fontsize=16, weight='bold')
    plt.xlabel('Number of Unique Customers', fontsize=12)
    plt.ylabel('City', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_10_cities_by_customers.png")
    print(f"\n✅ Plot saved to {OUTPUT_DIR / 'top_10_cities_by_customers.png'}")
    plt.show()

    # b. Orders per Customer by City
    city_summary = customer_orders_df.groupby('city').agg(
        total_orders=('order.uid', 'count'),
        unique_customers=('user.uid', 'nunique')
    ).dropna()
    city_summary['orders_per_customer'] = city_summary['total_orders'] / city_summary['unique_customers']
    
    # Filter for cities with a meaningful number of customers (e.g., > 50)
    top_cities_by_loyalty = city_summary[city_summary['unique_customers'] > 50].sort_values('orders_per_customer', ascending=False).head(10)

    print("\nTop 10 Cities by Average Orders Per Customer (Cities with >50 customers):")
    print(top_cities_by_loyalty['orders_per_customer'].round(2))

    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_cities_by_loyalty['orders_per_customer'].values, y=top_cities_by_loyalty.index, hue=top_cities_by_loyalty.index, palette="plasma", legend=False)
    plt.title('Top 10 Cities by Customer Order Frequency', fontsize=16, weight='bold')
    plt.xlabel('Average Orders Per Customer', fontsize=12)
    plt.ylabel('City', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_10_cities_by_order_frequency.png")
    print(f"\n✅ Plot saved to {OUTPUT_DIR / 'top_10_cities_by_order_frequency.png'}")
    plt.show()

    print("\n💡 Business Insight:")
    print("While Milan has the most customers, other cities may have more loyal (frequent) customers. This can identify markets ripe for loyalty programs or where customers are highly engaged.")

else:
    print("⚠️ 'customer_orders_df' not found. Please run the merging cell first.")

# %%
print("\n--- 5. Revenue and Orders by Source ---")

# Assuming a 'source' column exists in the orders data
if 'customer_orders_df' in locals() and 'source' in customer_orders_df.columns:
    # a. Revenue by Source
    revenue_by_source = customer_orders_df.groupby('source')['amount_paid_eur'].sum().sort_values(ascending=False)

    print("\nTotal Revenue by Source (€):")
    print(revenue_by_source.round(2))

    plt.figure(figsize=(10, 7))
    revenue_by_source.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=sns.color_palette("magma", len(revenue_by_source)))
    plt.title('Revenue Contribution by Source', fontsize=16, weight='bold')
    plt.ylabel('') # Hide the y-label for pie charts
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "revenue_by_source.png")
    print(f"\n✅ Plot saved to {OUTPUT_DIR / 'revenue_by_source.png'}")
    plt.show()

    print("\n💡 Business Insight:")
    print("This chart reveals which channels (e.g., 'web', 'mobile_app', 'partner_api') are most effective at generating revenue. Resources can be allocated to optimize high-performing channels or improve underperforming ones.")

else:
    print("⚠️ 'source' column not found in the data. This analysis cannot be performed.")

# %%
print("\n--- 6. Order Volume by Time ---")

if 'orders' in data:
    orders_df = data['orders'].copy()
    # Ensure order_date is datetime and set as index
    orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
    orders_df.set_index('order_date', inplace=True)
createdAtTime

    # a. Orders per Month
    orders_by_month = orders_df.resample('ME').size()
    
    print("\nTotal Orders per Month:")
    print(orders_by_month)

    plt.figure(figsize=(12, 7))
    orders_by_month.plot(kind='bar', color=sns.color_palette("viridis")[0])
    plt.title('Total Orders per Month', fontsize=16, weight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Orders', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "orders_per_month.png")
    print(f"\n✅ Plot saved to {OUTPUT_DIR / 'orders_per_month.png'}")
    plt.show()

    # b. Orders per Day of the Week
    orders_df['day_of_week'] = orders_df.index.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    orders_by_day = orders_df['day_of_week'].value_counts().reindex(day_order)

    print("\nTotal Orders per Day of the Week:")
    print(orders_by_day)

    plt.figure(figsize=(12, 7))
    sns.barplot(x=orders_by_day.index, y=orders_by_day.values, hue=orders_by_day.index, palette="plasma", legend=False)
    plt.title('Total Orders by Day of the Week', fontsize=16, weight='bold')
    plt.xlabel('Day of the Week', fontsize=12)
    plt.ylabel('Total Number of Orders', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "orders_per_day_of_week.png")
    print(f"\n✅ Plot saved to {OUTPUT_DIR / 'orders_per_day_of_week.png'}")
    plt.show()

    print("\n💡 Business Insight:")
    print("Monthly analysis highlights seasonality (e.g., peaks in winter for a ski business), crucial for inventory and staffing. Daily analysis reveals weekly customer habits, informing the best days for marketing campaigns, promotions, or customer support availability.")

else:
    print("⚠️ 'orders' DataFrame not found. Please ensure data is loaded correctly.")
