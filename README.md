# UNIMIB SNOWIT PROJECT
This project is prepared using the package manager PDM (https://pdm-project.org/).<br>
It is recommended to use a virtual environment, it is used Conda (https://docs.anaconda.com/free/miniconda/miniconda-install/).<br>
It is recommended to use an IDE for developing your code, it is used Visual Studio Code (https://code.visualstudio.com/).<br>
It is recommended to use git for version control (https://git-scm.com/).<br>

Please update the ```pyproject.toml``` file accordingly.<br>


To setup the project (after installed pdm and conda) follow these step:

# 1. Environment Setup
After installing Conda and pdm, set up the environment:

**Create and activate environment**
```conda create -n <ENV_NAME> python=3.12```
```conda activate <ENV_NAME>```

**Go to project folder (where pyproject.toml lives)**
```cd <PATH-TO-PROJECT>/unimib_snowit_project```

**Install Python dependencies**
```pdm install```

**After installing dependencies, start Jupyter inside the same environment:**
```pdm run jupyter notebook```

# 2. Execution Order of Notebooks
**First go to notebooks folder**

**2.1 Downloading raw datasets**

**Notebook: 00_download_data.ipynb**

```run all cells```

**What it does:**
Downloads all Snowit project CSV files from Google Drive into data_input/ (users.csv, orders.csv, order_details.csv, reviews.csv, reviews_labelled.csv, etc.) and ensures the folder exists.

**Check:** After running, verify that data_input/ contains all expected .csv files.

**2.2 ETL – Build cleaned PKL datasets**

**Notebook: 01_read_data.ipynb**

```run all cells```

**What it does:**
Load raw CSVs from data_input/ for:
users, profiles, cards, orders, order_details, reviews, reviews_labelled.
Clean & normalize columns:
Parse dates (createdAt, assignedAt, birthday, item.date, etc.).
Standardize text (lowercase cities, sources, statuses, etc.).
Convert JSON-like strings into Python lists (e.g. favouriteZones, types).
Primary & foreign key checks (combined):
Validate uniqueness and non-nullness for keys like user.uid, profile.uid, card.uid, order.uid, item.uid, review.uid, labelled_review.uid.
Enforce referential integrity (e.g. drop profiles whose user.uid does not exist in users; keep only order_details with matching order.uid; filter inconsistent cards).
Business-rule filters:
Keep only valid item.status values (e.g. ok, fulfilled, processing).
Remove negative or missing item.amount.
For cards, keep the last assignment per card.uid and drop empty “not-assigned” rows.
Save cleaned datasets to PKL:
Writes users.pkl, profiles.pkl, cards.pkl, orders.pkl, order_details.pkl, reviews.pkl, reviews_labelled.pkl into data_loaded/.

**Check:**
Make sure data_loaded/ exists and all .pkl files are created without errors

**2.3 Customer & Revenue Analytics (Geography, Behavior, Products & Language)**

**Notebook: 02_data_analysis.ipynb**

```run all cells```

**What it does:**
Reads the cleaned users.pkl dataset previously generated during ETL.
Extracts the language attribute from user profiles to understand customer platform preferences.
Counts occurrences of each language and selects the top 10 most represented ones.
Visualizes the distribution using a bar chart for intuitive comparison across languages.
Purpose & Business Value:
Identifies dominant languages among Snowit’s customer base.
Supports multilingual platform decisions (e.g., UI language priorities, customer support localization).
Helps design targeted marketing communication aligned to user language preferences.
Highlights minority language groups that may require additional UX improvements or campaign customization.

**Check:**
Make sure visualizations rendered correctly

**2.4 RFM Segmentation**

**Notebook: 03_RFM_analysis.ipynb**

```run all cells```

**Purpose**
Transform order history into Recency–Frequency–Monetary metrics and segment customers based on value and engagement.
Main Steps
Build order-level revenue KPIs using fulfilled items only.
Compute Recency, Frequency, and Monetary per user.
Assign RFM scores (1–5) using quantiles and generate segment labels.
Visualize customer distribution across segments.
Business Outcome
Identifies high-value, loyal, new, at-risk, and lost customers to drive targeted retention and upsell strategies.

**Check**
RFM table computed without nulls.
Segment counts plotted successfully.

**2.5 Churn Modeling (ML)**

**Notebook: 04_Churn_prediction_model.ipynb**

```run all cells```

This notebook builds a churn prediction model using transactional history. A 90-day inactivity rule defines the churn label, ensuring no data leakage by separating pre-cutoff behavior from post-cutoff outcomes. It engineers features such as frequency, monetary value, tenure, and last-30-day activity, then merges demographic attributes like language, source, and level.
After preprocessing (scaling numeric features and encoding categorical ones), Logistic Regression and Random Forest models are trained and evaluated using ROC-AUC and classification metrics. Finally, churn distribution is visualized to verify class balance and correctness of the labeling.

**Check:** churn labels match the cutoff logic and feature tables contain no missing values after imputation.

**2.6 Sentiment Analysis Model**

**Notebook: 05_sentiment_analysis.ipynb**

```run all cells```

This notebook builds a text-based sentiment classifier using labelled customer reviews. It loads raw and labelled datasets, cleans text by removing URLs, HTML, special characters, and normalizing punctuation. TF-IDF vectorization converts review text into numerical features, and a Logistic Regression model is trained and evaluated using a stratified train/test split.
Model performance is assessed via classification metrics and a confusion matrix to ensure correct label prediction. Finally, the trained model is applied to all unlabelled customer reviews, generating a new column sentiment_pred for downstream analytics and campaign insights.
Check: confusion matrix displays correctly; the sentiment_pred column exists and contains only expected classes (positive/neutral/negative).

**2.7 Data-Driven Marketing Campaign**

**Notebook: 06_Data_driven_marketing_campagn.ipynb**

```run all cells```

This notebook operationalizes churn prediction into an actionable marketing pipeline. First, it rebuilds order‐level KPIs by merging orders and fulfilled order items, ensuring accurate revenue per customer. Then, churn is defined with a strict cut-off window (90 days), allowing the model to distinguish active from inactive users without temporal leakage.
Customer behavior before the cut-off is transformed into features such as frequency, monetary value, tenure, and recent activity. These are enriched with basic profile attributes (city, sex, level, and registration source) to support more nuanced targeting. A Random Forest classifier is trained and evaluated, after which the model is retrained on all data to score every user.
Finally, high-risk and high-value customers are selected according to probability and monetary thresholds and exported as a CSV for targeted retention campaigns. The output is a ready-to-use prioritized list of users most worth engaging.



