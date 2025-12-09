

# %%
import re
import sys
from pathlib import Path

import dill
import pandas as pd

# parent directory to path to import src module
sys.path.insert(0, str(Path.cwd().parent))

import src.unimib_snowit_project.utils as u

# %% [markdown]
# # Setup

# %%
# Base Params

DATA_IN_DIR = '../../data_input'

USERS_IN_FILENAME = 'users.csv'
PROFILES_IN_FILENAME = 'profiles.csv'
CARDS_IN_FILENAME = 'cards.csv'
ORDERS_IN_FILENAME = 'orders.csv'
ORDER_DETAILS_IN_FILENAME = 'order_details.csv'
REVIEWS_IN_FILENAME = 'reviews.csv'                
REVIEWS_LABELLED_IN_FILENAME = 'reviews_labelled.csv' 

DATA_PKL_DIR = 'data_loaded'

USERS_PKL_FILENAME = 'users.pkl'
PROFILES_PKL_FILENAME = 'profiles.pkl'
CARDS_PKL_FILENAME = 'cards.pkl'
ORDERS_PKL_FILENAME = 'orders.pkl'
ORDER_DETAILS_PKL_FILENAME = 'order_details.pkl'
REVIEWS_PKL_FILENAME = 'reviews.pkl'                 
REVIEWS_LABELLED_PKL_FILENAME = 'reviews_labelled.pkl'  

NA_VALUES = ['', ' ', '""',
             '#N/A', '#N/A N/A', '#NA', 'N/A', '<NA>', 'n/a', # 'NA',
             '-1.#IND', '1.#IND',
             '-1.#QNAN', '-NaN', '-nan', '-NAN', '1.#QNAN', 'NaN', 'nan', 'NAN',
             'NULL', 'Null', 'null',
             'NONE', 'None', 'none',
             ]

# %%
# Base paths

root_dir_path = u.get_root_dir()

data_in_dir_path = root_dir_path.joinpath(DATA_IN_DIR)
users_in_path = data_in_dir_path.joinpath(USERS_IN_FILENAME)
profiles_in_path = data_in_dir_path.joinpath(PROFILES_IN_FILENAME)
cards_in_path = data_in_dir_path.joinpath(CARDS_IN_FILENAME)
orders_in_path = data_in_dir_path.joinpath(ORDERS_IN_FILENAME)
order_details_in_path = data_in_dir_path.joinpath(ORDER_DETAILS_IN_FILENAME)
reviews_in_path = data_in_dir_path.joinpath(REVIEWS_IN_FILENAME)               
reviews_labelled_in_path = data_in_dir_path.joinpath(REVIEWS_LABELLED_IN_FILENAME)

data_pkl_dir_path = root_dir_path.joinpath(DATA_PKL_DIR)
users_pkl_path = data_pkl_dir_path.joinpath(USERS_PKL_FILENAME)
profiles_pkl_path = data_pkl_dir_path.joinpath(PROFILES_PKL_FILENAME)
cards_pkl_path = data_pkl_dir_path.joinpath(CARDS_PKL_FILENAME)
orders_pkl_path = data_pkl_dir_path.joinpath(ORDERS_PKL_FILENAME)
order_details_pkl_path = data_pkl_dir_path.joinpath(ORDER_DETAILS_PKL_FILENAME)
reviews_pkl_path = data_pkl_dir_path.joinpath(REVIEWS_PKL_FILENAME)                
reviews_labelled_pkl_path = data_pkl_dir_path.joinpath(REVIEWS_LABELLED_PKL_FILENAME) 

# %% [markdown]
# # LOAD

# %% [markdown]
# ## Load Users

# %%
safeload_users_df = pd.read_csv(users_in_path,
                                dtype='string',
                                na_values=[],
                                keep_default_na=False
                                )

# %%
safeload_users_df.columns

# %%
# col_to_check = 'favouriteZones'
# safeload_users_df[col_to_check].drop_duplicates()

# %%
# Read and fix
users_df = pd.read_csv(users_in_path,
                       keep_default_na=False,
                       na_values=NA_VALUES,
                       dtype={
                           'user.uid': 'string',
                           'createdAt': 'string',
                           'source': 'string',
                           'isAnonymous': 'boolean',
                           'referralsCount': 'Int64',
                           'city': 'string',
                           'language': 'string',
                           'googleId': 'boolean',
                           'appleId': 'boolean',
                           'facebookId': 'boolean',
                           'referral.medium': 'string',
                           'referral.source': 'string',
                           'referral.type': 'Int64',
                           'favouriteZones': 'string'
                       }
                       )

users_df['createdAt'] = pd.to_datetime(users_df['createdAt'])

users_df['city'] = (users_df['city']
                    .apply(lambda x:
                           u.clean_str(x, 'lower')
                           if pd.notnull(x)
                           else None
                           )
                    )

users_df['referral.medium'] = (users_df['referral.medium']
                        .apply(lambda x:
                               u.clean_str(x, 'lower')
                               if pd.notnull(x)
                               else None
                               )
                        )

users_df['referral.source'] = (users_df['referral.source']
                        .apply(lambda x:
                               u.clean_str(x, 'lower')
                               if pd.notnull(x)
                               else None
                               )
                        )

users_df['favouriteZones'] = (users_df['favouriteZones']
                              .apply(lambda x:
                                     u.get_list_from_str(x)
                                     if pd.notnull(x)
                                     else []
                                     )
                              )

# %%
# CHECK PK VALIDITY

# SELECT count(1) as num_rows
# FROM users_df
# WHERE user.uid IS NULL

print(
    users_df
    .loc[lambda tbl: tbl['user.uid'].isnull()]
    .assign(aux=1.0)
    .shape[0]
)

# SELECT user.uid, count(1) as num_rows
# FROM users_df
# GROUP BY user.id
# HAVING num_rows > 1

print(
    users_df
    .assign(aux=1.0)
    .groupby(['user.uid'], dropna=False)
    .agg(num_rows=('aux', pd.Series.count))
    .loc[lambda tbl: tbl['num_rows'] > 1]
)

# %% [markdown]
# ## Load Profiles

# %%
safeload_profiles_df = pd.read_csv(profiles_in_path,
                                   dtype='string',
                                   na_values=[],
                                   keep_default_na=False
                                   )

# %%
safeload_profiles_df.columns

# %%
# col_to_check = 'types'
# safeload_profiles_df[col_to_check].drop_duplicates()

# %%
# Read and fix
profiles_df = pd.read_csv(profiles_in_path,
                       keep_default_na=False,
                       na_values=NA_VALUES,
                       dtype={
                           'user.uid': 'string',
                           'profile.uid': 'string',
                           'birthday': 'string',
                           'sex': 'string',
                           'city': 'string',
                           'height': 'Float64',
                           'weight': 'Float64',
                           'skibootsSize': 'Float64',
                           'level': 'string',
                           'types': 'string'
                       }
                       )

profiles_df['birthday'] = pd.to_datetime(profiles_df['birthday'])

def clean_profile_sex(sex: str) -> str | None:
    if pd.isna(sex):
        return None

    clean = u.clean_str(sex, 'upper')

    if clean in ['M', 'MASCHIO', 'UOMO']:
        return 'M'
    elif clean in ['F', 'FEMMINA', 'DONNA']:
        return 'F'
    else:
        return None
        
profiles_df['sex'] = (profiles_df['sex']
                      .apply(lambda x:
                             clean_profile_sex(x)
                             if pd.notnull(x)
                             else None
                             )
                      )

profiles_df['city'] = (profiles_df['city']
                    .apply(lambda x:
                           u.clean_str(x, 'lower')
                           if pd.notnull(x)
                           else None
                           )
                    )
profiles_df['types'] = (profiles_df['types']
                              .apply(lambda x:
                                     u.get_list_from_str(x)
                                     if pd.notnull(x)
                                     else []
                                     )
                              )

# %%
# CHECK PK VALIDITY

# SELECT count(1) as num_rows
# FROM profiles_df
# WHERE profile.uid IS NULL

print(
    profiles_df
    .loc[lambda tbl: tbl['profile.uid'].isnull()]
    .assign(aux=1.0)
    .shape[0]
)

# SELECT profile.uid, count(1) as num_rows
# FROM profiles_df
# GROUP BY profile.id
# HAVING num_rows > 1

print(
    profiles_df
    .assign(aux=1.0)
    .groupby(['profile.uid'], dropna=False)
    .agg(num_rows=('aux', pd.Series.count))
    .loc[lambda tbl: tbl['num_rows'] > 1]
)

# %%
# CHECK FK VALIDITY

# SELECT
#   A.user.uid,
#   count(1) as num_rows
# FROM 
#   (SELECT DISTINCT user.uid
#   FROM profiles_df
#   WHERE user.uid IS NOT NULL) AS A
#   LEFT JOIN
#   (SELECT user.uid, 1.0 AS in_users
#   FROM users_df) AS B
#   ON A.user.uid = B.user.uid
# GROUP BY in_users
# HAVING num_rows > 1

(profiles_df
 [['user.uid']]
 .loc[lambda tbl: tbl['user.uid'].notnull()]
 .drop_duplicates()
 .merge(users_df[['user.uid']].assign(in_users=1.0),
        how='left',
        on='user.uid'
        )
 .assign(aux=1.0)
 .groupby(['in_users'], dropna=False)
 .agg(num_rows=('aux', pd.Series.count))
 .loc[lambda tbl: tbl['num_rows'] > 1]
)

# %%
(profiles_df
 [['user.uid']]
 .loc[lambda tbl: tbl['user.uid'].notnull()]
 .drop_duplicates()
 .merge(users_df[['user.uid']].assign(in_users=1.0),
        how='left',
        on='user.uid'
        )
 .loc[lambda tbl: tbl['in_users'].isnull()]
)

# %%
profile_fail_useruids = (profiles_df
    [['user.uid']]
    .loc[lambda tbl: tbl['user.uid'].notnull()]
    .drop_duplicates()
    .merge(users_df[['user.uid']].assign(in_users=1.0),
            how='left',
            on='user.uid'
            )
    .loc[lambda tbl: tbl['in_users'].isnull()]
    ['user.uid']
)

print(profile_fail_useruids)

profile_fail_useruid_df = profiles_df.loc[lambda tbl: tbl['user.uid'].isin(profile_fail_useruids)]

print(profile_fail_useruid_df)

# %%
# FIX FK ISSUE
profiles_df.drop(list(profile_fail_useruid_df.index), inplace=True)

# %%
print(
    profiles_df
    [['user.uid']]
    .loc[lambda tbl: tbl['user.uid'].notnull()]
    .drop_duplicates()
    .merge(users_df[['user.uid']].assign(in_users=1.0),
            how='left',
            on='user.uid'
            )
    .assign(aux=1.0)
    .groupby(['in_users'], dropna=False)
    .agg(num_rows=('aux', pd.Series.count))
    .loc[lambda tbl: tbl['num_rows'] > 1]
)

# %% [markdown]
# ## Load Cards

# %%
safeload_cards_df = pd.read_csv(cards_in_path,
                                dtype='string',
                                na_values=[],
                                keep_default_na=False
                                )

# %%
safeload_cards_df.columns

# %%
# ...

# %%
# Read and fix
cards_df = pd.read_csv(
    cards_in_path,
    keep_default_na=False,
    na_values=NA_VALUES,
    dtype={
        "card.uid": "string",     
        "assignedAt": "string",  
        "birthday": "string",    
        "status": "string",      
        "user.uid": "string"     
    }
)

cards_df["assignedAt"] = pd.to_datetime(cards_df["assignedAt"], errors="coerce")
cards_df["birthday"] = pd.to_datetime(cards_df["birthday"], errors="coerce")

cards_df["status"] = (
    cards_df["status"]
    .apply(lambda x: u.clean_str(x, "lower") if pd.notna(x) else None)
)

VALID_CARD_STATUSES = {
    "error",
    "not-assigned",
    "pending",
    "warning",
    "valid",
    "rejected",
    "membership"
}

invalid_status = cards_df.loc[~cards_df["status"].isin(VALID_CARD_STATUSES)]

invalid_status.shape[0]



# %%
# CHECK PK VALIDITY

# SELECT count(1) as num_rows
# FROM cards_df
# WHERE card.uid IS NULL

print(
    cards_df
    .loc[lambda tbl: tbl["card.uid"].isnull()]
    .assign(aux=1.0)
    .shape[0]
)

# %%
# SELECT card.uid, count(1) as num_rows
# FROM cards_df
# GROUP BY card.uid
# HAVING num_rows > 1

print(
    cards_df
    .assign(aux=1.0)
    .groupby(["card.uid"], dropna=False)
    .agg(num_rows=("aux", pd.Series.count))
    .loc[lambda tbl: tbl["num_rows"] > 1]
)

# %%
cards_df = (
    cards_df
    .sort_values("assignedAt")               
    .groupby("card.uid", as_index=False)     
    .tail(1)                                
)

print(
    cards_df
    .loc[lambda tbl: tbl["card.uid"].isnull()]
    .shape[0]
)

print(
    cards_df
    .assign(aux=1.0)
    .groupby(["card.uid"], dropna=False)
    .agg(num_rows=("aux", pd.Series.count))
    .loc[lambda tbl: tbl["num_rows"] > 1]
)


# %%
# CHECK FK VALIDITY WITH BUSINESS RULE

# SELECT count(1)
# FROM cards_df
# WHERE status != 'not-assigned'
# AND user.uid NOT IN (SELECT user.uid FROM users_df)

print(
    cards_df
    .loc[
        (cards_df["status"] != "not-assigned") &
        (~cards_df["user.uid"].isin(users_df["user.uid"]))
    ]
    .assign(aux=1.0)
    .shape[0]
)

# %%
# FIX FK VIOLATIONS (KEEP ONLY VALID CARD RECORDS)

cards_df = (
    cards_df
    .loc[
        (cards_df["status"] == "not-assigned") |
        (cards_df["user.uid"].isin(users_df["user.uid"]))
    ]
    .copy()
)

print(
    cards_df
    .loc[
        (cards_df["status"] != "not-assigned") &
        (~cards_df["user.uid"].isin(users_df["user.uid"]))
    ]
    .assign(aux=1.0)
    .shape[0]
)

# %%
empty_rows = (
    (cards_df["status"] == "not-assigned") &
    (cards_df["user.uid"].isna() | (cards_df["user.uid"] == "")) &
    (cards_df["assignedAt"].isna()) &
    (cards_df["birthday"].isna())
)

print(empty_rows.sum())

cards_df = cards_df.loc[~empty_rows].reset_index(drop=True)

# %%
len(cards_df)

# %% [markdown]
# ## Load Orders

# %%
safeload_orders_df = pd.read_csv(orders_in_path,
                                 dtype='string',
                                 na_values=[],
                                 keep_default_na=False
                                )

# %%
safeload_orders_df.columns

# %%
# col_to_check = 'clientInfo'
# safeload_orders_df[col_to_check].drop_duplicates()

# %%
# Read and fix
orders_df = pd.read_csv(
    orders_in_path,
    keep_default_na=False,
    na_values=NA_VALUES,
    dtype={
        "order.uid": "string",         
        "user.uid": "string",          
        "createdAt": "string",
        "createdAtTime": "string",
        "paymentGateway": "string",
        "paymentBrand": "string",
        "pickup": "boolean",
        "pickupComplete": "boolean",
        "source": "string",
        "tenant": "string",
        "paymentAttempts": "Int64",
        "timeZone": "string",
        "clientInfo": "string"        
    }
)

orders_df["createdAt"] = pd.to_datetime(orders_df["createdAt"], errors="coerce")
orders_df["createdAtTime"] = pd.to_datetime(orders_df["createdAtTime"], errors="coerce")

# Text Normalization
orders_df["paymentGateway"] = orders_df["paymentGateway"].apply(
    lambda x: u.clean_str(x, "lower") if pd.notna(x) else None
)

orders_df["paymentBrand"] = orders_df["paymentBrand"].apply(
    lambda x: u.clean_str(x, "lower") if pd.notna(x) else None
)

orders_df["source"] = orders_df["source"].apply(
    lambda x: u.clean_str(x, "lower") if pd.notna(x) else None
)


# %%
# PK NULL CHECK

# SELECT count(1)
# FROM orders_df
# WHERE order.uid IS NULL;

print(
    orders_df
    .loc[lambda tbl: tbl["order.uid"].isnull()]
    .assign(aux=1.0)
    .shape[0]
)

# PK DUPLICATE CHECK

# SELECT order.uid, count(1) as num_rows
# FROM orders_df
# GROUP BY order.uid
# HAVING num_rows > 1;

print(
    orders_df
    .assign(aux=1.0)
    .groupby(["order.uid"], dropna=False)
    .agg(num_rows=("aux", "count"))
    .loc[lambda tbl: tbl["num_rows"] > 1]
)

# %%
# FK CHECK

# SELECT count(1)
# FROM orders_df
# WHERE user.uid NOT IN (SELECT user.uid FROM users_df);

print(
    orders_df
    .loc[~orders_df["user.uid"].isin(users_df["user.uid"])]
    .assign(aux=1.0)
    .shape[0]
)

# %%
orders_df = orders_df.loc[
    orders_df["user.uid"].isin(users_df["user.uid"])
].copy()

# %% [markdown]
# ## Load Order Details

# %%
safeload_order_details_df = pd.read_csv(order_details_in_path,
                                        dtype='string',
                                        na_values=[],
                                        keep_default_na=False
                                        )

# %%
safeload_order_details_df.columns

# %%
# ...

# %%
# Read and fix
order_details_df = pd.read_csv(
    order_details_in_path,
    keep_default_na=False,
    na_values=NA_VALUES,
    dtype={
        "item.uid": "string",                 
        "order.uid": "string",                
        "item.status": "string",              
        "item.date": "string",                
        "product.uid": "string",
        "product.dynamicPricing": "boolean",
        "item.amount": "Float64",             
        "item.discount": "boolean",          
        "product.type": "string",
        "item.zoneName": "string",
        "product.durationHours": "Float64",
        "item.profiles": "string",           
        "item.variantName": "string",
        "item.slotName": "string",
        "item.snowitcardNumber": "string"
    }
)

order_details_df["item.date"] = pd.to_datetime(order_details_df["item.date"], errors="coerce")

order_details_df["item.status"] = order_details_df["item.status"].apply(
    lambda x: u.clean_str(x, "lower") if pd.notna(x) else None
)

order_details_df["product.type"] = order_details_df["product.type"].apply(
    lambda x: u.clean_str(x, "lower") if pd.notna(x) else None
)

order_details_df["item.zoneName"] = order_details_df["item.zoneName"].apply(
    lambda x: u.clean_str(x, "lower") if pd.notna(x) else None
)

VALID_ITEM_STATUS = {
    "ok",
    "fulfilled",
    "on-hold",
    "cancelled",
    "transfer",
    "processing"
}

print(
    order_details_df
    .loc[~order_details_df["item.status"].isin(VALID_ITEM_STATUS)]
    .assign(aux=1.0)
    .shape[0]
)


# %%
order_details_df = order_details_df.loc[
    order_details_df["item.status"].isin(VALID_ITEM_STATUS)
].copy()

print(
    order_details_df
    .loc[~order_details_df["item.status"].isin(VALID_ITEM_STATUS)]
    .assign(aux=1.0)
    .shape[0]
)

# %%
# PK CHECK

# SELECT count(1)
# FROM order_details_df
# WHERE item.uid IS NULL

print(
    order_details_df
    .loc[lambda tbl: tbl["item.uid"].isnull()]
    .assign(aux=1.0)
    .shape[0]
)

# SELECT order.uid, item.uid, item.status, count(1) AS num_rows
# FROM order_details_df
# GROUP BY order.uid, item.uid, item.status
# HAVING num_rows > 1

print(
    order_details_df
    .assign(aux=1.0)
    .groupby(["order.uid", "item.uid", "item.status"], dropna=False)
    .agg(num_rows=("aux", "count"))
    .loc[lambda tbl: tbl["num_rows"] > 1]
)

# %%
# FK VALIDATION

# SELECT COUNT(1)
# FROM order_details_df
# WHERE order.uid NOT IN (SELECT order.uid FROM orders_df);

print(
    order_details_df
    .loc[~order_details_df["order.uid"].isin(orders_df["order.uid"])]
    .assign(aux=1.0)
    .shape[0]
)

order_details_df = order_details_df.loc[
    order_details_df["order.uid"].isin(orders_df["order.uid"])
].copy()

print(
    order_details_df
    .loc[~order_details_df["order.uid"].isin(orders_df["order.uid"])]
    .assign(aux=1.0)
    .shape[0]
)

# %%
print(
    order_details_df["item.amount"].isna().sum()
)

print(
    order_details_df
    .loc[order_details_df["item.amount"] < 0]
    .shape[0]
)

# %%
order_details_df = order_details_df.loc[
    order_details_df["item.amount"] >= 0
].copy()

print(
    order_details_df
    .loc[order_details_df["item.amount"] < 0]
    .shape[0]
)

# %% [markdown]
# ## Load Reviews

# %%
safeload_reviews_df = pd.read_csv(reviews_in_path,
                                  dtype="string",
                                  na_values=[],
                                  keep_default_na=False
                                  )

# %%
safeload_reviews_df.columns

# %%
reviews_df = pd.read_csv(reviews_in_path,
                         keep_default_na=False,
                         na_values=NA_VALUES,
                         dtype={
                             "review.uid": "string",   
                             "user.uid": "string",     
                             "text": "string"          
                         }
                         )

reviews_df["text"] = (
    reviews_df["text"]
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

# %%
# CHECK PK VALIDITY

# SELECT count(1) as num_rows
# FROM reviews_df
# WHERE review.uid IS NULL

print(
    reviews_df
    .loc[lambda tbl: tbl["review.uid"].isnull()]
    .assign(aux=1.0)
    .shape[0]
)

# SELECT review.uid, count(1) as num_rows
# FROM reviews_df
# GROUP BY review.uid
# HAVING num_rows > 1

print(
    reviews_df
    .assign(aux=1.0)
    .groupby(["review.uid"], dropna=False)
    .agg(num_rows=("aux", pd.Series.count))
    .loc[lambda tbl: tbl["num_rows"] > 1]
)

# %%
# CHECK FK VALIDITY (NULLS)

# SELECT count(1) as num_rows
# FROM reviews_df
# WHERE user.uid IS NULL

print(
    reviews_df
    .loc[lambda tbl: tbl["user.uid"].isnull()]
    .assign(aux=1.0)
    .shape[0]
)

# SELECT count(1) as num_rows
# FROM reviews_df
# WHERE user.uid NOT IN (SELECT user.uid FROM users_df)

print(
    reviews_df
    .loc[~reviews_df["user.uid"].isin(users_df["user.uid"])]
    .assign(aux=1.0)
    .shape[0]
)

# %% [markdown]
# ## Load Reviews Labelled

# %%
safeload_reviews_labelled_df = pd.read_csv(reviews_labelled_in_path,
                                           dtype="string",
                                           na_values=[],
                                           keep_default_na=False
                                           )

safeload_reviews_labelled_df.columns

# %%
reviews_labelled_df = pd.read_csv(
    reviews_labelled_in_path,
    keep_default_na=False,
    na_values=NA_VALUES,
    dtype={
        "labelled_review.uid": "string",   
        "text": "string",                  
        "sentiment_label": "string"        
    }
)

reviews_labelled_df["text"] = (
    reviews_labelled_df["text"]
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

reviews_labelled_df["sentiment_label"] = (
    reviews_labelled_df["sentiment_label"]
    .str.strip()
    .str.lower()
)

# %%
# CHECK PK VALIDITY

# SELECT count(1)
# FROM reviews_labelled_df
# WHERE labelled_review.uid IS NULL

print(
    reviews_labelled_df
    .loc[lambda tbl: tbl["labelled_review.uid"].isnull()]
    .assign(aux=1.0)
    .shape[0]
)

# SELECT labelled_review.uid, count(1)
# FROM reviews_labelled_df
# GROUP BY labelled_review.uid
# HAVING count(1) > 1

print(
    reviews_labelled_df
    .assign(aux=1.0)
    .groupby(["labelled_review.uid"], dropna=False)
    .agg(num_rows=("aux", pd.Series.count))
    .loc[lambda tbl: tbl["num_rows"] > 1]
)

# %% [markdown]
# # Save

# %%
# Save Cleaned Dataset

with users_pkl_path.open('wb') as fh:
    dill.dump(users_df, fh)
print(f"Save users data in {users_pkl_path.as_posix()}")

with profiles_pkl_path.open('wb') as fh:
    dill.dump(profiles_df, fh)
print(f"Save profiles data in {profiles_pkl_path.as_posix()}")

with cards_pkl_path.open('wb') as fh:
    dill.dump(cards_df, fh)
print(f"Save cards data in {cards_pkl_path.as_posix()}")

# %%
with orders_pkl_path.open('wb') as fh:
    dill.dump(orders_df, fh)
print(f"Save orders data in {orders_pkl_path.as_posix()}")

with order_details_pkl_path.open('wb') as fh:
    dill.dump(order_details_df, fh)
print(f"Save order details data in {order_details_pkl_path.as_posix()}")

# %%
with reviews_pkl_path.open('wb') as fh:
    dill.dump(reviews_df, fh)
print(f"Save reviews data in {reviews_pkl_path.as_posix()}")

with reviews_labelled_pkl_path.open('wb') as fh:
    dill.dump(reviews_labelled_df, fh)
print(f"Save reviews labelled data in {reviews_labelled_pkl_path.as_posix()}")

# %%



