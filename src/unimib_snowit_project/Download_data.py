"""
Download all project datasets from Google Drive into data_input/.

Usage:
    python src/download_data.py
"""

import os
from pathlib import Path
import gdown

# Ensure data_input/ exists
DATA_DIR = Path("data_input")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Google Drive FILE IDs for each dataset
FILE_IDS = {
    "cards.csv":             "1O03lb4ZqqMwy3aPeZquAT_OU9ChNyIEM",
    "order_details.csv":     "1u1vsiNpF6saIcG3iDN-R7AiYAFqf9uzq",
    "orders.csv":            "161Rv1xsAevewSnTRFV-9q6FWokgr_-Xt",
    "profiles.csv":          "1bOe2EaMQbrBZyINh9pLMPltDJ2BoNG_D",
    "reviews_labelled.csv":  "1DRYj7r7VnlNyCSMrEHLbwWqj0fodBymV",
    "reviews.csv":           "1kLhalVn5iczHSdp5o3I2PIWCqkOFfpuO",
    "users.csv":             "1K2lycSQXUyMSBx0_NqfuljYzJbW6RzsF",
}

def download_file(file_id: str, filename: str):
    """Download a single file from Google Drive into data_input/."""
    url = f"https://drive.google.com/uc?id={file_id}"
    out_path = DATA_DIR / filename
    print(f"Downloading {filename} ...")
    gdown.download(url, str(out_path), quiet=False)

def main():
    print("📥 Starting dataset download from Google Drive...\n")
    for fname, fid in FILE_IDS.items():
        download_file(fid, fname)
    print("\n✅ All files downloaded successfully to data_input/")

if __name__ == "__main__":
    main()
