import pandas as pd
import numpy as np


# Load data
df1 = pd.read_csv("../data_collection/final_5ghz_cleaned.csv")
df2 = pd.read_csv("../data_collection/final_2ghz_cleaned.csv")
# Combine datasets
df = pd.concat([df1, df2], ignore_index=True)

# BASIC CLEANING: strip whitespace from column names and Location values
df.columns = df.columns.str.strip()
df["Location"] = df["Location"].astype(str).str.strip()

# normalize text
df["Location_lower"] = df["Location"].str.lower()

# FEATURE EXTRACTION
# Extract distance and load from location (e.g. "skyDhaba_ch56_near_rush" -> "Near", "Rush")
df["Distance"] = df["Location_lower"].str.extract(r'(near|far)', expand=False).str.capitalize()
df["Load"] = df["Location_lower"].str.extract(r'(rush|empty|moderate)', expand=False).str.capitalize()
# Extract channel number from location (e.g. "skyDhaba_ch56" -> 56)
df["Channel"] = pd.to_numeric(
    df["Location_lower"].str.extract(r'ch(\d+)', expand=False),
    errors="coerce"
)
# Convert Frame_Type to numeric
df["Frame_Type"] = pd.to_numeric(df["Frame_Type"], errors="coerce")

# TIME FEATURES
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df["Hour"] = df["Timestamp"].dt.hour
df["Minute"] = df["Timestamp"].dt.minute
df["Second"] = df["Timestamp"].dt.second

# MCS CLEANING
df["MCS_WiFi6"] = pd.to_numeric(df["MCS_WiFi6"], errors="coerce")
df["MCS_WiFi5"] = pd.to_numeric(df["MCS_WiFi5"], errors="coerce")
df["MCS_Legacy"] = pd.to_numeric(df["MCS_Legacy"], errors="coerce")
# Determine the correct MCS value and Wi-Fi version for each row
def get_mcs(row):
    if pd.notna(row["MCS_WiFi6"]):
        return float(row["MCS_WiFi6"]), "WiFi6"
    elif pd.notna(row["MCS_WiFi5"]):
        return float(row["MCS_WiFi5"]), "WiFi5"
    elif pd.notna(row["MCS_Legacy"]):
        return float(row["MCS_Legacy"]), "WiFi4"
    else:
        return np.nan, "Unknown"
#axis=1 applies the function to each row, result_type="expand" allows returning multiple columns (MCS and WiFi_Version):
df[["MCS", "WiFi_Version"]] = df.apply(get_mcs, axis=1, result_type="expand")

# DROP RAW / UNNEEDED COLUMNS
df = df.drop(columns=[
    "Transmitter_MAC",
    "Receiver_MAC",
    "BSSID",
    "Timestamp",
    "Location_lower",
    "MCS_WiFi6",
    "MCS_WiFi5",
    "MCS_Legacy"
])

# FINAL CLEAN
df = df.drop_duplicates()

# SAVE
df.to_csv("wifi_preprocessed_clean.csv", index=False)

#information needed to conclude which features are most important for MCS prediction and to understand the distribution of the dataset
#pritning percentage distribution of distance and load classes
print(df["Distance"].value_counts(normalize=True) * 100)
print(df["Load"].value_counts(normalize=True) * 100)
# # print(df.head())

#checking number of samples:
df = pd.read_csv("wifi_preprocessed_clean.csv")

print("Total samples:", len(df))
print("Rows, Columns:", df.shape)

# missing = df["MCS"].isna().sum()
# total = len(df)
# print("Missing MCS:", missing)
# print("Total rows:", total)
# print("Missing %:", (missing / total) * 100)
