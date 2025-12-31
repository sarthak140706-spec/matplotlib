"""
Water Quality Data Visualization using Matplotlib
Dataset: NWMP Indian Lakes Water Quality (2017–2022)
Source: Kaggle (NWMP Water Quality Data for Indian Lakes)

Author: Sarthak Jadhav
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# Configuration
# ===============================
DATA_FOLDER = "../water_quality"
PLOTS_FOLDER = "../plots"

os.makedirs(PLOTS_FOLDER, exist_ok=True)

# ===============================
# Helper Functions
# ===============================
def load_and_merge_data(folder_path):
    """
    Loads all year-wise CSV files, extracts year from filename,
    and merges them into a single DataFrame.
    """
    all_frames = []

    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".csv"):
            year = int(file.split("_")[0])
            df = pd.read_csv(os.path.join(folder_path, file))
            df["year"] = year  # Ensure consistent year column
            all_frames.append(df)

    return pd.concat(all_frames, ignore_index=True)


def normalize_columns(df):
    """
    Normalizes column names to lowercase, removes spaces,
    and replaces parentheses with underscores.
    """
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
    )
    return df


# ===============================
# Load Dataset
# ===============================
df = load_and_merge_data(DATA_FOLDER)
df = normalize_columns(df)

print("Available Columns:")
print(df.columns)

# ===============================
# Convert Key Columns to Numeric
# ===============================
numeric_cols = [
    "min_ph", "max_ph",
    "min_dissolved_oxygen", "max_dissolved_oxygen",
    "min_bod", "max_bod"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # convert invalid entries to NaN

# ===============================
# Compute Average Parameters
# ===============================
df["avg_ph"] = df[["min_ph", "max_ph"]].mean(axis=1)
df["avg_do"] = df[["min_dissolved_oxygen", "max_dissolved_oxygen"]].mean(axis=1)
df["avg_bod"] = df[["min_bod", "max_bod"]].mean(axis=1)

# ===============================
# Data Cleaning
# ===============================
df = df.dropna(subset=["year", "avg_ph"])

print("\nCleaned Dataset Shape:", df.shape)

# ===============================
# Plot 1: Average pH Trend
# ===============================
ph_trend = df.groupby("year")["avg_ph"].mean()

plt.figure(figsize=(8, 5))
plt.plot(ph_trend.index, ph_trend.values, marker="o", linewidth=2)
plt.title("Average pH Trend in Indian Lakes (2017–2022)")
plt.xlabel("Year")
plt.ylabel("Average pH")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{PLOTS_FOLDER}/avg_ph_trend.png")
plt.show()

# ===============================
# Plot 2: Multi-Parameter Trend
# ===============================
parameters = {
    "pH": "avg_ph",
    "DO": "avg_do",
    "BOD": "avg_bod"
}

plt.figure(figsize=(9, 5))
for label, col in parameters.items():
    yearly_avg = df.groupby("year")[col].mean()
    plt.plot(yearly_avg.index, yearly_avg.values, marker="o", label=label)

plt.title("Water Quality Parameters Trend (2017–2022)")
plt.xlabel("Year")
plt.ylabel("Average Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{PLOTS_FOLDER}/parameter_trends.png")
plt.show()

# ===============================
# Plot 3: State-wise pH Comparison
# ===============================
state_col = "state_name"
if state_col in df.columns:
    top_states = df[state_col].value_counts().head(5).index
    state_data = df[df[state_col].isin(top_states)]

    state_ph = state_data.groupby(state_col)["avg_ph"].mean()

    plt.figure(figsize=(8, 5))
    state_ph.plot(kind="bar")
    plt.title("Average pH of Top 5 States")
    plt.xlabel("State")
    plt.ylabel("Average pH")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_FOLDER}/state_ph_comparison.png")
    plt.show()

# ===============================
# Plot 4: BOD Distribution
# ===============================
plt.figure(figsize=(7, 5))
plt.hist(df["avg_bod"].dropna(), bins=30, color="skyblue")
plt.title("Distribution of Biochemical Oxygen Demand (BOD)")
plt.xlabel("BOD (mg/L)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{PLOTS_FOLDER}/bod_distribution.png")
plt.show()

print("\n✅ Visualization completed successfully.")
print("📁 Plots saved in the 'plots/' folder.")
