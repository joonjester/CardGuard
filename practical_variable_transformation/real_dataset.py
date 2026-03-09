import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer
import matplotlib.pyplot as plt

# =========================
# 1. Datensatz laden
# =========================
file_path = "/Users/nikolaibelikov/CardGuard/creditcard.csv"
df = pd.read_csv(file_path)
df_original = df.copy()

timestamp_col = "Time"

# =========================
# 2. Spalten definieren
# =========================

binary_cols = [
    "Is_Weekend",
    "Class",
]

numeric_cols = [
    "Amount",
]

# =========================
# 3. Timestamp aufspalten
# =========================
df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

df["Timestamp_Year"] = df[timestamp_col].dt.year
df["Timestamp_Month"] = df[timestamp_col].dt.month
df["Timestamp_Day"] = df[timestamp_col].dt.day
df["Timestamp_DayOfWeek"] = df[timestamp_col].dt.dayofweek
df["Timestamp_Hour"] = df[timestamp_col].dt.hour
df["Timestamp_Minute"] = df[timestamp_col].dt.minute

# Originale Timestamp-Spalte entfernen
df = df.drop(columns=[timestamp_col])

# =========================
# 5. User_ID frequency encoding
# =========================

# Timestamp-Missing-Values entfernen oder separat behandeln
df = df.dropna(subset=[col for col in ["Timestamp_Year", "Timestamp_Month", "Timestamp_Day",
                                       "Timestamp_DayOfWeek", "Timestamp_Hour", "Timestamp_Minute"]
                       if col in df.columns])

# =========================
# 7. Zusätzliche numerische Timestamp-/ID-Features
# =========================
derived_numeric_cols = [
    "Timestamp_Year",
    "Timestamp_Month",
    "Timestamp_Day",
    "Timestamp_DayOfWeek",
    "Timestamp_Hour",
    "Timestamp_Minute",
]

all_numeric_to_transform = numeric_cols + derived_numeric_cols

# Nur die Spalten transformieren, die wirklich existieren

all_numeric_to_transform = [col for col in all_numeric_to_transform if col in df.columns]

# =========================
# 7b. Optional: Spalten entfernen
# =========================
columns_to_remove = [
                     ]

# Nur vorhandene Spalten entfernen
columns_to_remove = [col for col in columns_to_remove if col in df.columns]
df_encoded = df.drop(columns=columns_to_remove)

# =========================
# 8. Version A: Z-Score
# =========================
df_zscore = df_encoded.copy()

scaler = StandardScaler()
df_zscore[all_numeric_to_transform] = scaler.fit_transform(df_zscore[all_numeric_to_transform])

# =========================
# 9. Version B: Yeo-Johnson
# =========================
df_yeojohnson = df_encoded.copy()

pt = PowerTransformer(method="yeo-johnson", standardize=True)
df_yeojohnson[all_numeric_to_transform] = pt.fit_transform(df_yeojohnson[all_numeric_to_transform])

# =========================
# 10. Vergleichsplot für Transaction_Amount
# =========================
if "Amount" not in df_zscore.columns or "Amount" not in df_yeojohnson.columns:
    raise ValueError("Transaction_Amount wurde entfernt. Entferne sie nicht aus columns_to_remove, wenn du den Vergleichsplot erstellen willst.")
plot_df = pd.DataFrame({
    "Original": df_original["Amount"],
    "Z-Score": df_zscore["Amount"],
    "Yeo-Johnson": df_yeojohnson["Amount"],
})

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Boxplot Original einzeln
axes[0].boxplot(
    plot_df["Original"].dropna(),
    tick_labels=["Original"]
)
axes[0].set_title("Transaction_Amount: Original")
axes[0].set_ylabel("Wert")

# Boxplot Z-Score vs. Yeo-Johnson
axes[1].boxplot([
    plot_df["Z-Score"].dropna(),
    plot_df["Yeo-Johnson"].dropna()
], tick_labels=["Z-Score", "Yeo-Johnson"])
axes[1].set_title("Transaction_Amount: Transformierte Werte")
axes[1].set_ylabel("Wert")

# Histogramm: Z-Score vs. Yeo-Johnson
axes[2].hist(plot_df["Z-Score"].dropna(), bins=40, alpha=0.5, label="Z-Score")
axes[2].hist(plot_df["Yeo-Johnson"].dropna(), bins=40, alpha=0.5, label="Yeo-Johnson")
axes[2].set_title("Transaction_Amount: Verteilungsvergleich")
axes[2].set_xlabel("Wert")
axes[2].set_ylabel("Häufigkeit")
axes[2].legend()

plt.tight_layout()
plt.savefig("transaction_amount_comparison_real_dataset.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================
# 11. Speichern
# =========================
#df_zscore.to_csv("fraud_transformed_zscore.csv", index=False)
#df_yeojohnson.to_csv("fraud_transformed_yeojohnson.csv", index=False)

#print("Fertig.")
#print("Gespeichert als:")
#print("- fraud_transformed_zscore_real_dataset.csv")
#print("- fraud_transformed_yeojohnson_real.csv")
#print("- transaction_amount_comparison.png")