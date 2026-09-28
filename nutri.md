# ==========================================
# CELL 1 - IMPORT LIBRARY
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 4 algoritma
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# Evaluasi
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

print("Library berhasil di-import")

# ==========================================
# CELL 2 - UPLOAD DATASET
# ==========================================

from google.colab import files

uploaded = files.upload()

file_name = list(uploaded.keys())[0]

df = pd.read_csv(file_name)

print("Dataset berhasil dibaca")
print("Ukuran dataset:", df.shape)

df.head()

# ==========================================
# CELL 3 - CEK DATASET
# ==========================================

print("Jumlah data:", len(df))

print("\nKolom dataset:")
print(df.columns.tolist())

print("\nJumlah masing-masing Status:")
print(df["Status"].value_counts())

print("\nJumlah masing-masing Gender:")
print(df["JenisKelamin"].value_counts())

# ==========================================
# CELL 4 - PREPROCESSING STATUS
# ==========================================

df = df.copy()

# Bersihkan teks status
df["Status"] = df["Status"].astype(str).str.strip()

# Menggabungkan 4 status menjadi 2 kelas
status_mapping = {
    "Normal": "Normal",
    "Tinggi": "Normal",
    "Stunting": "Stunting",
    "Stunting Parah": "Stunting"
}

df["Target"] = df["Status"].map(status_mapping)

print("Status sebelum digabung:")
print(df["Status"].value_counts())

print("\nStatus setelah digabung:")
print(df["Target"].value_counts())

# ==========================================
# CELL 5 - MEMBUAT FITUR
# ==========================================

# Encoding gender
# Laki-laki = 0
# Perempuan = 1

df["GenderEncoded"] = (
    df["JenisKelamin"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "laki-laki": 0,
        "perempuan": 1
    })
)

# Daftar kolom spektral
spektral_cols = [
    f"Spektral_{i}"
    for i in range(1, 19)
]

# ==========================================
# 3 KONDISI FITUR
# ==========================================

conditions = {

    # Gender + Usia + seluruh spektral
    "Kondisi 1 - Gender + Usia + Spektral":
        ["GenderEncoded", "UsiaBulan"] + spektral_cols,

    # Gender + seluruh spektral
    "Kondisi 2 - Gender + Spektral":
        ["GenderEncoded"] + spektral_cols,

    # Usia + seluruh spektral
    "Kondisi 3 - Usia + Spektral":
        ["UsiaBulan"] + spektral_cols
}

for nama, fitur in conditions.items():
    print("\n", nama)
    print("Jumlah fitur:", len(fitur))
    print(fitur)

# ==========================================
# CELL 6 - TRAIN TEST SPLIT
# ==========================================

# Hapus data yang target/gender-nya kosong jika ada
df_model = df.dropna(
    subset=["Target", "GenderEncoded", "UsiaBulan"]
).reset_index(drop=True)

y = df_model["Target"]

# Membagi INDEX, bukan langsung fiturnya
train_idx, test_idx = train_test_split(
    np.arange(len(df_model)),
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Total data:", len(df_model))
print("Data training:", len(train_idx))
print("Data testing :", len(test_idx))

print("\nDistribusi Training:")
print(y.iloc[train_idx].value_counts())

print("\nDistribusi Testing:")
print(y.iloc[test_idx].value_counts())

# ==========================================
# CELL 6.5 - METODE BALANCING
# ==========================================

from imblearn.over_sampling import RandomOverSampler

# Membuat metode balancing
oversampler = RandomOverSampler(
    sampling_strategy="auto",
    random_state=42
)

print("Random Over Sampling siap digunakan")

# 1. Import modul yang dibutuhkan
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# 2. Inisialisasi model XGBoost
model_xgb = Pipeline([
    ("model", XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    ))
])

# ==========================================
# CELL 8 - TRAINING DAN EVALUASI XGBOOST
# ==========================================

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

results = []
confusion_matrices = {}

# Mapping label khusus XGBoost
label_mapping = {
    "Normal": 0,
    "Stunting": 1
}

reverse_mapping = {
    0: "Normal",
    1: "Stunting"
}

# ==========================================
# LOOP KONDISI FITUR
# ==========================================

for condition_name, features in conditions.items():

    print("\n" + "=" * 80)
    print(condition_name)
    print("=" * 80)

    # Mengambil fitur sesuai kondisi
    X = df_model[features]
    y = df_model["Target"]

    # Split train & test
    X_train = X.iloc[train_idx].copy()
    X_test  = X.iloc[test_idx].copy()

    y_train = y.iloc[train_idx]
    y_test  = y.iloc[test_idx]

    # ==========================================
    # PENANGANAN MISSING VALUES (NaN)
    # ==========================================
    imputer = SimpleImputer(strategy='median')

    X_train_clean = imputer.fit_transform(X_train)
    X_train = pd.DataFrame(X_train_clean, columns=X_train.columns, index=X_train.index)

    X_test_clean = imputer.transform(X_test)
    X_test = pd.DataFrame(X_test_clean, columns=X_test.columns, index=X_test.index)

    # ==========================================
    # KHUSUS XGBOOST (MENGGUNAKAN CLASS WEIGHT)
    # ==========================================

    # Ubah target menjadi 0 dan 1
    y_train_xgb = y_train.map(label_mapping)

    # Hitung jumlah masing-masing kelas
    jumlah_normal = (y_train_xgb == 0).sum()
    jumlah_stunting = (y_train_xgb == 1).sum()

    # Hitung bobot kelas positif (Stunting)
    scale_pos_weight = jumlah_normal / jumlah_stunting

    print(f"Jumlah Normal   : {jumlah_normal}")
    print(f"Jumlah Stunting : {jumlah_stunting}")
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    # Set parameter scale_pos_weight pada model XGBoost
    model_xgb.set_params(
        model__scale_pos_weight=scale_pos_weight
    )

    # Training menggunakan Data Asli
    model_xgb.fit(
        X_train,
        y_train_xgb
    )

    # Prediksi
    y_pred_xgb = model_xgb.predict(X_test)

    # Kembalikan 0/1 menjadi label asli ("Normal"/"Stunting")
    y_pred = pd.Series(y_pred_xgb).map(reverse_mapping).values

    # ==========================================
    # EVALUASI
    # ==========================================

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="Stunting", zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label="Stunting", zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label="Stunting", zero_division=0)

    # Simpan Hasil
    results.append({
        "Kondisi": condition_name,
        "Algoritma": "XGBoost",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=["Normal", "Stunting"])
    confusion_matrices[(condition_name, "XGBoost")] = cm

    # Tampilkan Hasil
    print(
        f"\nXGBoost                | "
        f"Accuracy: {accuracy:.4f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )

print("\n" + "=" * 80)
print("TRAINING XGBOOST SELESAI")
print("=" * 80)

