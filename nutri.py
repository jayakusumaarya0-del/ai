import os

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


FILE_CSV = "dataset_gabungan_training.csv"
OUTPUT_DIR = "hasil_analisis_kondisi_1"
CSV_COLUMNS = [
    "No", "Tanggal", "SessionID", "IDAlat", "RSSI", "NamaAnak",
    "JenisKelamin", "TinggiCm", "BeratKg", "UsiaBulan", "ZTBU", "ZBBU",
    "ZBBTB", "Status", "Status_Prediksi",
] + [f"Spektral_{index}" for index in range(1, 19)]
REFERENCE_DATE = pd.Timestamp("2026-09-24")


def set_dummy_birth_dates(data, dummy_mask):
    birth_dates = [
        (REFERENCE_DATE - pd.DateOffset(months=int(months))).strftime(
            "%d-%m-%Y"
        )
        for months in data.loc[dummy_mask, "UsiaBulan"].fillna(0)
    ]
    data.loc[dummy_mask, "Tanggal"] = birth_dates


def load_and_combine_data():
    data = pd.read_csv(FILE_CSV)
    added_rows = 0

    dummy_mask = (
        data["Tanggal"].astype(str).str.strip().str.upper().eq("SINTETIS")
        | data["NamaAnak"].astype(str).str.startswith("Dummy_")
        | data["SessionID"].astype(str).str.startswith("DUMMY-")
    )
    dummy_numbers = np.arange(1, dummy_mask.sum() + 1)
    set_dummy_birth_dates(data, dummy_mask)
    data.loc[dummy_mask, "NamaAnak"] = [
        f"Dummy_Anak_{number:03d}" for number in dummy_numbers
    ]
    data.loc[dummy_mask, "SessionID"] = [
        f"DUMMY-{number:04d}" for number in dummy_numbers
    ]

    return data, added_rows


def build_models(scale_pos_weight):
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(class_weight="balanced", random_state=42)),
        ]),
        "KNN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                tree_method="exact",
                random_state=42,
                eval_metric="logloss",
            )),
        ]),
    }


def main():
    if not os.path.exists(FILE_CSV):
        raise FileNotFoundError(
            f"File CSV tidak ditemukan: {FILE_CSV}. "
            "Letakkan CSV di folder yang sama dengan nutri.py."
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df, _ = load_and_combine_data()
    print(f"Dataset training: {FILE_CSV}")
    print(f"Ukuran dataset: {df.shape}")

    status_mapping = {
        "Normal": "Normal",
        "Tinggi": "Normal",
        "Stunting": "Stunting",
        "Stunting Parah": "Stunting",
    }
    df["Status"] = df["Status"].astype(str).str.strip()
    df["Target"] = df["Status"].map(status_mapping)
    df["GenderEncoded"] = (
        df["JenisKelamin"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"laki-laki": 0, "perempuan": 1})
    )

    spectral_columns = [f"Spektral_{index}" for index in range(1, 19)]
    features = ["GenderEncoded", "UsiaBulan"] + spectral_columns
    df_model = df.dropna(
        subset=["Target", "GenderEncoded", "UsiaBulan"]
    ).reset_index(drop=True)
    X = df_model[features]
    y = df_model["Target"]

    print("Distribusi target setelah penggabungan:")
    print(y.value_counts().to_string())

    train_idx, test_idx = train_test_split(
        np.arange(len(df_model)),
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    label_mapping = {"Normal": 0, "Stunting": 1}
    reverse_mapping = {0: "Normal", 1: "Stunting"}
    y_train_num = y_train.map(label_mapping)
    y_test_num = y_test.map(label_mapping)
    scale_pos_weight = (y_train_num == 0).sum() / (y_train_num == 1).sum()

    models = build_models(scale_pos_weight)
    results = []
    predictions = {}
    fitted_models = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train_num)
        prediction_numbers = model.predict(X_test).astype(int)
        predictions[model_name] = pd.Series(prediction_numbers).map(
            reverse_mapping
        ).to_numpy()
        fitted_models[model_name] = model
        prediction = predictions[model_name]
        results.append({
            "Algoritma": model_name,
            "Accuracy": accuracy_score(y_test, prediction),
            "Precision": precision_score(
                y_test, prediction, pos_label="Stunting", zero_division=0
            ),
            "Recall": recall_score(
                y_test, prediction, pos_label="Stunting", zero_division=0
            ),
            "F1-Score": f1_score(
                y_test, prediction, pos_label="Stunting", zero_division=0
            ),
        })

    results_df = pd.DataFrame(results).sort_values(
        "F1-Score", ascending=False
    )
    results_df.to_csv(
        os.path.join(OUTPUT_DIR, "perbandingan_model_kondisi_1.csv"),
        index=False,
    )
    xgb_parameters = [
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
    ]
    xgb_model = fitted_models["XGBoost"].named_steps["model"]
    xgb_parameters_df = pd.DataFrame({
        "Parameter": xgb_parameters,
        "Ada": "✓",
        "Nilai": [xgb_model.get_params()[parameter] for parameter in xgb_parameters],
    })
    xgb_parameters_df.to_csv(
        os.path.join(OUTPUT_DIR, "parameter_xgboost.csv"),
        index=False,
    )
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.axis("off")
    table = axis.table(
        cellText=xgb_parameters_df.values,
        colLabels=["Parameter XGBoost", "Ada", "Nilai"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.55, 0.18, 0.27],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.1)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#b7cdd1")
        if row == 0:
            cell.set_facecolor("#2f6690")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#e8f1f2" if row % 2 else "#f7faf9")
            if column == 1:
                cell.set_text_props(color="#2f6690", weight="bold", fontsize=16)
    axis.set_title("Konfigurasi Hyperparameter XGBoost", pad=18, weight="bold")
    figure.tight_layout()
    figure.savefig(
        os.path.join(OUTPUT_DIR, "parameter_xgboost.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)
    print("\nPerbandingan model Kondisi 1:")
    print(results_df.to_string(index=False))

    matrix_colors = LinearSegmentedColormap.from_list(
        "matrix_colors",
        ["#e8f1f2", "#3f8f8c", "#2f6690"],
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, (model_name, prediction) in zip(axes.ravel(), predictions.items()):
        matrix = confusion_matrix(
            y_test, prediction, labels=["Normal", "Stunting"]
        )
        ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["Normal", "Stunting"],
        ).plot(
            ax=axis,
            cmap=matrix_colors,
            colorbar=False,
            values_format="d",
        )
        axis.set_title(model_name)
    figure.suptitle("Confusion Matrix - Kondisi 1", fontsize=16)
    figure.tight_layout()
    figure.savefig(
        os.path.join(OUTPUT_DIR, "confusion_matrix_kondisi_1.png"),
        dpi=200,
    )
    plt.close(figure)

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    plot_data = results_df.set_index("Algoritma")[metrics]
    axis = plot_data.plot(
        kind="bar",
        figsize=(14, 7),
        ylim=(0, 1),
        rot=25,
        color=["#2f6690", "#3f8f8c", "#f2a541", "#d95d39"],
    )
    axis.set_title("Perbandingan Performa Model - Kondisi 1")
    axis.set_ylabel("Nilai metrik")
    axis.set_xlabel("")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(title="Metrik", loc="lower right")
    for container in axis.containers:
        axis.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "perbandingan_metrik_kondisi_1.png"),
        dpi=200,
    )
    plt.close()

    best_model_name = results_df.iloc[0]["Algoritma"]
    importance = permutation_importance(
        fitted_models[best_model_name],
        X_test,
        y_test_num,
        scoring="f1",
        n_repeats=30,
        random_state=42,
    )
    importance_df = pd.DataFrame({
        "Fitur": features,
        "ImportanceMean": importance.importances_mean,
        "ImportanceStd": importance.importances_std,
    }).sort_values("ImportanceMean", ascending=False)
    importance_df.to_csv(
        os.path.join(OUTPUT_DIR, "feature_importance_kondisi_1.csv"),
        index=False,
    )

    spectral_importance = importance_df[
        importance_df["Fitur"].isin(spectral_columns)
    ].head(10)
    print(f"\nModel terbaik berdasarkan F1-Score: {best_model_name}")
    print("10 parameter spektral paling berpengaruh:")
    print(spectral_importance.to_string(index=False))

    figure, axis = plt.subplots(figsize=(11, 6))
    axis.barh(
        spectral_importance["Fitur"][::-1],
        spectral_importance["ImportanceMean"][::-1],
        xerr=spectral_importance["ImportanceStd"][::-1],
        color="#2f6690",
        ecolor="#d95d39",
    )
    axis.set_title(
        "Pengaruh Parameter Spektral terhadap Prediksi Stunting "
        f"({best_model_name})"
    )
    axis.set_xlabel("Permutation importance terhadap F1-Score")
    axis.set_ylabel("Parameter spektral")
    figure.tight_layout()
    figure.savefig(
        os.path.join(OUTPUT_DIR, "pengaruh_parameter_spektral.png"),
        dpi=200,
    )
    plt.close(figure)

    print(f"\nSemua output tersimpan di folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
