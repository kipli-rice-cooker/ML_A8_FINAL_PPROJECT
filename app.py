import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    balanced_accuracy_score, classification_report, confusion_matrix, accuracy_score
)

st.set_page_config(page_title="Prediksi Kebutuhan Irigasi", layout="wide")

ARTIFACT_DIR = "artifacts"


@st.cache_resource
def load_artifacts():
    path = ARTIFACT_DIR
    artifacts = {
        "rf_model": joblib.load(os.path.join(path, "model_rf.pkl")),
        "lgbm_model": joblib.load(os.path.join(path, "model_lgbm.pkl")),
        "label_encoder": joblib.load(os.path.join(path, "label_encoder.pkl")),
        "encoded_columns": joblib.load(os.path.join(path, "encoded_columns.pkl")),
        "numeric_features": joblib.load(os.path.join(path, "numeric_features.pkl")),
        "categorical_features": joblib.load(os.path.join(path, "categorical_features.pkl")),
        "category_options": joblib.load(os.path.join(path, "category_options.pkl")),
        "numeric_ranges": joblib.load(os.path.join(path, "numeric_ranges.pkl")),
    }
    val_path = os.path.join(path, "validation_data.parquet")
    artifacts["validation_data"] = pd.read_parquet(val_path) if os.path.exists(val_path) else None
    return artifacts


try:
    A = load_artifacts()
    ARTIFACTS_OK = True
except Exception as e:
    ARTIFACTS_OK = False
    LOAD_ERROR = e

CLASS_COLORS = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}

# Tujuh fitur penting yang ditampilkan sebagai input wajib pada Mode Sederhana.
# Fitur lain otomatis diisi dengan nilai rata-rata (numerik) atau nilai paling umum (kategorikal).
IMPORTANT_NUMERIC = ["Rainfall_mm", "Soil_Moisture", "Temperature_C", "Humidity"]
IMPORTANT_CATEGORICAL = ["Crop_Type", "Season", "Soil_Type"]
IMPORTANT_FEATURES = IMPORTANT_NUMERIC + IMPORTANT_CATEGORICAL


def get_category_mode():
    """category_mode.pkl mungkin belum ada di artifacts versi lama -> fallback ke opsi pertama."""
    try:
        return joblib.load(os.path.join(ARTIFACT_DIR, "category_mode.pkl"))
    except Exception:
        return {col: opts[0] for col, opts in A["category_options"].items()}


# format fitur yang sama persis dengan saat training

def encode_input(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_enc = pd.get_dummies(df_raw, columns=A["categorical_features"], drop_first=True, dtype=int)
    df_enc = df_enc.reindex(columns=A["encoded_columns"], fill_value=0)
    return df_enc


def predict_all(df_raw: pd.DataFrame):
    df_enc = encode_input(df_raw)
    le = A["label_encoder"]

    pred_rf = le.inverse_transform(A["rf_model"].predict(df_enc))
    pred_lgbm = le.inverse_transform(A["lgbm_model"].predict(df_enc))

    proba_rf = A["rf_model"].predict_proba(df_enc)
    proba_lgbm = A["lgbm_model"].predict_proba(df_enc)

    result = df_raw.copy()
    result["Prediksi_RandomForest"] = pred_rf
    result["Prediksi_LightGBM"] = pred_lgbm
    for i, cls in enumerate(le.classes_):
        result[f"Prob_RF_{cls}"] = proba_rf[:, i]
    for i, cls in enumerate(le.classes_):
        result[f"Prob_LGBM_{cls}"] = proba_lgbm[:, i]
    return result


def class_badge(label):
    color = CLASS_COLORS.get(label, "#7f8c8d")
    st.markdown(
        f"""<div style="background-color:{color};padding:14px;border-radius:8px;
        text-align:center;color:white;font-size:22px;font-weight:bold;">{label}</div>""",
        unsafe_allow_html=True,
    )

# SIDEBAR NAVIGATION

st.sidebar.title("🍆 Menu")
page = st.sidebar.radio(
    "Pilih Halaman",
    ["Beranda", "Prediksi Manual", "Prediksi via CSV", "Dashboard Perbandingan Model"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Model yang digunakan: **Random Forest** & **LightGBM**")
st.sidebar.caption("Kelompok 8 - Prediksi Kebutuhan Irigasi")

if not ARTIFACTS_OK:
    st.error(
        "Artefak model belum ditemukan di folder 'artifacts/'. "
        "Jalankan dulu cell penyimpanan artefak di Colab (01_simpan_artifacts_colab.py) "
        "sebelum menjalankan Streamlit ini."
    )
    st.code(str(LOAD_ERROR))
    st.stop()

CATEGORY_MODE = get_category_mode()

# HALAMAN: BERANDA

if page == "Beranda":
    st.title("💦Sistem Prediksi Kebutuhan Irigasi")
    st.markdown(
        """
        Aplikasi ini memprediksi tingkat kebutuhan irigasi (**Low / Medium / High**)
        berdasarkan data sensor lingkungan dan agronomi, menggunakan dua model machine
        learning terbaik dari hasil penelitian kelompok 8: **Random Forest** dan **LightGBM**.

        Gunakan menu di sebelah kiri untuk:
        - **Prediksi Manual** → isi nilai sensor satu per satu lalu dapatkan prediksi langsung.
        - **Prediksi via CSV** → upload banyak baris data sekaligus untuk diprediksi.
        - **Dashboard Perbandingan Model** → lihat performa Random Forest vs LightGBM
        (balanced accuracy, classification report, confusion matrix) pada data validasi.
        """
    )
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Jumlah fitur input", len(A["numeric_features"]) + len(A["categorical_features"]))
    with col2:
        st.metric("Kelas target", "Low / Medium / High")

# HALAMAN: PREDIKSI MANUAL

elif page == "Prediksi Manual":
    st.title("Prediksi Manual")

    mode = st.radio(
        "Mode Input",
        ["🟢 Sederhana (7 fitur penting)", "🔧 Lengkap (semua 19 fitur)"],
        horizontal=True,
    )
    simple_mode = mode.startswith("🟢")

    if simple_mode:
        st.write(
            "Cukup isi **7 fitur paling berpengaruh** di bawah ini. "
            "Fitur lainnya otomatis diisi dengan nilai rata-rata (numerik) "
            "atau nilai paling umum (kategorikal) dari data penelitian."
        )
    else:
        st.write("Isi nilai sensor & informasi lahan di bawah ini, lalu klik **Prediksi**.")

    with st.form("manual_form"):
        numeric_to_show = IMPORTANT_NUMERIC if simple_mode else A["numeric_features"]
        categorical_to_show = IMPORTANT_CATEGORICAL if simple_mode else A["categorical_features"]

        st.subheader("Fitur Numerik")
        numeric_values = {}
        n_cols = 3
        cols = st.columns(n_cols)
        for i, feat in enumerate(numeric_to_show):
            lo, hi, mean = A["numeric_ranges"][feat]
            with cols[i % n_cols]:
                numeric_values[feat] = st.number_input(
                    feat, min_value=float(lo), max_value=float(hi), value=float(round(mean, 2))
                )

        st.subheader("Fitur Kategorikal")
        categorical_values = {}
        cols2 = st.columns(n_cols)
        for i, feat in enumerate(categorical_to_show):
            options = A["category_options"][feat]
            default_idx = options.index(CATEGORY_MODE[feat]) if CATEGORY_MODE[feat] in options else 0
            with cols2[i % n_cols]:
                categorical_values[feat] = st.selectbox(feat, options, index=default_idx)

        if simple_mode:
            with st.expander("Lihat nilai default fitur lain (otomatis terisi)"):
                other_numeric = [f for f in A["numeric_features"] if f not in IMPORTANT_NUMERIC]
                other_categorical = [f for f in A["categorical_features"] if f not in IMPORTANT_CATEGORICAL]
                default_info = {}
                for f in other_numeric:
                    default_info[f] = round(A["numeric_ranges"][f][2], 2)
                for f in other_categorical:
                    default_info[f] = CATEGORY_MODE[f]
                st.dataframe(pd.DataFrame(default_info.items(), columns=["Fitur", "Nilai Default"]))

        submitted = st.form_submit_button("🔍 Prediksi")

    if submitted:
        # Lengkapi fitur yang tidak ditampilkan dengan nilai default (mean/mode)
        input_dict = {}
        for feat in A["numeric_features"]:
            input_dict[feat] = numeric_values.get(feat, A["numeric_ranges"][feat][2])
        for feat in A["categorical_features"]:
            input_dict[feat] = categorical_values.get(feat, CATEGORY_MODE[feat])

        df_input = pd.DataFrame([input_dict])
        result = predict_all(df_input)

        st.markdown("### Hasil Prediksi")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Random Forest**")
            class_badge(result["Prediksi_RandomForest"].iloc[0])
        with c2:
            st.markdown("**LightGBM**")
            class_badge(result["Prediksi_LightGBM"].iloc[0])

        st.markdown("### Probabilitas Tiap Kelas")
        le = A["label_encoder"]
        prob_df = pd.DataFrame({
            "Kelas": le.classes_,
            "Random Forest": [result[f"Prob_RF_{c}"].iloc[0] for c in le.classes_],
            "LightGBM": [result[f"Prob_LGBM_{c}"].iloc[0] for c in le.classes_],
        }).set_index("Kelas")
        st.bar_chart(prob_df)

        with st.expander("Lihat seluruh data input yang dipakai untuk prediksi (termasuk nilai default)"):
            st.dataframe(df_input)

# HALAMAN: PREDIKSI VIA CSV

elif page == "Prediksi via CSV":
    st.title("Prediksi via Upload CSV")
    st.write(
        "Upload file CSV dengan kolom yang sama seperti dataset training "
        "(tanpa kolom `id` dan `Irrigation_Need`)."
    )

    expected_cols = A["numeric_features"] + A["categorical_features"]
    st.caption("Kolom yang dibutuhkan: " + ", ".join(expected_cols))

    file = st.file_uploader("Upload file CSV", type=["csv"])

    if file is not None:
        df_csv = pd.read_csv(file)
        missing = [c for c in expected_cols if c not in df_csv.columns]
        if missing:
            st.error(f"Kolom berikut tidak ditemukan di file CSV: {missing}")
        else:
            with st.spinner("Memproses prediksi..."):
                result = predict_all(df_csv[expected_cols])

            st.success(f"Berhasil memprediksi {len(result)} baris data.")
            st.dataframe(result.head(50))

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Distribusi Prediksi - Random Forest**")
                st.bar_chart(result["Prediksi_RandomForest"].value_counts())
            with col2:
                st.markdown("**Distribusi Prediksi - LightGBM**")
                st.bar_chart(result["Prediksi_LightGBM"].value_counts())

            csv_out = result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Hasil Prediksi (CSV)",
                data=csv_out,
                file_name="hasil_prediksi_irigasi.csv",
                mime="text/csv",
            )

# HALAMAN: DASHBOARD PERBANDINGAN MODEL

elif page == "Dashboard Perbandingan Model":
    st.title("Dashboard Perbandingan Model: Random Forest vs LightGBM")

    val_data = A["validation_data"]
    if val_data is None:
        st.warning(
            "Data validasi (validation_data.parquet) tidak ditemukan di folder artifacts/. "
            "Menampilkan ringkasan hasil penelitian (statis) sebagai gantinya."
        )
        summary = pd.DataFrame({
            "Model": ["Logistic Regression", "KNN (k=11)", "Random Forest", "XGBoost", "LightGBM"],
            "Balanced Accuracy": [0.7938, 0.6094, 0.9663, 0.9634, 0.9677],
            "Accuracy": [0.79, 0.84, 0.98, 0.98, 0.98],
        })
        st.dataframe(summary, use_container_width=True)
        st.bar_chart(summary.set_index("Model")["Balanced Accuracy"])
    else:
        X_val = val_data.drop(columns=["__target__"])
        y_val = val_data["__target__"]
        le = A["label_encoder"]

        pred_rf = A["rf_model"].predict(X_val)
        pred_lgbm = A["lgbm_model"].predict(X_val)

        bal_acc_rf = balanced_accuracy_score(y_val, pred_rf)
        bal_acc_lgbm = balanced_accuracy_score(y_val, pred_lgbm)
        acc_rf = accuracy_score(y_val, pred_rf)
        acc_lgbm = accuracy_score(y_val, pred_lgbm)

        st.subheader("Ringkasan Metrik (dihitung ulang dari data validasi)")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Random Forest - Balanced Accuracy", f"{bal_acc_rf:.4f}")
            st.metric("Random Forest - Accuracy", f"{acc_rf:.4f}")
        with c2:
            st.metric("LightGBM - Balanced Accuracy", f"{bal_acc_lgbm:.4f}")
            st.metric("LightGBM - Accuracy", f"{acc_lgbm:.4f}")

        st.markdown("---")
        st.subheader("Classification Report")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Random Forest**")
            report_rf = classification_report(
                y_val, pred_rf, target_names=le.classes_, output_dict=True
            )
            st.dataframe(pd.DataFrame(report_rf).T.round(3))
        with c2:
            st.markdown("**LightGBM**")
            report_lgbm = classification_report(
                y_val, pred_lgbm, target_names=le.classes_, output_dict=True
            )
            st.dataframe(pd.DataFrame(report_lgbm).T.round(3))

        st.markdown("---")
        st.subheader("Confusion Matrix")
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 4))
            cm = confusion_matrix(y_val, pred_rf)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Random Forest")
            st.pyplot(fig)
        with c2:
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            cm2 = confusion_matrix(y_val, pred_lgbm)
            sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens",
                        xticklabels=le.classes_, yticklabels=le.classes_, ax=ax2)
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("True")
            ax2.set_title("LightGBM")
            st.pyplot(fig2)

        st.markdown("---")
        st.subheader("Perbandingan Balanced Accuracy")
        comp_df = pd.DataFrame({
            "Model": ["Random Forest", "LightGBM"],
            "Balanced Accuracy": [bal_acc_rf, bal_acc_lgbm],
        }).set_index("Model")
        st.bar_chart(comp_df)