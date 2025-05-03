# ✅ Works in both Colab and Streamlit (comment st lines in Colab)
try:
    import streamlit as st
    is_streamlit = True
except ImportError:
    is_streamlit = False

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score

import io

# ---------------------------------------------
# 🎨 Streamlit UI (only runs if in Streamlit)
if is_streamlit:
    # 🔧 Set up page config with new name
    st.set_page_config(page_title="SmartFinanceAI", layout="centered")

    # 🌈 Custom background color (via emoji or optional markdown block)
    st.markdown(
        """
        <style>
        .main {
            background-color: #f0f8ff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # 🧠 App Title and GIF
    st.title("💰 SmartFinanceAI: Predictive Insights for Your Data")
    st.image("https://media.giphy.com/media/3o7aD2saalBwwftBIY/giphy.gif", use_column_width=True)
    st.markdown("Upload a financial dataset and apply ML models to uncover trends and predictions.")

    uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])
else:
    from google.colab import files
    uploaded = files.upload()
    uploaded_file = list(uploaded.keys())[0]

# ---------------------------------------------
# 📊 Load Data
if uploaded_file:
    if is_streamlit:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Data Loaded!")
        st.dataframe(df.head())
    else:
        df = pd.read_csv(uploaded_file)
        print("✅ Data Loaded from:", uploaded_file)
        print(df.head())

    # 🧹 Preprocessing
    df_clean = df.dropna()
    if is_streamlit:
        st.info("🧹 Removed missing values.")
        st.write(df_clean.isnull().sum())
    else:
        print("🧹 Cleaned Missing Values:")
        print(df_clean.isnull().sum())

    # 📈 Feature Selection
    features = df_clean.select_dtypes(include=np.number)
    if features.shape[1] < 2:
        raise Exception("Need at least 2 numeric columns (features + target)")
    
    X = features.iloc[:, :-1]
    y = features.iloc[:, -1]

    # ✂️ Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if is_streamlit:
        st.success("✅ Data Split (80/20)")
        st.write("Training Size:", X_train.shape[0])
        st.write("Testing Size:", X_test.shape[0])
    else:
        print("✅ Data Split:")
        print("Train size:", X_train.shape)
        print("Test size:", X_test.shape)

    # 📚 Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    if is_streamlit:
        st.success("✅ Model Trained: Linear Regression")
        st.write("📉 Mean Squared Error:", round(mse, 3))
        st.write("📈 R² Score:", round(r2, 3))
    else:
        print("✅ Linear Regression Trained")
        print("MSE:", mse)
        print("R2 Score:", r2)

    # 📊 Visualization
    if is_streamlit:
        st.subheader("📊 Actual vs Predicted")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.7, color='green')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)
    else:
        plt.figure(figsize=(6, 4))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted")
        plt.grid(True)
        plt.show()
