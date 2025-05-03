# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import datetime
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    auc
)
from pandas.tseries.offsets import BDay
import requests
from io import StringIO
import time

# =============================================
# CONFIGURATION & THEMING
# =============================================
st.set_page_config(
    page_title="Stock Trend Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .sidebar .sidebar-content {
        background-color: #1A1D24;
        background-image: linear-gradient(315deg, #1A1D24 0%, #23272F 74%);
    }
    .stButton>button {
        border: 2px solid #4CAF50;
        border-radius: 8px;
        color: white;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049 !important;
        transform: scale(1.05);
    }
    .stSelectbox>div>div>div>input {
        background-color: #2D3036 !important;
        color: white !important;
    }
    .stAlert {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-box {
        background: #1E2229;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================
# UTILITY FUNCTIONS
# =============================================
def add_technical_indicators(df):
    """Calculate technical indicators for stock data"""
    # Moving Averages
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['Upper_Bollinger'] = df['MA_20'] + (df['Close'].rolling(20).std() * 2)
    df['Lower_Bollinger'] = df['MA_20'] - (df['Close'].rolling(20).std() * 2)
    
    return df.dropna()

def prepare_target(df):
    """Create binary target variable (1 if next day's close is higher)"""
    df['Tomorrow_Close'] = df['Close'].shift(-1)
    df['Target'] = (df['Tomorrow_Close'] > df['Close']).astype(int)
    return df.dropna()

def download_link(object_to_download, download_filename):
    """Generate download link for files"""
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    
    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">📥 Download {download_filename}</a>'

# =============================================
# SESSION STATE MANAGEMENT
# =============================================
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'features' not in st.session_state:
    st.session_state.features = []
if 'target' not in st.session_state:
    st.session_state.target = ''
if 'train_test_data' not in st.session_state:
    st.session_state.train_test_data = None

# =============================================
# MAIN APP INTERFACE
# =============================================
def main():
    # Welcome Section
    st.title("📈 AI Stock Trend Predictor")
    st.markdown("---")
    
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("""
                ### Predict Stock Market Trends with Machine Learning
                Upload your dataset or fetch real-time market data to predict 
                whether a stock will rise or fall using Logistic Regression.
                """)
        with col2:
            st.image("https://media.giphy.com/media/RiR0DDbO1Jf8A/giphy.gif", width=250)
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        data_source = st.radio("Data Source:", 
                              ("Yahoo Finance", "Kragle CSV Upload"))
        
        if data_source == "Yahoo Finance":
            st.subheader("Stock Data Settings")
            ticker = st.text_input("Ticker Symbol:", "AAPL")
            start_date = st.date_input("Start Date:", datetime.date(2020, 1, 1))
            end_date = st.date_input("End Date:", datetime.date.today())
            
            if st.button("🚀 Fetch Stock Data"):
                with st.spinner("Downloading market data..."):
                    try:
                        df = yf.download(ticker, start=start_date, end=end_date)
                        if df.empty:
                            st.error("No data found for this ticker/date range!")
                        else:
                            df = add_technical_indicators(df)
                            df = prepare_target(df)
                            st.session_state.data = df.reset_index()
                            st.success(f"Successfully loaded {len(df)} records!")
                    except Exception as e:
                        st.error(f"Error fetching data: {str(e)}")
        
        else:  # Kragle CSV Upload
            st.subheader("Dataset Upload")
            uploaded_file = st.file_uploader("Choose CSV File", type="csv")
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    if all(col in df.columns for col in required_cols):
                        df = add_technical_indicators(df)
                        df = prepare_target(df)
                        st.session_state.data = df
                        st.success("Dataset successfully processed!")
                    else:
                        st.error("CSV missing required columns!")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    # Main Workflow Steps
    st.markdown("---")
    st.header("🧠 Machine Learning Pipeline")
    
    # Step 1: Data Preview
    if st.button("1. Show Data Preview"):
        if st.session_state.data is not None:
            st.subheader("Processed Data Preview")
            with st.expander("View Full Dataset"):
                st.dataframe(st.session_state.data.style.background_gradient(cmap='viridis'))
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(px.line(st.session_state_data, 
                                      x='Date', y='Close', 
                                      title=f"Price History"))
            with col2:
                st.plotly_chart(px.histogram(st.session_state_data, 
                                           x='Target', 
                                           title="Class Distribution"))
        else:
            st.error("Please load data first!")
    
    # Step 2: Feature Selection
    if st.button("2. Feature Engineering"):
        if st.session_state.data is not None:
            st.subheader("Feature Selection")
            numerical_features = st.session_state_data.select_dtypes(include=[np.number]).columns.tolist()
            numerical_features.remove('Target')
            
            selected_features = st.multiselect(
                "Select Features for Model:",
                numerical_features,
                default=['MA_20', 'RSI', 'MACD', 'Volume']
            )
            
            st.session_state.features = selected_features
            st.session_state.target = 'Target'
            
            with st.expander("Feature Analysis"):
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(px.imshow(st.session_state_data[selected_features].corr(), 
                                          title="Feature Correlation Matrix"))
                with col2:
                    st.plotly_chart(px.box(st.session_state_data[selected_features], 
                                        title="Feature Distributions"))
            
            st.success(f"{len(selected_features)} features selected!")
        else:
            st.error("Load data first!")
    
    # Step 3: Train/Test Split
    if st.button("3. Prepare Training Data"):
        if st.session_state.features:
            st.subheader("Data Splitting")
            split_size = st.slider("Test Set Size:", 0.1, 0.4, 0.2)
            
            X = st.session_state_data[st.session_state.features]
            y = st.session_state_data[st.session_state.target]
            
            # Feature Scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, 
                test_size=split_size, 
                random_state=42,
                stratify=y
            )
            
            st.session_state.train_test_data = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'scaler': scaler
            }
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(px.pie(
                    names=['Training', 'Testing'],
                    values=[len(X_train), len(X_test)],
                    title="Train/Test Split"
                ))
            with col2:
                st.write("**Class Balance:**")
                st.write(y_train.value_counts(normalize=True))
            
            st.success("Data split completed with feature scaling!")
        else:
            st.error("Select features first!")
    
    # Step 4: Model Training
    if st.button("4. Train Model"):
        if st.session_state.train_test_data:
            st.subheader("Model Training")
            
            with st.spinner("Training Logistic Regression Model..."):
                model = LogisticRegression(
                    max_iter=1000,
                    class_weight='balanced',
                    random_state=42
                )
                
                model.fit(
                    st.session_state.train_test_data['X_train'],
                    st.session_state.train_test_data['y_train']
                )
                
                st.session_state.model = model
                train_acc = model.score(
                    st.session_state.train_test_data['X_train'],
                    st.session_state.train_test_data['y_train']
                )
                
                st.session_state.train_test_data['y_pred'] = model.predict(
                    st.session_state.train_test_data['X_test']
                )
                st.session_state.train_test_data['y_proba'] = model.predict_proba(
                    st.session_state.train_test_data['X_test']
                )[:, 1]
            
            st.markdown(f"""
                <div class="metric-box">
                    <h3>Training Accuracy: {train_acc:.2%}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            st.plotly_chart(px.bar(
                x=st.session_state.features,
                y=model.coef_[0],
                title="Feature Importance"
            ))
            
            st.success("Model training completed!")
        else:
            st.error("Prepare training data first!")
    
    # Step 5: Model Evaluation
    if st.button("5. Evaluate Model"):
        if st.session_state.model:
            st.subheader("Model Performance")
            
            y_test = st.session_state.train_test_data['y_test']
            y_pred = st.session_state.train_test_data['y_pred']
            y_proba = st.session_state.train_test_data['y_proba']
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            # Create tabs for different metrics
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Accuracy", "📉 Confusion Matrix", 
                                            "📈 ROC Curve", "📋 Classification Report"])
            
            with tab1:
                st.plotly_chart(px.bar(
                    x=['Accuracy'],
                    y=[accuracy],
                    title=f"Test Accuracy: {accuracy:.2%}"
                ))
            
            with tab2:
                fig = px.imshow(cm, 
                              labels=dict(x="Predicted", y="Actual"),
                              x=['Down', 'Up'],
                              y=['Down', 'Up'],
                              text_auto=True)
                fig.update_layout(title="Confusion Matrix")
                st.plotly_chart(fig)
            
            with tab3:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr,
                                       mode='lines',
                                       name=f'ROC Curve (AUC = {roc_auc:.2f})'))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                       mode='lines',
                                       name='Random',
                                       line=dict(dash='dash')))
                fig.update_layout(title="ROC Curve",
                                xaxis_title="False Positive Rate",
                                yaxis_title="True Positive Rate")
                st.plotly_chart(fig)
            
            with tab4:
                st.code(report)
            
            st.success("Evaluation completed!")
        else:
            st.error("Train model first!")
    
    # Step 6: Predictions & Visualization
    if st.button("6. Generate Predictions"):
        if st.session_state.model:
            st.subheader("Live Predictions")
            
            # Create synthetic future data
            last_date = st.session_state_data['Date'].iloc[-1]
            future_dates = pd.date_range(
                start=last_date + BDay(1),
                periods=5,
                freq='B'
            )
            
            # Generate predictions
            future_preds = st.session_state.model.predict(
                st.session_state.train_test_data['scaler'].transform(
                    st.session_state_data[st.session_state.features].iloc[-5:]
                )
            )
            
            # Create prediction dataframe
            pred_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Trend': ['Up' if x == 1 else 'Down' for x in future_preds],
                'Confidence': np.max(
                    st.session_state.model.predict_proba(
                        st.session_state.train_test_data['scaler'].transform(
                            st.session_state_data[st.session_state.features].iloc[-5:]
                        )
                    ), axis=1)
            })
            
            # Show predictions
            st.plotly_chart(px.bar(
                pred_df,
                x='Date',
                y='Confidence',
                color='Predicted_Trend',
                title="Next 5 Trading Days Predictions",
                color_discrete_map={'Up': '#4CAF50', 'Down': '#FF5252'}
            ))
            
            st.dataframe(pred_df.style.applymap(
                lambda x: 'color: #4CAF50' if x == 'Up' else 'color: #FF5252',
                subset=['Predicted_Trend']
            ))
            
            st.markdown(download_link(pred_df, "predictions.csv"), 
                        unsafe_allow_html=True)
            
            st.success("Predictions generated!")
        else:
            st.error("Train model first!")

# =============================================
# RUN THE APP
# =============================================
if __name__ == "__main__":
    main()
