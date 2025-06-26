#!/usr/bin/env python3
"""
Machine Learning Model Explorer
A comprehensive Streamlit app for training, evaluating, and visualizing ML models.

Features:
- Data preprocessing and exploration
- Multiple ML algorithms (Classification, Regression, Clustering)
- Model training and hyperparameter tuning
- Model evaluation and comparison
- Feature importance analysis
- Model interpretation with SHAP and LIME
- Model saving and loading

Author: ML Enthusiast
Version: 1.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import pickle
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Advanced ML Libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Model Interpretation
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="ML Model Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .model-card {
        border: 2px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .best-model {
        border-color: #28a745;
        background-color: #d4edda;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None

def main():
    """Main function for the ML Model Explorer app."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 ML Model Explorer</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your dataset to get started"
    )
    
    if uploaded_file is not None:
        # Load data
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.data = df
            st.success(f"✅ Successfully loaded: {uploaded_file.name}")
            
            # Show basic info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            with col4:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            # Main tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Data Exploration", 
                "🔧 Data Preprocessing", 
                "🤖 Model Training", 
                "📈 Model Evaluation", 
                "💾 Model Management"
            ])
            
            with tab1:
                show_data_exploration(df)
            
            with tab2:
                show_data_preprocessing(df)
            
            with tab3:
                show_model_training(df)
            
            with tab4:
                show_model_evaluation()
            
            with tab5:
                show_model_management()
                
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.info("Please make sure your file is a valid CSV or Excel file.")
    
    else:
        # Welcome message
        st.info("👆 Please upload a dataset using the sidebar to get started!")
        
        # Show features
        st.subheader("🚀 Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📊 Data Exploration**
            - Statistical analysis
            - Data visualization
            - Correlation analysis
            - Missing value analysis
            """)
            
            st.markdown("""
            **🔧 Data Preprocessing**
            - Feature scaling
            - Encoding categorical variables
            - Handling missing values
            - Feature selection
            """)
        
        with col2:
            st.markdown("""
            **🤖 Model Training**
            - Multiple ML algorithms
            - Hyperparameter tuning
            - Cross-validation
            - Model comparison
            """)
            
            st.markdown("""
            **📈 Model Evaluation**
            - Performance metrics
            - Visualization
            - Feature importance
            - Model interpretation
            """)

def show_data_exploration(df):
    """Display data exploration features."""
    st.subheader("📊 Data Exploration")
    
    # Basic info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write(f"**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    
    with col2:
        if st.button("🔄 Refresh Analysis"):
            st.rerun()
    
    # Data preview
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Data types and info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Data Types")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum()
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    with col2:
        st.subheader("📈 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
    
    # Missing values
    if df.isnull().sum().sum() > 0:
        st.subheader("❌ Missing Values Analysis")
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum(),
            'Missing Percentage': (df.isnull().sum() / len(df)) * 100
        }).sort_values('Missing Count', ascending=False)
        
        fig = px.bar(missing_df, x='Column', y='Missing Count', 
                    title='Missing Values by Column')
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 1:
        st.subheader("🔗 Correlation Analysis")
        
        # Correlation matrix
        corr_matrix = df[numerical_cols].corr()
        
        fig = px.imshow(corr_matrix, 
                       title='Correlation Matrix',
                       color_continuous_scale='RdBu',
                       aspect='auto')
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation pairs
        st.write("**Top Correlated Pairs:**")
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
        st.dataframe(corr_df.head(10), use_container_width=True)

def show_data_preprocessing(df):
    """Display data preprocessing features."""
    st.subheader("🔧 Data Preprocessing")
    
    if st.session_state.data is None:
        st.warning("Please upload data first!")
        return
    
    # Target variable selection
    st.subheader("🎯 Target Variable Selection")
    target_col = st.selectbox(
        "Select target variable:",
        df.columns.tolist(),
        help="Choose the column you want to predict"
    )
    
    if target_col:
        # Store target column
        st.session_state.target_col = target_col
        
        # Determine problem type
        if df[target_col].dtype in ['object', 'category']:
            problem_type = "Classification"
            unique_values = df[target_col].nunique()
            st.info(f"🔍 Detected Classification problem with {unique_values} classes")
        else:
            problem_type = "Regression"
            st.info("🔍 Detected Regression problem")
        
        st.session_state.problem_type = problem_type
        
        # Feature selection
        st.subheader("🔍 Feature Selection")
        feature_cols = [col for col in df.columns if col != target_col]
        selected_features = st.multiselect(
            "Select features to use:",
            feature_cols,
            default=feature_cols,
            help="Choose which features to include in your model"
        )
        
        if selected_features:
            st.session_state.selected_features = selected_features
            
            # Data preprocessing options
            st.subheader("⚙️ Preprocessing Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Handle missing values
                st.write("**Missing Values:**")
                missing_strategy = st.selectbox(
                    "Strategy:",
                    ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode"],
                    help="How to handle missing values"
                )
                
                # Scale features
                st.write("**Feature Scaling:**")
                scale_features = st.checkbox("Scale numerical features", value=True)
                
            with col2:
                # Encode categorical variables
                st.write("**Categorical Encoding:**")
                encode_categorical = st.checkbox("Encode categorical variables", value=True)
                
                # Test size
                st.write("**Train/Test Split:**")
                test_size = st.slider("Test size ratio", 0.1, 0.5, 0.2, 0.05)
            
            # Preprocess data
            if st.button("🚀 Preprocess Data"):
                with st.spinner("Preprocessing data..."):
                    processed_data = preprocess_data(
                        df, selected_features, target_col, 
                        missing_strategy, scale_features, encode_categorical, test_size
                    )
                    
                    if processed_data:
                        st.session_state.X_train = processed_data['X_train']
                        st.session_state.X_test = processed_data['X_test']
                        st.session_state.y_train = processed_data['y_train']
                        st.session_state.y_test = processed_data['y_test']
                        st.session_state.feature_names = processed_data['feature_names']
                        st.session_state.scaler = processed_data.get('scaler')
                        st.session_state.encoder = processed_data.get('encoder')
                        
                        st.success("✅ Data preprocessing completed!")
                        
                        # Show preprocessing results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Training set:** {st.session_state.X_train.shape}")
                        with col2:
                            st.write(f"**Test set:** {st.session_state.X_test.shape}")

def preprocess_data(df, features, target, missing_strategy, scale_features, encode_categorical, test_size):
    """Preprocess the data according to selected options."""
    try:
        # Create a copy
        data = df.copy()
        
        # Handle missing values
        if missing_strategy == "Drop rows":
            data = data.dropna(subset=features + [target])
        elif missing_strategy == "Fill with mean":
            data[features] = data[features].fillna(data[features].mean())
        elif missing_strategy == "Fill with median":
            data[features] = data[features].fillna(data[features].median())
        elif missing_strategy == "Fill with mode":
            data[features] = data[features].fillna(data[features].mode().iloc[0])
        
        # Separate features and target
        X = data[features]
        y = data[target]
        
        # Encode categorical variables
        encoder = None
        if encode_categorical:
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                encoder = LabelEncoder()
                for col in categorical_cols:
                    X[col] = encoder.fit_transform(X[col].astype(str))
        
        # Scale features
        scaler = None
        if scale_features:
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                scaler = StandardScaler()
                X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': features,
            'scaler': scaler,
            'encoder': encoder
        }
        
    except Exception as e:
        st.error(f"Error during preprocessing: {str(e)}")
        return None

def show_model_training(df):
    """Display model training features."""
    st.subheader("🤖 Model Training")
    
    if 'X_train' not in st.session_state:
        st.warning("Please preprocess your data first!")
        return
    
    # Model selection
    st.subheader("🎯 Model Selection")
    
    problem_type = st.session_state.problem_type
    
    if problem_type == "Classification":
        models = {
            "Logistic Regression": LogisticRegression(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "SVM": SVC(random_state=42, probability=True),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Naive Bayes": GaussianNB()
        }
        
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = xgb.XGBClassifier(random_state=42)
        if LIGHTGBM_AVAILABLE:
            models["LightGBM"] = lgb.LGBMClassifier(random_state=42)
        if CATBOOST_AVAILABLE:
            models["CatBoost"] = cb.CatBoostClassifier(random_state=42, verbose=False)
            
    else:  # Regression
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(random_state=42),
            "Lasso Regression": Lasso(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "SVR": SVR(),
            "K-Nearest Neighbors": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(random_state=42)
        }
        
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = xgb.XGBRegressor(random_state=42)
        if LIGHTGBM_AVAILABLE:
            models["LightGBM"] = lgb.LGBMRegressor(random_state=42)
        if CATBOOST_AVAILABLE:
            models["CatBoost"] = cb.CatBoostRegressor(random_state=42, verbose=False)
    
    # Model selection
    selected_models = st.multiselect(
        "Select models to train:",
        list(models.keys()),
        default=list(models.keys())[:3],
        help="Choose which models to train and compare"
    )
    
    if selected_models:
        # Training options
        st.subheader("⚙️ Training Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_cross_validation = st.checkbox("Use Cross-Validation", value=True)
            cv_folds = st.slider("CV Folds", 3, 10, 5) if use_cross_validation else 5
            
        with col2:
            use_grid_search = st.checkbox("Use Grid Search", value=False)
            max_iter = st.slider("Max Iterations", 100, 1000, 100) if use_grid_search else 100
        
        # Train models
        if st.button("🚀 Train Models"):
            with st.spinner("Training models..."):
                results = train_models(
                    models, selected_models, use_cross_validation, 
                    cv_folds, use_grid_search, max_iter
                )
                
                if results:
                    st.session_state.models = results['models']
                    st.session_state.results = results['results']
                    st.session_state.best_model = results['best_model']
                    st.session_state.feature_importance = results['feature_importance']
                    
                    st.success("✅ Model training completed!")
                    
                    # Show results
                    show_training_results(results)

def train_models(models, selected_models, use_cv, cv_folds, use_grid_search, max_iter):
    """Train the selected models."""
    try:
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
        problem_type = st.session_state.problem_type
        
        trained_models = {}
        results = []
        feature_importance = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, model_name in enumerate(selected_models):
            status_text.text(f"Training {model_name}...")
            
            model = models[model_name]
            
            # Grid search if enabled
            if use_grid_search:
                param_grid = get_param_grid(model_name, problem_type)
                if param_grid:
                    model = GridSearchCV(model, param_grid, cv=cv_folds, n_jobs=-1)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Get best model if using GridSearchCV
            if use_grid_search and hasattr(model, 'best_estimator_'):
                best_model = model.best_estimator_
            else:
                best_model = model
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            y_pred_proba = None
            if hasattr(best_model, 'predict_proba'):
                y_pred_proba = best_model.predict_proba(X_test)
            
            # Calculate metrics
            metrics = calculate_metrics(y_test, y_pred, y_pred_proba, problem_type)
            
            # Cross-validation score
            cv_score = None
            if use_cv:
                cv_score = cross_val_score(best_model, X_train, y_train, cv=cv_folds, n_jobs=-1)
                metrics['CV Score'] = cv_score.mean()
                metrics['CV Std'] = cv_score.std()
            
            # Feature importance
            if hasattr(best_model, 'feature_importances_'):
                feature_importance[model_name] = best_model.feature_importances_
            elif hasattr(best_model, 'coef_'):
                feature_importance[model_name] = best_model.coef_
            
            # Store results
            trained_models[model_name] = best_model
            results.append({
                'Model': model_name,
                **metrics
            })
            
            progress_bar.progress((i + 1) / len(selected_models))
        
        # Find best model
        if problem_type == "Classification":
            best_model_name = max(results, key=lambda x: x.get('Accuracy', 0))['Model']
        else:
            best_model_name = max(results, key=lambda x: x.get('R² Score', 0))['Model']
        
        best_model = trained_models[best_model_name]
        
        status_text.text("Training completed!")
        
        return {
            'models': trained_models,
            'results': results,
            'best_model': best_model,
            'best_model_name': best_model_name,
            'feature_importance': feature_importance
        }
        
    except Exception as e:
        st.error(f"Error during training: {str(e)}")
        return None

def get_param_grid(model_name, problem_type):
    """Get parameter grid for grid search."""
    if problem_type == "Classification":
        grids = {
            "Logistic Regression": {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2']
            },
            "Random Forest": {
                'n_estimators': [50, 100],
                'max_depth': [None, 10, 20]
            },
            "Gradient Boosting": {
                'n_estimators': [50, 100],
                'learning_rate': [0.1, 0.2]
            },
            "SVM": {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear']
            }
        }
    else:  # Regression
        grids = {
            "Ridge Regression": {
                'alpha': [0.1, 1, 10]
            },
            "Lasso Regression": {
                'alpha': [0.1, 1, 10]
            },
            "Random Forest": {
                'n_estimators': [50, 100],
                'max_depth': [None, 10, 20]
            },
            "Gradient Boosting": {
                'n_estimators': [50, 100],
                'learning_rate': [0.1, 0.2]
            }
        }
    
    return grids.get(model_name, {})

def calculate_metrics(y_true, y_pred, y_pred_proba, problem_type):
    """Calculate evaluation metrics."""
    metrics = {}
    
    if problem_type == "Classification":
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        metrics['Precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['Recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['F1 Score'] = f1_score(y_true, y_pred, average='weighted')
        
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            metrics['ROC AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
    else:  # Regression
        metrics['MSE'] = mean_squared_error(y_true, y_pred)
        metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        metrics['R² Score'] = r2_score(y_true, y_pred)
    
    return metrics

def show_training_results(results):
    """Display training results."""
    st.subheader("📊 Training Results")
    
    # Results table
    results_df = pd.DataFrame(results['results'])
    st.dataframe(results_df, use_container_width=True)
    
    # Best model
    st.success(f"🏆 Best Model: {results['best_model_name']}")
    
    # Feature importance
    if results['feature_importance']:
        st.subheader("🔍 Feature Importance")
        
        feature_names = st.session_state.feature_names
        
        for model_name, importance in results['feature_importance'].items():
            if importance is not None:
                # Handle different importance formats
                if len(importance.shape) > 1:
                    importance = importance.mean(axis=0)
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(importance_df.head(10), x='Importance', y='Feature',
                           title=f'Feature Importance - {model_name}',
                           orientation='h')
                st.plotly_chart(fig, use_container_width=True)

def show_model_evaluation():
    """Display model evaluation features."""
    st.subheader("📈 Model Evaluation")
    
    if 'best_model' not in st.session_state or st.session_state.best_model is None:
        st.warning("Please train models first!")
        return
    
    model = st.session_state.best_model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    problem_type = st.session_state.problem_type
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = None
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)
    
    # Detailed metrics
    st.subheader("📊 Detailed Metrics")
    
    if problem_type == "Classification":
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, 
                       title='Confusion Matrix',
                       color_continuous_scale='Blues',
                       aspect='auto')
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification report
        st.write("**Classification Report:**")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
        
        # ROC curve
        if y_pred_proba is not None and len(np.unique(y_test)) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                   name=f'ROC Curve (AUC = {auc_score:.3f})'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                   name='Random', line=dict(dash='dash')))
            fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', 
                            yaxis_title='True Positive Rate')
            st.plotly_chart(fig, use_container_width=True)
    
    else:  # Regression
        # Actual vs Predicted
        fig = px.scatter(x=y_test, y=y_pred, 
                        title='Actual vs Predicted Values',
                        labels={'x': 'Actual', 'y': 'Predicted'})
        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                               y=[y_test.min(), y_test.max()], 
                               mode='lines', name='Perfect Prediction'))
        st.plotly_chart(fig, use_container_width=True)
        
        # Residuals
        residuals = y_test - y_pred
        fig = px.scatter(x=y_pred, y=residuals,
                        title='Residual Plot',
                        labels={'x': 'Predicted', 'y': 'Residuals'})
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    # Model interpretation
    if SHAP_AVAILABLE:
        st.subheader("🧠 Model Interpretation (SHAP)")
        
        if st.button("Generate SHAP Analysis"):
            with st.spinner("Generating SHAP analysis..."):
                try:
                    # Create SHAP explainer
                    explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else shap.LinearExplainer(model, X_test)
                    shap_values = explainer.shap_values(X_test)
                    
                    # Summary plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
                    st.pyplot(fig)
                    plt.close()
                    
                    # Waterfall plot for a sample
                    if len(shap_values.shape) > 1:
                        shap_values = shap_values[0]
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                                       base_values=explainer.expected_value,
                                                       data=X_test.iloc[0]), show=False)
                    st.pyplot(fig)
                    plt.close()
                    
                except Exception as e:
                    st.warning(f"SHAP analysis failed: {str(e)}")

def show_model_management():
    """Display model management features."""
    st.subheader("💾 Model Management")
    
    if 'best_model' not in st.session_state or st.session_state.best_model is None:
        st.warning("Please train models first!")
        return
    
    # Save model
    st.subheader("💾 Save Model")
    
    model_name = st.text_input("Model name:", value="my_model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save Model"):
            try:
                # Save model
                model_path = f"{model_name}.pkl"
                joblib.dump(st.session_state.best_model, model_path)
                
                # Save preprocessing objects
                preprocess_path = f"{model_name}_preprocess.pkl"
                preprocess_data = {
                    'scaler': st.session_state.scaler,
                    'encoder': st.session_state.encoder,
                    'feature_names': st.session_state.feature_names,
                    'target_col': st.session_state.target_col,
                    'problem_type': st.session_state.problem_type
                }
                joblib.dump(preprocess_data, preprocess_path)
                
                st.success(f"✅ Model saved as {model_path}")
                
                # Download link
                with open(model_path, "rb") as f:
                    bytes_data = f.read()
                st.download_button(
                    label="📥 Download Model",
                    data=bytes_data,
                    file_name=model_path,
                    mime="application/octet-stream"
                )
                
            except Exception as e:
                st.error(f"Error saving model: {str(e)}")
    
    with col2:
        # Load model
        st.subheader("📂 Load Model")
        uploaded_model = st.file_uploader(
            "Upload a saved model (.pkl):",
            type=['pkl'],
            help="Upload a previously saved model"
        )
        
        if uploaded_model:
            try:
                model = joblib.load(uploaded_model)
                st.success("✅ Model loaded successfully!")
                st.session_state.loaded_model = model
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")

if __name__ == "__main__":
    main() 