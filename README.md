# 🤖 ML Model Explorer

A comprehensive Streamlit application for training, evaluating, and visualizing machine learning models. This tool provides an interactive interface for the entire ML workflow, from data preprocessing to model deployment.

## 🚀 Features

### 📊 Data Exploration
- **Statistical Analysis**: Comprehensive data statistics and summaries
- **Data Visualization**: Interactive charts and plots
- **Correlation Analysis**: Feature correlation matrices and heatmaps
- **Missing Value Analysis**: Detection and visualization of missing data

### 🔧 Data Preprocessing
- **Feature Selection**: Choose which features to include in your model
- **Target Variable Selection**: Automatically detect classification vs regression problems
- **Missing Value Handling**: Multiple strategies (drop, mean, median, mode)
- **Feature Scaling**: Standardize numerical features
- **Categorical Encoding**: Encode categorical variables
- **Train/Test Split**: Configurable data splitting

### 🤖 Model Training
- **Multiple Algorithms**: Support for various ML algorithms
- **Classification Models**:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors
  - Decision Tree
  - Naive Bayes
  - XGBoost (if available)
  - LightGBM (if available)
  - CatBoost (if available)

- **Regression Models**:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Random Forest
  - Gradient Boosting
  - Support Vector Regression (SVR)
  - K-Nearest Neighbors
  - Decision Tree
  - XGBoost (if available)
  - LightGBM (if available)
  - CatBoost (if available)

- **Advanced Features**:
  - Cross-validation
  - Hyperparameter tuning with Grid Search
  - Model comparison and ranking
  - Progress tracking during training

### 📈 Model Evaluation
- **Performance Metrics**:
  - Classification: Accuracy, Precision, Recall, F1-Score, ROC-AUC
  - Regression: MSE, RMSE, MAE, R² Score
- **Visualizations**:
  - Confusion matrices
  - ROC curves
  - Actual vs Predicted plots
  - Residual plots
- **Feature Importance**: Analyze which features contribute most to predictions
- **Model Interpretation**: SHAP analysis for model explainability

### 💾 Model Management
- **Save Models**: Export trained models as pickle files
- **Load Models**: Import previously saved models
- **Download Models**: Easy model sharing and deployment

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. **Clone or download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Optional Dependencies
Some advanced features require additional libraries:
- **XGBoost**: `pip install xgboost`
- **LightGBM**: `pip install lightgbm`
- **CatBoost**: `pip install catboost`
- **SHAP**: `pip install shap`
- **LIME**: `pip install lime`

## 🚀 Usage

### Running the Application
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Step-by-Step Workflow

1. **📁 Upload Data**
   - Use the sidebar to upload your CSV or Excel file
   - The app will automatically detect the data format

2. **📊 Explore Data**
   - View data preview and statistics
   - Analyze correlations and missing values
   - Understand your dataset structure

3. **🔧 Preprocess Data**
   - Select your target variable
   - Choose features to include
   - Configure preprocessing options
   - Handle missing values and scale features

4. **🤖 Train Models**
   - Select multiple models to compare
   - Configure training options (cross-validation, grid search)
   - Monitor training progress
   - View model comparison results

5. **📈 Evaluate Models**
   - Analyze performance metrics
   - View detailed visualizations
   - Generate feature importance plots
   - Perform SHAP analysis for model interpretation

6. **💾 Save Models**
   - Export your best model
   - Download for deployment
   - Load previously saved models

## 📋 Supported Data Formats

- **CSV files** (.csv)
- **Excel files** (.xlsx, .xls)

## 🎯 Use Cases

### Classification Problems
- Customer churn prediction
- Spam detection
- Disease diagnosis
- Credit risk assessment
- Image classification

### Regression Problems
- House price prediction
- Sales forecasting
- Temperature prediction
- Stock price prediction
- Demand forecasting

## 🔧 Configuration Options

### Data Preprocessing
- **Missing Value Strategies**: Drop rows, fill with mean/median/mode
- **Feature Scaling**: StandardScaler for numerical features
- **Categorical Encoding**: Label encoding for categorical variables
- **Train/Test Split**: Configurable ratio (10% to 50%)

### Model Training
- **Cross-Validation**: 3-10 folds
- **Grid Search**: Hyperparameter optimization
- **Model Selection**: Choose from available algorithms
- **Training Options**: Configure iterations and parameters

## 📊 Output and Results

### Model Performance
- **Comparison Table**: Side-by-side model metrics
- **Best Model Selection**: Automatic identification of top performer
- **Feature Importance**: Visual representation of feature contributions

### Visualizations
- **Confusion Matrices**: For classification problems
- **ROC Curves**: Model discrimination ability
- **Residual Plots**: For regression problems
- **Correlation Heatmaps**: Feature relationships
- **SHAP Plots**: Model interpretability

## 🚀 Deployment

### Local Deployment
```bash
streamlit run app.py --server.port 8501
```

### Streamlit Cloud Deployment
1. Push your code to GitHub
2. Connect your repository to Streamlit Cloud
3. Deploy with one click

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

2. **Memory Issues**
   - Reduce dataset size for large files
   - Use fewer models in training
   - Increase system memory

3. **Training Time**
   - Disable grid search for faster training
   - Reduce cross-validation folds
   - Use simpler models

4. **SHAP Analysis Fails**
   - Install SHAP: `pip install shap`
   - Ensure model supports SHAP analysis
   - Check data compatibility

### Performance Tips

- **Large Datasets**: Use sampling for initial exploration
- **Model Selection**: Start with simpler models (Linear Regression, Random Forest)
- **Feature Selection**: Remove irrelevant features to improve performance
- **Cross-Validation**: Use 5-fold CV for good balance of speed and accuracy

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Streamlit** for the amazing web framework
- **Scikit-learn** for the comprehensive ML library
- **Plotly** for interactive visualizations
- **SHAP** for model interpretability
- **Pandas** and **NumPy** for data manipulation

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section
2. Review the documentation
3. Open an issue on GitHub
4. Contact the development team

---

**Happy Modeling! 🎉**

*Built with ❤️ for the ML community* 