# Customer Churn Analysis & Prediction Model

This repository contains the complete machine learning workflow and deployment for predicting telecom customer churn, designed to reduce churn rate, maximize customer retention, and increase Customer Lifetime Value (CLV).

## Project Structure

- **`notebook.ipynb`**: Complete Jupyter notebook detailing the exploratory data analysis (EDA), outlier clipping, feature engineering, correlation analysis, hyperparameter tuning (Grid Search & Random Search), and model selection.
- **`app.py`**: A premium, interactive Streamlit web application that serves the trained model for making predictions.
- **`best_model.pkl`**: Serialized Scikit-Learn Pipeline containing the pre-processing scaler and the tuned Decision Tree classifier.
- **`requirements.txt`**: List of dependencies required to run the notebook and web application.
- **`customer_data.csv`**: Historical customer dataset used for training, evaluation, and batch prediction testing.
- **`train_pipeline.py`**: Automates pipeline preprocessing and Grid Search tuning to export the serialized model.

---

## Getting Started

### 1. Installation

Install all required python packages:
```bash
pip install -r requirements.txt
```

### 2. Training the Model Pipeline
To retrain the model and save the pipeline:
```bash
python train_pipeline.py
```

### 3. Running the Streamlit Application
Launch the web portal locally:
```bash
streamlit run app.py
```

---

## Features & Usage

### Single Customer Prediction
Input custom account details, call minutes, international plans, voicemail subscription, and customer service calls to view real-time churn risk indicators and probabilities.

### Batch Predictions
Upload a customer database in CSV format (matching the schema of `customer_data.csv`) to calculate churn predictions in bulk and download the annotated dataset.