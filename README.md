# **Rossmann Pharmaceuticals Sales Forecasting**

## **Overview**

This project focuses on building an end-to-end sales forecasting solution for Rossmann Pharmaceuticals. The objective is to predict daily sales for each store up to six weeks in advance, based on factors such as promotions, holidays, store types, and competition data. The project involves exploratory data analysis (EDA), machine learning, and deep learning models, and serving the predictions through a REST API.

### **Key Objectives**:
- **Exploratory Data Analysis (EDA)**: Understand the data, identify patterns in customer purchasing behavior, and analyze the influence of different features on sales.
- **Machine Learning**: Implement a tree-based model for predicting store sales.
- **Deep Learning**: Build an LSTM model to perform time-series forecasting.

### **Future Work**
- **Model Deployment**: Develop and deploy a REST API to serve real-time sales predictions.

---

## **Project Structure**
```bash
project-root/
│
├── data/
│   ├── rossman_store_sales.zip
│   ├── cleaned_trained_data.csv
│   ├── merged_test_data.csv 
│   ├── sample_submission.csv
│   ├── test.csv
│   ├── train.csv
│
│
│
├── notebooks/
│   ├── 1_EDA.ipynb              # Exploratory Data Analysis (Task 1)
│   ├── 2_Sales_Predictions.ipynb # ML model building (Task 2)
│   ├── 3_LSTM_Deep_Learning.ipynb # LSTM deep learning model (Task 2.6)
│
├── models/
│   ├── random_forest_model.pkl  # Serialized Random Forest model
│   ├── lstm_model.h5            # Serialized LSTM model
│
├── src/
│   ├── preprocessing.py         # Data preprocessing scripts
│   ├── feature_engineering.py   # Feature extraction scripts
│   ├── train_model.py           # ML model training script
│   ├── deep_learning.py         # LSTM training script
│   ├── logger.py                # Logging for reproducibility
│
├── api/
│   ├── app.py                   # REST API using Flask/FastAPI
│   ├── requirements.txt         # API dependencies
│
└── README.md                    # Project documentation
```

## Installation and Setup
- Step 1: Clone the repository

```git clone https://github.com/Seife1/Pharmaceuticals-Sales-Forecasting.git
cd Pharmaceuticals-Sales-Forecasting```

- Step 2: Install Dependencies

Set up the virtual environment and install the necessary packages.

```python3 -m venv venv
source venv/bin/activate
pip install -r api/requirements.txt```