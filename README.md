# Fetal Health Classification Project

## Overview

This project focuses on predicting fetal health status using cardiotocography (CTG) data. The primary objective is to develop and evaluate various machine learning models to classify fetal health into three categories: **Normal**, **Suspect**, and **Pathological**. The final model is deployed on the Azure cloud platform using Azure Machine Learning Studio for real-time prediction.

## Project Structure

- **data/**: Contains the dataset used for model training and evaluation.
- **notebooks/**: Jupyter notebooks with exploratory data analysis (EDA), model training, and evaluation.
- **models/**: Serialized model files saved after training.
- **scripts/**: Python scripts for data preprocessing, model training, and evaluation.
- **output/**: Contains visualizations, reports, and model performance metrics.
- **README.md**: This file providing an overview of the project.

## Dataset

The dataset used for this project was sourced from Kaggle. It contains 21 features derived from CTG data and a target variable (`fetal_health`) that indicates the health status of the fetus:
- **Normal**: Healthy fetus
- **Suspect**: Potential issues requiring closer monitoring
- **Pathological**: Clear health problems requiring immediate medical attention

## Methodology

### 1. Data Preprocessing
- **Exploratory Data Analysis (EDA)**: Analyzed the distribution, central tendency, and correlations of features.
- **Handling Outliers**: Used Z-score method to detect and manage outliers.
- **Feature Scaling**: Standardized features to ensure uniformity across the dataset.
- **Data Splitting**: Split data into training and testing sets with a 70-30 ratio.

### 2. Model Selection and Training
Implemented and compared several machine learning models:
- **Support Vector Machines (SVM)**
- **Multilayer Perceptron (MLP)**
- **Convolutional Neural Network (CNN)**
- **Random Forest Classifier**
- **XGBoost Classifier**

### 3. Clustering Techniques
Experimented with unsupervised learning methods:
- **Gaussian Mixture Model**
- **KMeans Clustering**
- **Agglomerative Clustering**

### 4. Model Evaluation
- Accuracy of each model was compared.
- Hyperparameter tuning was performed for the XGBoost model, achieving a final accuracy of **96%**.

### 5. Model Deployment
- Deployed the best-performing model (XGBoost) on Azure Machine Learning Studio.
- Integrated the model with a RESTful API endpoint for real-time predictions.

## Results

- **Best Model**: XGBoost with an accuracy of **96%** after hyperparameter tuning.
- **Deployment**: Successfully deployed on Azure, accessible via an API for real-time predictions.

## Tools and Technologies

- **Programming Languages**: Python
- **Libraries**: Pandas, Scikit-learn, TensorFlow, XGBoost, Matplotlib, Seaborn
- **Cloud Platforms**: Azure Machine Learning Studio, Azure Databricks
- **Visualization**: Power BI
- **Automation**: Power Automate

## Ethical and Social Impact

This project addresses ethical concerns such as data privacy, fairness in prediction, and transparency. It also considers the social impacts of AI in healthcare, including access to care, patient-provider relationships, and public perception.

## How to Run the Project
Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fetal-health-classification.git
