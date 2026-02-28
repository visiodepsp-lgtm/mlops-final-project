# MLOps Introduction: Final Project
FInal work description in  the [final_project_description.md](final_project_description.md) file.

Student info:
- Full name: Gabriel Quintana Urbano   
- e-mail: visiodepsp@gmail.com  
- Grupo: 2

## Project Name: Diabetes Prediction System Using Machine Learning and MLOps Practices

## 📌 Project Description

This project implements an **end-to-end Machine Learning pipeline following MLOps principles** to predict the likelihood of diabetes using clinical and demographic patient data.

The goal of this work is to demonstrate the complete ML lifecycle:

- Problem definition  
- Data acquisition and preparation  
- Model experimentation and evaluation  
- Model selection (champion model)  
- Model serialization  
- Reproducible project structure  

The dataset used comes from Kaggle’s *Diabetes Prediction Dataset*, which includes relevant medical indicators such as blood glucose level, BMI, HbA1c level, smoking history, age, and other health-related features.

---

## 🎯 Problem Definition

Diabetes is a chronic condition that can lead to severe health complications if not diagnosed early. Using supervised machine learning techniques, this project aims to build a classification model capable of predicting whether a patient has diabetes based on available clinical features.

### Business / Practical Objective

Develop a reproducible and modular ML system that:

- Predicts diabetes with high accuracy
- Can be extended to deployment (API or production environment)
- Follows MLOps best practices for maintainability and reproducibility

---

## 📊 Data Acquisition & Exploration

The dataset includes:

- `age`
- `bmi`
- `blood_glucose_level`
- `HbA1c_level`
- `smoking_history`
- `hypertension`
- `heart_disease`
- Target variable: `diabetes`

During exploratory data analysis (EDA), the following steps were performed:

- Missing value analysis
- Feature distribution visualization
- Correlation analysis
- Baseline clinical rule comparison (HbA1c ≥ 6.5%)

A baseline rule based on medical standards (HbA1c ≥ 6.5%) achieved an accuracy of approximately 0.81, serving as a reference benchmark.

---

## ⚙️ Implementation Details

### 1️⃣ Data Preparation

- Categorical variables encoded using One-Hot Encoding
- Numerical features scaled using StandardScaler
- Train-validation split using `train_test_split`
- Pipeline created using `sklearn.pipeline.Pipeline`

Data transformation logic was modularized for reproducibility.

---

### 2️⃣ Model Experimentation

The following models were trained and evaluated:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine (SVM)

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

### 3️⃣ Champion Model Selection

After comparison, the best-performing model was selected as the **champion model** based on validation performance.

The final model was serialized using:

```python
joblib.dump(model, "models/best_diabetes_model.pkl")
### Serving

a.a.a.a

### Predictions
Running predictions using `curl` command:
```sh
curl -X POST http://127.0.0.1:5000/predict2 -H "Content-Type: application/json" -d '{"experience_score": 4, "test_score": 7, "interview_score": 8}
```

Using Python request library: 
```sh
python src/predict.py
```
