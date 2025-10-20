# Kaiburr Assessment 2025 — Task 5: Consumer Complaint Classification

## Overview

This repository contains the solution for Task 5, focused on multi-class text classification of US consumer finance complaints. The code builds an end-to-end machine learning pipeline for pre-processing, feature extraction, model training, evaluation, and prediction using the `consumer_complaints.csv` dataset from Kaggle.

---

## Steps

1. **Dataset Download**
   - Downloaded via Kaggle CLI: `kaggle/us-consumer-finance-complaints`
   - Contains open US consumer complaint narrative texts and product category labels.

2. **Dependencies**
   - Install via:
     ```
     pip install pandas scikit-learn matplotlib nltk seaborn
     ```

3. **Code**
   - Complete pipeline in `consumer_complaint_classification.py`
   - Key steps: data cleaning, TF-IDF extraction, four ML models (LR, SVM, RF, NB), evaluation.

4. **How to Run**
   - Place `consumer_complaints.csv` and the script in the project root.
   - Run:
     ```
     python consumer_complaint_classification.py
     ```

---

## Results

Below are the saved results and evaluation visualizations, each included as PNG from the `screenshots/` folder. Every screenshot contains system date/time and my username in the window for verification.

---

### 1. Model Comparison Table

- Model metrics (accuracy, precision, recall, F1) for all classifiers as output by the classification report.
<img width="1918" height="1078" alt="Task-5-TABLE" src="https://github.com/user-attachments/assets/78cc8171-fa5c-466b-8644-ce0cf3e2568b" />

---

### 2. Confusion Matrices by Model

Confusion matrices for each classifier, allowing visual inspection of prediction breakdown for each product category:

- **Logistic Regression:**  
<img width="1917" height="1078" alt="Task-5-Logistic Regression Confusion Matrix" src="https://github.com/user-attachments/assets/e521ead0-051c-4b97-8a17-8c21b6af4ec9" />

- **Naive Bayes:**  
<img width="1918" height="1078" alt="Task-5-NaiveBayesConfusionMatrix" src="https://github.com/user-attachments/assets/97a3d6c2-adf4-473a-90f4-74f119e5859a" />

- **Random Forest:**  
<img width="1918" height="1078" alt="Task-5-RandomForestConfusionMatrix" src="https://github.com/user-attachments/assets/77df38e2-9441-43e3-bf8b-9585b79484e4" />

- **SVM:**  
<img width="1918" height="1078" alt="Task-5-SVMConfusionMatrix" src="https://github.com/user-attachments/assets/51d4049c-ae30-4ea2-bde4-487e8e0fe181" />

---

### 3. Best Performing Model Proof

Sample output printout and screenshot of the best-performing model selection, along with its prediction evidence.

-<img width="1918" height="1076" alt="Task-5-BestPerformingModel" src="https://github.com/user-attachments/assets/3b63f489-1b49-42ca-a1ef-4d8ec35cc6d2" />

---

## Key Files

- **consumer_complaint_classification.py** — Complete ML pipeline with all steps.
- **consumer_complaints.csv** — Dataset from Kaggle.
- 
---

## Author

Final Year B.Tech CCE  
Shyam Anand  
October 2025

---

## License

This project submitted for Kaiburr Assessment 2025.
