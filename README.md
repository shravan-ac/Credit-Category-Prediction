# 💳 Credit Category Prediction using Machine Learning

## 📌 Overview

This project predicts a customer's **Credit Category** (Good, Standard,
or Poor) based on financial and behavioral data. It covers data
preprocessing, feature selection, model training, and deployment. The
final model --- a **Random Forest Classifier** optimized with **RFE**
and **GridSearchCV** --- achieved **96.3% accuracy**.

------------------------------------------------------------------------

## ⚙️ Workflow

-   **Data Cleaning:** Handled missing values and inconsistent entries
    using regex and pandas operations.
-   **Encoding & Scaling:** Applied Label/Ordinal Encoding and
    MinMaxScaler to standardize data.
-   **Feature Selection:** Used **Recursive Feature Elimination (RFE)**
    to select top features.
-   **Model Comparison:** Evaluated multiple ML models --- Random
    Forest, XGBoost, Gradient Boosting, AdaBoost, SVM, and others.
-   **Optimization:** Tuned the Random Forest model using
    **GridSearchCV** for improved performance.
-   **Deployment:** Built an interactive **Streamlit app** with
    Pickle-saved model and encoders for real-time predictions.

------------------------------------------------------------------------

## 📊 Results

  Model               Accuracy    Remarks
  ------------------- ----------- -----------------------
  **Random Forest**   **96.16%**   Best performer
  XGBoost             95.52%       Excellent alternative
  Gradient Boosting   94.33%       Strong performer
  Decision Tree       93.03%       Good performer
  AdaBoost            92.00%       Good performer
  GaussianNB          87.53%       Moderate
  KNN                 82.0%        Moderate
  

------------------------------------------------------------------------

## 🧰 Tech Stack

**Python**, **Pandas**, **NumPy**, **Scikit-learn**, **XGBoost**,
**Seaborn**, **Matplotlib**, **Streamlit**

------------------------------------------------------------------------

## 🚀 Run Locally

``` bash
git clone https://github.com/shravan-ac/Credit-Category-Prediction.git
cd Credit-Category-Prediction
pip install -r requirements.txt
streamlit run main.py
```

------------------------------------------------------------------------

## 🏁 Conclusion

An end-to-end ML pipeline demonstrating data cleaning, feature selection
with RFE, model tuning, and deployment. The optimized **Random Forest
model** delivers highly accurate and interpretable credit category
predictions.
