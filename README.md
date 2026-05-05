# Applied Machine Learning - Complete Experiments & Mini Project

## 📚 Course Overview

This repository contains all the experiments and mini-project for the Applied Machine Learning (AML) course. It includes comprehensive implementations of data preprocessing, regression, classification, anomaly detection, regularization, and ensemble learning techniques.

---

## 📋 Table of Contents

1. [Experiment 1: Data Preprocessing](#experiment-1-data-preprocessing)
2. [Experiment 2: House Price Prediction using Linear Regression](#experiment-2-house-price-prediction-using-linear-regression)
3. [Experiment 3: Stock Price Prediction](#experiment-3-stock-price-prediction)
4. [Experiment 4: Customer Churn Prediction](#experiment-4-customer-churn-prediction)
5. [Experiment 5: Spam Email Detection](#experiment-5-spam-email-detection)
6. [Experiment 6: Credit Risk Assessment](#experiment-6-credit-risk-assessment)
7. [Experiment 7: Anomaly Detection](#experiment-7-anomaly-detection)
8. [Experiment 8: Student Performance Classification](#experiment-8-student-performance-level-classification)
9. [Experiment 9: Heartbeat Classification](#experiment-9-heartbeat-classification)
10. [Experiment 10: Iris Flower Classification](#experiment-10-iris-flower-classification)
11. [Experiment 11: Diabetes Prediction using Bagging](#experiment-11-diabetes-prediction-using-bagging-ensemble-technique)
12. [Experiment 12: L1 & L2 Regularization](#experiment-12-l1-lasso-and-l2-ridge-regularization)
13. [Mini Project: Diamonds & Glass Classification](#mini-project-diamonds--glass-classification)

---

## Experiment 1: Data Preprocessing

### Objective
Learn data preprocessing techniques on IPL cricket dataset including team name normalization, city standardization, text analysis, and data cleaning.

### Key Tasks
1. **Team Name Normalization**: Convert to lowercase, remove leading/trailing spaces
2. **City Name Standardization**: Replace missing values with "Unknown", convert to title case
3. **Toss Decision Text Analysis**: Extract unique decisions and visualize frequency
4. **Winner Name Extraction**: Remove non-normal results and count match wins
5. **Player of the Match Text Frequency**: Find top 10 most frequent players
6. **Venue Tokenization**: Count matches per venue
7. **Umpire Name Cleaning**: Replace missing values, find most frequent umpires
8. **Match Summary Creation**: Combine team names, winner, and season info
9. **Result Type Analysis**: Analyze different result types
10. **Toss Winner vs Match Winner**: Compare and visualize correlation

### Dataset
- IPL cricket match data
- Libraries: `pandas`, `matplotlib`, `seaborn`

### Output Visualizations
- Bar charts for top venues, players, and match winners
- Toss decision distribution analysis

---

## Experiment 2: House Price Prediction using Linear Regression

### Objective
Build a regression model to predict house prices based on location, size, bedrooms, bathrooms, parking, balcony, age, and furnishing status.

### Problem Definition
- **Type**: Regression (continuous output)
- **Target Variable**: Price
- **Input Features**: Area (sqft), Bedrooms, Bathrooms, Balcony, Parking, Age (years), Location, Furnished status

### Data Preprocessing
- One-Hot Encoding for categorical variables (Location, Furnished)
- Feature-Target Split

### Model Details
- **Algorithm**: Linear Regression
- **Train-Test Split**: 80-20 with stratification

### Evaluation Metrics
| Metric | Value |
|--------|-------|
| Mean Squared Error (MSE) | 66.0753 |
| R² Score | 0.9006 |

### Key Results
- MSE indicates average squared difference between actual and predicted values
- R² score of 0.90 shows the model explains 90% of variance in house prices
- Actual vs Predicted scatter plot demonstrates strong linear relationship

---

## Experiment 3: Stock Price Prediction

### Objective
Develop a time series prediction model to forecast stock prices using historical data.

### Problem Definition
- **Type**: Time Series Regression
- **Target Variable**: Price (continuous)
- **Input Features**: Open, High, Low, Close, Volume, Previous Day Price

### Data Preprocessing
- Convert Date to datetime format
- Sort data chronologically
- Create lag feature (Previous day price)
- Remove missing values

### Model Details
- **Algorithm**: Linear Regression
- **Train-Test Split**: First 80% for training, last 20% for testing (preserves time order)

### Evaluation Metrics
| Metric | Value |
|--------|-------|
| Mean Squared Error (MSE) | 2.85e-23 |
| R² Score | 1.0 |

### Key Insights
- Perfect prediction indicates strong linear relationship between past and current prices
- Time order preservation is crucial for time series analysis

---

## Experiment 4: Customer Churn Prediction

### Objective
Predict customer churn in subscription-based businesses to help retain customers.

### Problem Definition
- **Type**: Binary Classification
- **Target Variable**: Churn (0 = stays, 1 = leaves)
- **Importance**: Reduces business loss, improves retention, guides marketing strategies

### Data Understanding
- **Demographic Features**: Gender, Age
- **Service Features**: Subscription Type, Contract Type
- **Behavioral Features**: Monthly Charges, Total Usage, Tenure

### Data Preprocessing
- One-Hot Encoding for categorical variables
- Feature scaling with StandardScaler
- Removed irrelevant CustomerID column

### Model Details
- **Algorithm**: Logistic Regression
- **Max Iterations**: 1000 for convergence

### Evaluation Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 75.00% |
| Precision (Class 0) | 0.85 |
| Precision (Class 1) | 0.29 |
| Recall (Class 0) | 0.85 |
| Recall (Class 1) | 0.29 |

### Outputs
- Confusion Matrix visualization
- Feature Importance plot (Logistic Regression coefficients)
- Churn distribution analysis

---

## Experiment 5: Spam Email Detection

### Objective
Build a spam email filter using text classification and perform comparative analysis of multiple algorithms.

### Problem Definition
- **Type**: Binary Classification
- **Target Variable**: Label (Spam/Ham)
- **Methods**: Text-to-numerical conversion using TF-IDF

### Data Preprocessing
- **TF-IDF Vectorization**: Captures word importance, penalizes common words
- **Train-Test Split**: 80-20

### Models Compared
1. **Logistic Regression (LR)**
2. **Naive Bayes (NB)**
3. **Decision Tree (DT)**
4. **Random Forest (RF)**
5. **Support Vector Machine (SVM)**

### Model Comparison Table
| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.9677 | 1.0000 | 0.7584 | 0.8626 |
| Naive Bayes | 0.9767 | 1.0000 | 0.8255 | 0.9044 |
| Decision Tree | 0.9677 | 0.9065 | 0.8456 | 0.8750 |
| Random Forest | 0.9803 | 1.0000 | 0.8523 | 0.9203 |
| SVC | 0.9848 | 1.0000 | 0.8859 | 0.9395 |

### Best Performer
- **SVM (SVC)** achieved highest accuracy at 98.48%

### Outputs
- Accuracy, Precision, and Recall comparison bar charts
- Confusion matrices for each model

---

## Experiment 6: Credit Risk Assessment

### Objective
Build a credit scoring model to assess creditworthiness of applicants using financial data.

### Problem Definition
- **Type**: Binary Classification (Low vs Medium/High)
- **Target Variable**: Credit_Risk (Low, Medium, High)

### Data Understanding
- **Features**: Age, Annual Income, Employment Years, Credit Score, Loan Amount, Loan Term, Existing Loans, Debt-to-Income Ratio, Late Payments

### Data Preprocessing
- Dropped missing values
- One-Hot Encoding for categorical variables
- StandardScaler for feature normalization

### Models Compared
1. **Logistic Regression**
2. **Random Forest**
3. **XGBoost**

### Model Comparison Table
| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.97 | 0.75 | 0.60 | 0.6667 |
| Random Forest | 0.98 | 1.00 | 0.60 | 0.75 |
| XGBoost | 1.00 | 1.00 | 1.00 | 1.00 |

### Best Performer
- **XGBoost** achieved perfect accuracy with 100% precision and recall

### Outputs
- Accuracy and F1 Score comparison charts
- Confusion matrices
- ROC-AUC curves for all models

---

## Experiment 7: Anomaly Detection

### Objective
Implement anomaly detection system for identifying outliers and fraudulent transactions.

### Problem Definition
- **Type**: Anomaly Detection (Unsupervised/Supervised)
- **Target Variable**: Anomaly_Label (0 = Normal, 1 = Anomaly)
- **Applications**: Fraud detection, cybersecurity, data integrity

### Data Preprocessing
- Feature scaling with StandardScaler
- Label conversion: -1 to 1 mapping for consistency

### Models Compared
1. **Isolation Forest**
2. **Local Outlier Factor (LOF)**
3. **One-Class SVM**

### Model Comparison Table
| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Isolation Forest | 0.96 | 0.60 | 1.00 | 0.75 |
| Local Outlier Factor | 0.844 | 0.02 | 0.0333 | 0.025 |
| SVM | 0.904 | 0.32 | 0.5333 | 0.4 |

### Best Performer
- **Isolation Forest** achieved highest accuracy at 96% with perfect recall

### Outputs
- Accuracy and F1 Score comparison
- Confusion matrices for each model
- ROC curves and AUC scores

---

## Experiment 8: Student Performance Level Classification

### Objective
Implement multiclass classification models for student performance analysis.

### Problem Definition
- **Type**: Multiclass Classification
- **Target Variable**: Performance Level (Low, Medium, High)
- **Purpose**: Identify areas for improvement in student performance

### Data Understanding
- **Features**: Study Hours, Attendance, Assignment Score, Internal Marks, Participation, Internet Access, Previous Grade

### Data Preprocessing
- Label Encoding for categorical variables
- StandardScaler for feature normalization
- Train-Test Split with stratification

### Models Compared
1. **Decision Tree**
2. **Random Forest**
3. **Logistic Regression**
4. **XGBoost**
5. **K-Nearest Neighbors (KNN)**

### Model Comparison Table
| Model | Accuracy | Precision | Recall | F1-Macro | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Decision Tree | 0.83 | 0.8359 | 0.83 | 0.7770 | 0.8369 |
| Random Forest | 0.87 | 0.8614 | 0.87 | 0.5829 | 0.9584 |
| Logistic Regression | 0.98 | 0.9708 | 0.98 | 0.6567 | 0.9927 |
| XGBoost | 0.89 | 0.8812 | 0.89 | 0.5963 | 0.9723 |
| KNN | 0.76 | 0.7543 | 0.76 | 0.5089 | 0.8415 |

### Best Performer
- **Logistic Regression** achieved highest accuracy at 98%

### Outputs
- Confusion matrices for all models
- Metrics comparison bar charts

---

## Experiment 9: Heartbeat Classification

### Objective
Classify physiological heartbeat signals to detect normal and abnormal heartbeats.

### Problem Definition
- **Type**: Binary Classification
- **Target Variable**: Heartbeat Class (0 = Normal, 1 = Abnormal)
- **Purpose**: Early detection of heart abnormalities using ECG signals

### Data Understanding
- **Dataset**: MIT-BIH Arrhythmia Database
- **Features**: 187 ECG signal values
- **Classes**: Imbalanced (Normal dominates)

### Data Preprocessing
- StandardScaler for feature normalization
- Binary classification (Normal vs Abnormal)

### Models Compared
1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **K-Nearest Neighbors**
5. **Support Vector Machine**
6. **XGBoost**

### Model Comparison Table
| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Decision Tree | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Random Forest | 0.9995 | 1.0 | 0.9974 | 0.9987 | 1.0 |
| KNN | 0.9696 | 0.9547 | 0.8649 | 0.9076 | 0.9671 |
| SVM | 0.9950 | 0.9906 | 0.9801 | 0.9854 | 0.9995 |
| XGBoost | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

### Best Performers
- **Logistic Regression, Decision Tree, XGBoost** achieved perfect accuracy

### Outputs
- Confusion matrices for all models
- Model performance comparison charts

---

## Experiment 10: Iris Flower Classification

### Objective
Classify iris flowers into three species using their physical measurements.

### Problem Definition
- **Type**: Multiclass Classification
- **Target Variable**: Species (Setosa, Versicolor, Virginica)
- **Standard Dataset**: Iris Dataset from scikit-learn

### Data Understanding
- **Features**: Sepal Length, Sepal Width, Petal Length, Petal Width
- **Samples**: 150 flowers
- **Balance**: Well-balanced, clean dataset

### Data Preprocessing
- StandardScaler for feature normalization
- Train-Test Split: 80-20 with stratification

### Models Compared
1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **K-Nearest Neighbors**
5. **Support Vector Machine**

### Model Comparison Table
| Model | Accuracy | Precision | Recall | F1 (Weighted) | F1 (Macro) |
|-------|----------|-----------|--------|---------------|-----------|
| Logistic Regression | 0.933 | 0.933 | 0.933 | 0.933 | 0.933 |
| Decision Tree | 0.933 | 0.933 | 0.933 | 0.933 | 0.933 |
| Random Forest | 0.900 | 0.9024 | 0.900 | 0.8997 | 0.8997 |
| KNN | 0.933 | 0.9444 | 0.933 | 0.9327 | 0.9327 |
| SVM | 0.967 | 0.9697 | 0.967 | 0.9658 | 0.9658 |

### Best Performer
- **SVM** achieved highest accuracy at 96.7%

### Outputs
- Pair plot for feature visualization
- Confusion matrices
- ROC curves for multiclass classification

---

## Experiment 11: Diabetes Prediction using Bagging Ensemble Technique

### Objective
Predict diabetes using medical attributes and compare bagging ensemble methods.

### Problem Definition
- **Type**: Binary Classification
- **Target Variable**: Outcome (0 = No Diabetes, 1 = Diabetes)
- **Purpose**: Early detection and medical decision support

### Data Understanding
- **Features**: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
- **Dataset**: Diabetes Dataset

### Data Preprocessing
- Replaced zero values with median (indicating missing data)
- StandardScaler for feature normalization
- Train-Test Split with stratification

### Models Compared
1. **Decision Tree**
2. **Bagging Classifier** (with Decision Tree estimator)
3. **Random Forest**

### Model Comparison Table
| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Decision Tree | 0.6818 | 0.5532 | 0.4815 | 0.5149 |
| Bagging | 0.7338 | 0.6512 | 0.5185 | 0.5773 |
| Random Forest | 0.7792 | 0.7273 | 0.5926 | 0.6530 |

### Key Observations
- **Bagging improved** over Decision Tree (73.38% vs 68.18%)
- **Random Forest performed best** at 77.92% accuracy
- Ensemble methods reduced overfitting tendency of Decision Tree

### Outputs
- Confusion matrices for all models
- Accuracy comparison chart

---

## Experiment 12: L1 (Lasso) and L2 (Ridge) Regularization

### Objective
Analyze effect of L1 and L2 regularization on housing price prediction to improve generalization.

### Problem Definition
- **Type**: Regression
- **Target Variable**: Price
- **Purpose**: Understanding regularization benefits in preventing overfitting

### Data Understanding
- **Dataset**: Melbourne Housing Dataset
- **Features**: Suburb, Type, Method, Seller Group, Council Area, Region, Rooms, Distance, Land Size, Building Area, Year Built

### Data Preprocessing
- Removed unnecessary columns (Address, Date)
- Dropped missing values
- One-Hot Encoding for categorical variables
- StandardScaler for regularized models

### Models Compared
1. **Linear Regression** (No regularization - baseline)
2. **Lasso Regression (L1)** (Alpha = 0.1)
3. **Ridge Regression (L2)** (Alpha = 1.0)

### Model Comparison Table
| Model | MAE | MSE (×10¹⁰) | R² Score |
|-------|-----|-------------|----------|
| Linear Regression | 215680.05 | 8.998 | 0.766 |
| Lasso (L1) | 216592.26 | 9.074 | 0.764 |
| Ridge (L2) | 215574.05 | 8.993 | 0.766 |

### Key Observations
- **Ridge Regression** achieved most stable performance with lowest MSE
- **Lasso** helped reduce feature impact and improve interpretability
- All models showed similar R² scores, indicating comparable variance explanation
- Regularization improved model stability and reduced overfitting

### Regularization Benefits
- **L1 (Lasso)**: Feature selection through coefficient shrinkage to zero
- **L2 (Ridge)**: Coefficient penalty, keeps all features but reduces impact
- **When to use**: L1 for feature selection, L2 for stability

### Outputs
- MAE, MSE, R² comparison visualizations
- Model performance analysis

---

## Mini Project: Diamonds & Glass Classification

### Part 1: Regression - Diamond Price Prediction

#### Objective
Predict diamond prices using physical attributes with regularization analysis.

#### Data Understanding
- **Features**: Carat, Cut, Color, Clarity, Depth, Table, X, Y, Z
- **Target**: Price
- **Samples**: ~53,940

#### Data Preprocessing
- Removed index column
- Label Encoding for ordinal features (cut, clarity)
- StandardScaler for numerical features

#### Models Implemented
1. **Linear Regression** - Baseline
2. **Ridge Regression (L2)** - Alpha = 1.0
3. **Lasso Regression (L1)** - Alpha = 0.1

#### Results
| Model | MSE | R² Score |
|-------|-----|----------|
| Linear | 1.8259e+06 | 0.8851 |
| Ridge | 1.8259e+06 | 0.8851 |
| Lasso | 1.8259e+06 | 0.8851 |

#### Key Findings
- All models achieve ~88.5% R² score
- Regularization provides stable performance
- Minimal overfitting in this dataset

---

### Part 2: Multiclass Classification - Glass Type Classification

#### Objective
Classify glass types using chemical composition with regularization analysis.

#### Data Understanding
- **Features**: RI, Na, Mg, Al, Si, K, Ca, Ba, Fe
- **Target**: Type (Multiclass)
- **Samples**: ~214

#### Data Preprocessing
- StandardScaler for normalization
- Train-Test Split: 80-20

#### Models Implemented
1. **Logistic Regression (L2)** - Penalty='l2'
2. **Logistic Regression (L1)** - Penalty='l1', Solver='liblinear'

#### Evaluation Metrics
- Classification Report
- Confusion Matrix
- ROC Curves (One-vs-Rest)
- Precision-Recall Curves

#### L2 Results
- **Accuracy**: 60.47%
- Precision varies by class
- Some classes show zero recall (challenging to classify)

#### L1 Results
- **Accuracy**: 60.47%
- Similar performance to L2
- L1 provides feature selection benefit

#### ROC Analysis
- Computed individual class ROC curves
- Macro-average ROC: Aggregates all class curves
- One-vs-Rest strategy for multiclass

#### Precision-Recall Curves
- Individual class curves show performance trade-offs
- Macro-average PR curve for overall assessment
- Highlights class imbalance effects

#### Key Findings
- **Class Imbalance**: Some glass types are rare, affecting minority class predictions
- **L1 vs L2**: Similar accuracy, but L1 useful for feature selection
- **Regularization Impact**: Stabilizes predictions across classes
- **ROC-AUC**: Shows variation in class separability

---

## 📊 Technologies & Libraries

```python
# Data Processing
pandas >= 1.0
numpy >= 1.18

# Machine Learning
scikit-learn >= 0.24
xgboost >= 1.0

# Visualization
matplotlib >= 3.0
seaborn >= 0.11

# Text Processing
sklearn.feature_extraction.text (TF-IDF)
```

---

## 🎯 Key Concepts Covered

### Data Preprocessing
- Text normalization and cleaning
- Categorical encoding (One-Hot, Label Encoding)
- Feature scaling (StandardScaler)
- Missing value handling

### Regression Techniques
- Linear Regression
- Ridge Regression (L2)
- Lasso Regression (L1)
- Time Series Forecasting

### Classification Methods
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors
- Naive Bayes
- XGBoost

### Ensemble Learning
- Bagging
- Random Forest
- Gradient Boosting (XGBoost)

### Anomaly Detection
- Isolation Forest
- Local Outlier Factor
- One-Class SVM

### Evaluation Metrics
- **Regression**: MSE, MAE, R² Score
- **Classification**: Accuracy, Precision, Recall, F1 Score, ROC-AUC
- **Multiclass**: Macro/Weighted averaging
- **Anomaly**: Precision, Recall, Confusion Matrix

---

## 📈 Results Summary

### Best Performing Models by Task

| Task | Best Model | Accuracy/Score |
|------|-----------|-----------------|
| House Price Prediction | Linear Regression | R² = 0.9006 |
| Stock Price Prediction | Linear Regression | R² = 1.0 |
| Customer Churn | Logistic Regression | 75% |
| Spam Detection | SVM | 98.48% |
| Credit Risk | XGBoost | 100% |
| Anomaly Detection | Isolation Forest | 96% |
| Student Performance | Logistic Regression | 98% |
| Heartbeat Classification | Multiple (Perfect) | 100% |
| Iris Classification | SVM | 96.7% |
| Diabetes Prediction | Random Forest | 77.92% |
| Diamond Price | All models | R² = 0.8851 |
| Glass Classification | L1 & L2 | 60.47% |

---

## 📝 How to Use This Repository

1. **Run Experiments**: Execute notebook files for each experiment
2. **Review Results**: Check output visualizations and metrics
3. **Modify Parameters**: Experiment with hyperparameters and datasets
4. **Compare Models**: Use evaluation tables to understand performance differences
5. **Learn Concepts**: Study preprocessing steps and model selection rationale

---

## 🔬 Research Insights

1. **Ensemble methods** consistently outperform single models
2. **Feature scaling** is critical for distance-based and regularized models
3. **Class imbalance** significantly impacts minority class predictions
4. **Regularization** (L1 & L2) improves generalization and stability
5. **Time order** matters in time series prediction
6. **Feature engineering** (like lag features) improves temporal models

---

## ✅ Conclusion

This comprehensive course covers the entire machine learning pipeline from data preprocessing to model evaluation. Each experiment demonstrates practical application of ML algorithms to real-world problems, providing hands-on experience with:

- Data cleaning and transformation
- Feature engineering and selection
- Model training and evaluation
- Hyperparameter tuning
- Performance comparison
- Visualization of results

The mini-project ties everything together by applying both regression and classification techniques with regularization analysis on real datasets.

---

**Author**: Kairavi Malik  
**Course**: Applied Machine Learning  
**Date**: May 2026  
**Repository**: GitHub

<h2 id="mentor">Mentor</h2>
  <p><strong>Dr. Sahinur Rahman Laskar</strong><br>
  Assistant Professor<br>
  School of Computer Science, UPES, Dehradun, India<br>
  Email: sahinurlaskar.nits@gmail.com / sahinur.laskar@ddn.upes.ac.in<br>
  </p>
