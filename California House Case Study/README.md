# 🏠 California Housing Price Prediction using Bagging Regressor

A Machine Learning Regression project that predicts California housing prices using the **Bagging Regressor** ensemble algorithm with **Decision Tree Regressor** as the base estimator.

The project demonstrates the complete Machine Learning workflow including data preprocessing, exploratory analysis, model training, evaluation, visualization, model serialization, and prediction result export.

---

## 📌 Project Overview

This project builds a regression model capable of predicting house prices using various housing-related features from the California Housing dataset.

The implementation follows a modular programming approach where each stage of the machine learning pipeline is implemented as an individual function.

---

## 🚀 Features

- Modular Python implementation
- Clean and well-documented source code
- Dataset statistics and analysis
- Data visualization
- Train-Test Split
- Decision Tree Regressor
- Bagging Ensemble Learning
- Model Evaluation using Regression Metrics
- Actual vs Predicted Visualization
- Model Saving using Joblib
- Model Loading for Future Prediction
- Prediction Output CSV Generation

---

# 📂 Project Structure

```
California-Housing-Bagging-Regressor/
│
├── california_housing.csv
├── california_housing_Output.csv
├── california_housing.joblib
├── Bagging_Regration_california_housing.py
├── README.md
└── requirements.txt
```

---

# 📊 Dataset

Dataset used:

**California Housing Dataset**

The dataset contains housing information collected from different districts of California.

### Input Features

- MedInc
- HouseAge
- AveRooms
- AveBedrms
- Population
- AveOccup
- Latitude
- Longitude

### Target

```
House Price (target)
```

---

# 🛠 Technologies Used

- Python 3.x
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- Joblib

---

# 📦 Required Libraries

```bash
pip install pandas
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install joblib
```

or

```bash
pip install -r requirements.txt
```

---

# ⚙️ Machine Learning Workflow

```
Load Dataset
        │
        ▼
Dataset Statistics
        │
        ▼
Feature Selection
        │
        ▼
Train-Test Split
        │
        ▼
Decision Tree Regressor
        │
        ▼
Bagging Regressor
        │
        ▼
Model Training
        │
        ▼
Prediction
        │
        ▼
Model Evaluation
        │
        ▼
Visualization
        │
        ▼
Save Model
        │
        ▼
Load Model
        │
        ▼
Export Prediction CSV
```

---

# 📈 Model Used

## Bagging Regressor

Bagging (Bootstrap Aggregating) is an ensemble learning algorithm that combines predictions from multiple Decision Trees to improve prediction accuracy and reduce overfitting.

### Base Estimator

```
DecisionTreeRegressor
```

### Ensemble Parameters

| Parameter | Value |
|-----------|--------|
| Estimator | DecisionTreeRegressor |
| n_estimators | 100 |
| max_depth | 15 |
| min_samples_split | 5 |
| random_state | 42 |

---

# 📊 Evaluation Metrics

The regression model is evaluated using:

- Mean Squared Error (MSE)
- R² Score (Coefficient of Determination)

Example Output

```
Train MSE : 0.08
Train R²  : 0.96

Test MSE  : 0.27
Test R²   : 0.79
```

---

# 📉 Visualization

The project generates the following visualizations:

### Target Distribution

- Histogram
- Kernel Density Estimation (KDE)

### Model Evaluation

- Actual vs Predicted Scatter Plot
- Perfect Prediction Reference Line

---

# 💾 Model Serialization

The trained model is saved using Joblib.

```
california_housing.joblib
```

The saved model can later be loaded for prediction without retraining.

---

# 📄 Output Files

### Trained Model

```
california_housing.joblib
```

### Prediction Output

```
california_housing_Output.csv
```

Output CSV contains:

| Feature Columns | Actual | Predicted |
|----------------|---------|-----------|

---

# ▶️ How to Run

Clone the repository

```bash
git clone git clone https://github.com/yogikh2005/ML_Case_Study.git
```

Go to project directory

```bash
cd California House Case Study
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python Bagging_Regration_california_housing.py
```

---

# 📷 Sample Output

```
STEP 1 : Load Dataset

STEP 2 : Dataset Statistics

STEP 3 : Select Features & Target

STEP 4 : Split Dataset

STEP 5 : Train Bagging Regressor

STEP 6 : Predict

STEP 7 : Evaluate Model

Train MSE : 0.08
Train R²  : 0.96

Test MSE  : 0.27
Test R²   : 0.79

STEP 8 : Plot Actual vs Predicted

STEP 9 : Save Model

STEP 10 : Load Model

STEP 11 : Save Prediction CSV
```

---

# 📚 Learning Outcomes

This project demonstrates:

- Regression Analysis
- Ensemble Learning
- Bagging Algorithm
- Decision Tree Regression
- Data Visualization
- Model Serialization
- Performance Evaluation
- Modular Python Programming
- Machine Learning Pipeline Design

---

# 👨‍💻 Author

**Yogiraj Khaladkar**

Engineering Student | Machine Learning Developer

---

# ⭐ If you found this project useful

Please consider giving this repository a ⭐ on GitHub.