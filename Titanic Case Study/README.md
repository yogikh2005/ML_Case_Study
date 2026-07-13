# Titanic Survival Prediction using Logistic Regression & Pipeline

This project demonstrates an end-to-end Machine Learning pipeline for predicting whether a passenger survived the Titanic disaster using the Titanic dataset.

The project follows industrial best practices by:

- Automating preprocessing with Scikit-learn Pipeline
- Handling missing values in Age, Fare, and Embarked columns
- Encoding categorical variables
- Training a Logistic Regression classifier
- Saving & loading trained models using Joblib
- Visualizing model performance using a Confusion Matrix

---

# Dependencies

Install the required Python packages before running the project:

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

---

# Dataset Information

Dataset: Titanic Dataset

### Features Used

- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked

### Target

- 0 = Did Not Survive
- 1 = Survived

---

# Workflow

## Data Preparation

- Load Titanic CSV dataset
- Remove unnecessary columns:
  - PassengerId
  - Name
  - Cabin
  - zero
- Handle missing values
- Convert data into proper numeric format
- Encode categorical features using One-Hot Encoding

---

## Train-Test Split

- Split dataset into:
  - 70% Training Data
  - 30% Testing Data

---

## Pipeline Construction

Pipeline contains:

### Step 1

Data preprocessing

- Missing value handling
- Feature encoding

### Step 2

Logistic Regression Classifier

```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

---

## Model Training & Evaluation

Evaluation Metrics:

- Training Accuracy
- Testing Accuracy
- Classification Report
- Confusion Matrix

---

## Visualization

The project displays a Confusion Matrix using Matplotlib for better understanding of model performance.

---

## Model Saving & Loading

The trained pipeline is saved using Joblib.

```python
joblib.dump(model, "Titanic_pipeline.joblib")
```

Load later without retraining:

```python
from titanic_pipeline import load_model

model = load_model("Titanic_pipeline.joblib")
```

---

# Running the Project

Run the project:

```bash
python titanic_pipeline.py
```

Example Output

```
Train Accuracy :: 0.78

Test Accuracy  :: 0.77

```

Model saved successfully:

```
Titanic_pipeline.joblib
```

Loaded model prediction:

```
Loaded model prediction for first test sample: 1
```

---

# Visualizations

- Confusion Matrix using Matplotlib

---

# Model Storage

The trained model is saved as:

```
Titanic_pipeline.joblib
```

The saved model can be loaded anytime for prediction without retraining.

---

# Sample Prediction

```python
sample = test_x.iloc[[0]]

prediction = model.predict(sample)

print("Prediction:", prediction[0])

# 0 = Did Not Survive
# 1 = Survived
```

---

# Project Structure

```
Titanic-Survival-Prediction/
│
├── TitanicDataset.csv
├── titanic_pipeline.py
├── Titanic_pipeline.joblib
├── README.md
└── requirements.txt
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

---

# Author

**Yogiraj Mohan Khaladkar**

**Date:** 14/03/2026