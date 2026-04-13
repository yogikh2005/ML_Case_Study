# 🩺 Diabetes Prediction System using Machine Learning

A professional Machine Learning project developed in Python to predict whether a patient is diabetic using the **Pima Indians Diabetes Dataset**. The project performs data preprocessing, exploratory data analysis, model training, evaluation, visualization, and model persistence using a **Decision Tree Classifier**.

---

## 📌 Project Overview

This project demonstrates a complete Machine Learning pipeline:

- Dataset loading
- Data preprocessing
- Missing value handling
- Outlier handling
- Data visualization
- Feature preparation
- Model training
- Model evaluation
- Confusion Matrix visualization
- Model saving and loading
- Prediction result export

---

## 🚀 Features

- Clean and modular Python code
- Handles missing values using Median Imputation
- Removes invalid values from medical attributes
- Detects and handles outliers
- Dataset statistics and visualization
- Decision Tree Classification
- Confusion Matrix Heatmap
- Classification Report
- Accuracy Calculation
- Model serialization using Joblib
- Export predictions into CSV

---

## 📂 Project Structure

```
Diabetes-Prediction-System/
│
├── diabetes.csv
├── diabetes.joblib
├── final_diabetes_predictions.csv
├── diabetes_prediction.py
├── README.md
└── requirements.txt
```

---

## 🛠 Technologies Used

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

---

## 📊 Machine Learning Algorithm

**Decision Tree Classifier**

Model Parameters:

- Criterion : Gini
- Max Depth : 2
- Random State : 42

---

## 🔄 Project Workflow

```
Load Dataset
       │
       ▼
Dataset Statistics
       │
       ▼
Data Cleaning
       │
       ▼
Missing Value Handling
       │
       ▼
Train-Test Split
       │
       ▼
Train Decision Tree Model
       │
       ▼
Model Prediction
       │
       ▼
Evaluate Accuracy
       │
       ▼
Confusion Matrix
       │
       ▼
Save Model (.joblib)
       │
       ▼
Load Saved Model
       │
       ▼
Export Prediction CSV
```

---

## 📈 Data Preprocessing

The project preprocesses important medical attributes:

- SkinThickness
- Insulin
- BMI
- BloodPressure
- Glucose

Processing includes:

- Replace invalid zero values
- Convert invalid data into NaN
- Fill missing values using Median
- Handle outliers (Insulin)

---

## 📊 Evaluation Metrics

The model is evaluated using:

- Training Accuracy
- Testing Accuracy
- Classification Report
- Confusion Matrix

---

## 📷 Visualizations

The project generates:

- Outcome Distribution
- Histograms
- Boxplots
- Confusion Matrix Heatmap

---

## 💾 Model Persistence

The trained model is saved using **Joblib**.

```
diabetes.joblib
```

The saved model can later be loaded for prediction without retraining.

---

## 📄 Output Files

| File | Description |
|------|-------------|
| diabetes.joblib | Saved Machine Learning model |
| final_diabetes_predictions.csv | Prediction results |

---

## ▶️ How to Run

### Clone Repository

```bash
git clone https://github.com/yogikh2005/ML_Case_Study.git
```

### Move into Project

```bash
cd Diabetes-Prediction-System
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run

```bash
python diabetes_prediction.py
```

---

## 📦 Required Libraries

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

---

## 🎯 Learning Outcomes

This project demonstrates practical implementation of:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Decision Tree Classification
- Model Evaluation
- Model Serialization
- Data Visualization
- Machine Learning Pipeline

---

## 🔮 Future Improvements

- Random Forest Classifier
- XGBoost
- Hyperparameter Tuning
- Cross Validation
- Flask/FastAPI Web Application
- Streamlit Dashboard
- Docker Deployment

---

## 👨‍💻 Author

**Yogiraj Khaladkar**

Engineering Student | Java Developer | Python Developer | Machine Learning Enthusiast

---

## 📜 License

This project is created for educational and learning purposes.

Feel free to fork, improve, and contribute.