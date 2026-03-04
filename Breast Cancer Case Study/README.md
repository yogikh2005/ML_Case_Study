# 🩺 Breast Cancer Prediction using Random Forest (Machine Learning)

A Machine Learning project that predicts whether a breast cancer tumor is **Benign** or **Malignant** using the **Random Forest Classifier** from Scikit-learn.

The project follows a complete Machine Learning pipeline including data preprocessing, missing value handling, model training, evaluation, feature importance visualization, and model persistence using Joblib. :contentReference[oaicite:0]{index=0}

---

## 📌 Features

- Load breast cancer dataset from CSV
- Handle missing values using `SimpleImputer`
- Convert invalid values (`?`) into `NaN`
- Split dataset into training and testing sets
- Train Random Forest Classifier
- Evaluate model performance using:
  - Accuracy
  - Confusion Matrix
  - Classification Report
- Visualize Feature Importance
- Save trained model using Joblib
- Reload saved model for future predictions

---

## 🛠️ Technologies Used

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

---

## 📂 Project Structure

```
Breast Cancer Case Study/
│
├── breast_canser.csv           			# Input dataset
├── breast-cancer-Output.csv    			# Output file (optional)
├── bc_rf_pipeline.joblib       			# Saved trained model
├── BreastCanserRandomForestClassifier.py   # Main Python source code
├── Feature Importances.png
├── README.md
└── requirements.txt
```

---

## ⚙️ Machine Learning Workflow

1. Load Dataset
2. Display Dataset Statistics
3. Prepare Features and Target
4. Handle Missing Values
5. Split Dataset (70% Train / 30% Test)
6. Build ML Pipeline
7. Train Random Forest Model
8. Evaluate Model
9. Plot Feature Importance
10. Save Model
11. Load Saved Model and Predict

The implementation builds a Scikit-learn `Pipeline` with `SimpleImputer` and `RandomForestClassifier`, trains it, evaluates it, and saves the trained pipeline with Joblib. :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}

---

## 📊 Model Evaluation

The project evaluates the model using:

- Training Accuracy
- Testing Accuracy
- Classification Report
- Confusion Matrix
- Feature Importance Graph

---
## 📊 Feature Importance

The Random Forest model ranks each feature based on its contribution to the prediction.

<p align="center">
  <img src="Feature%20Importances.png" alt="Feature Importances" width="700"/>
</p>

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yogikh2005/ML_Case_Study.git
```

Move into the project directory

```bash
cd Breast-Cancer-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python main.py
```

---

## 📦 Requirements

```text
pandas
numpy
matplotlib
scikit-learn
joblib
```

You can install all packages using:

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

---

## 📈 Output

The program displays:

- Dataset statistics
- Training accuracy
- Testing accuracy
- Classification report
- Confusion matrix
- Feature importance graph
- Saved model file (`bc_rf_pipeline.joblib`)
- Prediction using the loaded model

---

## 💡 Future Improvements

- Hyperparameter tuning using GridSearchCV
- Compare multiple ML algorithms
- Deploy using Flask or Streamlit
- Build a web interface
- Add ROC Curve and AUC Score
- Perform Cross Validation

---

## 🎯 Learning Objectives

This project demonstrates:

- Data preprocessing
- Missing value handling
- Machine Learning pipelines
- Random Forest classification
- Model evaluation
- Feature importance analysis
- Model serialization with Joblib

---

## 👨‍💻 Author

**Yogiraj Khaladkar**

Computer Engineering Student | Machine Learning Developer | Java Developer

---

## ⭐ If you found this project useful

Please consider giving it a ⭐ on GitHub!