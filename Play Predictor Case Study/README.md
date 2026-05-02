# 🎮 Play Predictor using K-Nearest Neighbors (KNN)

A Machine Learning case study that predicts whether a game should be played based on **Weather** and **Temperature** conditions using the **K-Nearest Neighbors (KNN) Classification Algorithm**.

This project demonstrates a complete Machine Learning classification workflow, including preprocessing, model training, evaluation, model persistence, and prediction.

---

## 📌 Project Overview

This project performs the following tasks:

- Load the dataset
- Perform Exploratory Data Analysis (EDA)
- Encode categorical features using Label Encoding
- Visualize the dataset
- Split the dataset into training and testing sets
- Train a K-Nearest Neighbors (KNN) classifier
- Predict outcomes
- Evaluate model performance
- Display Confusion Matrix
- Save the trained model and encoders
- Load the saved model
- Predict a new sample
- Export prediction results to a CSV file

---

## 📂 Project Structure

```
PLAY PREDICTOR CASE STUDY
│
├── Play_Predictor.py
├── PlayPredictor.csv
├── PlayPredictor_Output.csv
├── PlayPredictor_knn.joblib
├── PlayPredictor_encoders.joblib
├── README.md
└── requirements.txt
```

---

## 🛠 Technologies Used

- Python 3.x
- Pandas
- Matplotlib
- Scikit-learn
- Joblib

---

## 📦 Required Libraries

Install all required packages using:

```bash
pip install -r requirements.txt
```

or

```bash
pip install pandas matplotlib scikit-learn joblib
```

---

## ▶️ How to Run

Clone the repository:

```bash
git clone https://github.com/yogikh2005/ML_Case_Study.git
```

Go to the project folder:

```bash
cd ML_Case_Study
```

Run the application:

```bash
python Play_Predictor.py
```

---

## 📊 Dataset Features

The model uses the following features:

| Feature | Description |
|----------|-------------|
| Whether | Weather condition |
| Temperature | Temperature condition |

### Target Variable

| Target | Description |
|--------|-------------|
| Play | Yes / No |

---

## ⚙️ Machine Learning Workflow

1. Load Dataset
2. Perform Data Analysis
3. Encode Categorical Features
4. Visualize Dataset
5. Split Dataset
6. Train KNN Model
7. Predict Results
8. Evaluate Model
9. Display Confusion Matrix
10. Save Model & Encoders
11. Load Saved Model
12. Predict New Sample
13. Export Predictions to CSV

---

## 🤖 Model Details

### Algorithm

- K-Nearest Neighbors (KNN)

### Model Parameters

```python
KNeighborsClassifier(
    n_neighbors=3
)
```

### Data Preprocessing

- Label Encoding
- Train-Test Split

---

## 📈 Model Evaluation

The model is evaluated using:

- Accuracy Score
- Classification Report
- Confusion Matrix

---

## 💾 Output Files

### Trained Model

```
PlayPredictor_knn.joblib
```

### Saved Encoders

```
PlayPredictor_encoders.joblib
```

### Prediction Output

```
PlayPredictor_Output.csv
```

---

## 📁 Generated Files

After successful execution, the following files are generated:

- PlayPredictor_Output.csv
- PlayPredictor_knn.joblib
- PlayPredictor_encoders.joblib

---

## 📚 Concepts Covered

- Supervised Machine Learning
- K-Nearest Neighbors (KNN)
- Classification
- Label Encoding
- Data Preprocessing
- Train-Test Split
- Model Evaluation
- Confusion Matrix
- Joblib Model Persistence
- Prediction using Saved Model

---

## 🚀 Future Improvements

- Hyperparameter Tuning
- Cross Validation
- Feature Scaling
- GridSearchCV
- Interactive User Interface
- Flask API Deployment
- Streamlit Web Application

---

## 👨‍💻 Author

**Yogiraj Khaladkar**

Engineering Student | Machine Learning Developer

---

## ⭐ Repository

If you found this project helpful, please consider giving it a ⭐ on GitHub.