# 🎓 Student Performance Clustering using K-Means

A Machine Learning case study that groups students into different performance categories using the **K-Means Clustering Algorithm**.

The project uses students' academic information such as grades, study time, failures, and absences to automatically discover similar student groups without using predefined labels.

---

## 📌 Project Overview

This project demonstrates an **Unsupervised Machine Learning** workflow using **K-Means Clustering**.

The application performs the following tasks:

- Load the student dataset
- Display dataset statistics
- Select important academic features
- Scale the data using StandardScaler
- Determine the optimal number of clusters using the Elbow Method
- Train a K-Means clustering model
- Predict cluster labels
- Save the trained model and scaler
- Load the saved model and scaler
- Predict the cluster of a new student
- Export clustered data into a CSV file

---

## 📂 Project Structure

```
STUDENT PERFORMANCE CASE STUDY CLUSTER
│
├── Student Performance Cluster.py
├── student-mat.csv
├── student-mat_Output.csv
├── Final_Student_Predictions.csv
├── student_kmeans.joblib
├── student_scaler.joblib
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

Install the required libraries using:

```bash
pip install -r requirements.txt
```

or

```bash
pip install pandas matplotlib scikit-learn joblib
```

---

## ▶️ How to Run

Clone the repository

```bash
git clone https://github.com/yogikh2005/ML_Case_Study.git
```

Go to the project directory

```bash
cd ML_Case_Study
```

Run the application

```bash
python "Student Performance Cluster.py"
```

---

## 📊 Dataset Features

The clustering model uses the following student attributes:

| Feature | Description |
|----------|-------------|
| G1 | First Period Grade |
| G2 | Second Period Grade |
| G3 | Final Grade |
| studytime | Weekly Study Time |
| failures | Number of Past Class Failures |
| absences | Number of School Absences |

---

## ⚙️ Machine Learning Workflow

1. Load Dataset
2. Display Dataset Statistics
3. Select Features
4. Scale Features using StandardScaler
5. Determine Optimal Clusters (Elbow Method)
6. Train K-Means Model
7. Predict Student Clusters
8. Save Model and Scaler
9. Load Saved Model
10. Predict Cluster for New Student
11. Export Results to CSV

---

## 🤖 Model Details

### Algorithm

- K-Means Clustering

### Parameters

```python
KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)
```

### Feature Scaling

```python
StandardScaler()
```

---

## 💾 Output Files

### Trained K-Means Model

```
student_kmeans.joblib
```

### Saved Scaler

```
student_scaler.joblib
```

### Prediction Output

```
Final_Student_Predictions.csv
```

---

## 📁 Generated Files

After successful execution, the following files are generated:

- Final_Student_Predictions.csv
- student_kmeans.joblib
- student_scaler.joblib

---

## 📚 Concepts Covered

- Unsupervised Machine Learning
- K-Means Clustering
- Feature Scaling
- StandardScaler
- Elbow Method
- Model Persistence using Joblib
- Cluster Prediction
- CSV File Export

---

## 🚀 Future Improvements

- Automatic Cluster Naming
- Cluster Quality Evaluation
- Silhouette Score Analysis
- Hyperparameter Optimization
- Interactive Dashboard
- Web Application Deployment using Flask or Streamlit

---

## 👨‍💻 Author

**Yogiraj Khaladkar**

Engineering Student | Java Developer | Machine Learning Enthusiast

GitHub:

https://github.com/yogikh2005

---

## ⭐ Repository

If you found this project useful, please consider giving it a ⭐ on GitHub.