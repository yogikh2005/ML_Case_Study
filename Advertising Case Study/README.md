# 📺 Advertising Sales Prediction using Linear Regression

A Machine Learning case study that predicts **product sales** based on advertising expenditure across **TV**, **Radio**, and **Newspaper** using the **Linear Regression** algorithm.

This project demonstrates a complete regression workflow including data preprocessing, model training, evaluation, visualization, model persistence, and prediction.

---

## 📌 Project Overview

This project performs the following tasks:

- Load the advertising dataset
- Remove unnecessary columns
- Analyze the dataset
- Check missing values
- Calculate feature correlations
- Split the dataset into training and testing sets
- Train a Linear Regression model
- Evaluate model performance
- Display Actual vs Predicted visualization
- Display model coefficients
- Save the trained model
- Load the saved model
- Predict sales for new data
- Export prediction results to a CSV file

---

## 📂 Project Structure

```text
ADVERTISING SALES PREDICTION
│
├── AdvertisingPrediction.py
├── Advertising.csv
├── Advertising_Output.csv
├── Advertising_LR_Model.joblib
├── README.md
└── requirements.txt
```

---

## 🛠 Technologies Used

- Python 3.x
- Pandas
- NumPy
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
pip install pandas numpy matplotlib scikit-learn joblib
```

---

## ▶️ How to Run

Clone the repository

```bash
git clone https://github.com/yogikh2005/ML_Case_Study.git
```

Go to the project folder

```bash
cd ML_Case_Study
```

Run the application

```bash
python AdvertisingPrediction.py
```

---

## 📊 Dataset

The project uses the **Advertising.csv** dataset.

The dataset contains advertising budgets spent on different marketing channels and the corresponding product sales.

---

## 📄 Dataset Features

| Feature | Description |
|----------|-------------|
| TV | Advertising budget spent on TV |
| Radio | Advertising budget spent on Radio |
| Newspaper | Advertising budget spent on Newspaper |

### Target Variable

| Target | Description |
|--------|-------------|
| Sales | Product Sales |

---

## ⚙️ Machine Learning Workflow

1. Load Dataset
2. Remove Unwanted Columns
3. Analyze Dataset
4. Check Missing Values
5. Calculate Correlation Matrix
6. Split Dataset
7. Train Linear Regression Model
8. Predict Sales
9. Evaluate Model
10. Display Model Coefficients
11. Plot Actual vs Predicted Values
12. Save Model using Joblib
13. Load Saved Model
14. Export Predictions to CSV

---

## 🤖 Model Details

### Algorithm

- Linear Regression

### Features Used

- TV
- Radio
- Newspaper

---

## 📈 Model Evaluation

The model is evaluated using:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score (Coefficient of Determination)

---

## 💾 Output Files

### Trained Model

```text
Advertising_LR_Model.joblib
```

### Prediction Output

```text
Advertising_Output.csv
```

---

## 📁 Generated Files

After successful execution, the following files are generated:

- Advertising_Output.csv
- Advertising_LR_Model.joblib

---

## 📚 Concepts Covered

- Supervised Machine Learning
- Linear Regression
- Regression Analysis
- Data Preprocessing
- Correlation Analysis
- Train-Test Split
- Model Evaluation
- MSE
- RMSE
- R² Score
- Model Persistence using Joblib
- Sales Prediction

---

## 🚀 Future Improvements

- Polynomial Regression
- Ridge Regression
- Lasso Regression
- Elastic Net Regression
- Feature Engineering
- Hyperparameter Tuning
- Flask REST API
- Streamlit Web Application

---

## 👨‍💻 Author

**Yogiraj Khaladkar**

Engineering Student  | Machine Learning Developer

---

## ⭐ Repository

If you found this project useful, please consider giving it a ⭐ on GitHub.