#----------------------------------------------------------------- 
# Required Python Packages 
#----------------------------------------------------------------- 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler

import joblib

#----------------------------------------------------------------- 
# File Paths 
#-----------------------------------------------------------------

INPUT_PATH = "bank-full.csv" 
OUTPUT_PATH = "bank-full_Output.csv" 
MODEL_PATH = "bank_rf_model.joblib" 
SCALER_PATH = "bank_scaler.joblib" 

#-----------------------------------------------------------------
# Function Name : display_Info
# Description   : It display the formated title
# Parameters    : title(str)
# Return        : None
# Author        : Yogiraj Khaladkar 
# Date          : 06/05/2026
#------------------------------------------------------------------
def display_Info(title):
    """Display the message"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

#-----------------------------------------------------------------
# Function Name  : dataset_statistics
# Description    : It shows basic information about the dataset
# Parameters     : df -> pandas dataframe object
# Return         : None
# Author         : Yogiraj Khaladkar
# Date           : 06/05/2026
#------------------------------------------------------------------
def dataset_statistics(df):
    """It shows basic information about the dataset and visualizations"""

    print("\n First 5 rows of dataset")
    print(df.head())

    print("\n Shape of dataset")
    print(df.shape)

    print("\n Columns of dataset")
    print(df.columns.tolist())

    print("\n Statistical report")
    print(df.describe())

    print("\n Missing value in dataset")
    print(df.isnull().sum())

    # Visualizations
    plt.figure(figsize=(6, 4))
    sns.countplot(x='y', data=df)
    plt.title("Distribution of Outcome")
    plt.show()

    df.hist(figsize=(12, 10))
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df.select_dtypes(include=[np.number]))
    plt.xticks(rotation=45)
    plt.title("Boxplot for Outlier Detection")
    plt.show()

#-----------------------------------------------------------------
# Function Name  : preprocess_dataset 
# Description    : Preprocesses the data, handles outliers and missing values
# Parameters     : df
# Return         : Processed DataFrame
# Author         : Yogiraj Khaladkar
# Date           : 06/05/2026
#-----------------------------------------------------------------
def preprocess_dataset(df):
    """Data cleaning, encoding, and outlier handling"""
    
    # Drop unknown or irrelevant columns
    df.drop(['contact', 'poutcome'], axis=1, inplace=True, errors='ignore')

    print("\nAfter removing columns dataset shape:", df.shape)

    # Label Encoding
    mapping_dict = {'yes': 1, 'no': 0}
    df['default'] = df['default'].map(mapping_dict)
    df['housing'] = df['housing'].map(mapping_dict)
    df['loan'] = df['loan'].map(mapping_dict)
    df['y'] = df['y'].map(mapping_dict)

    # One-Hot Encoding
    df = pd.get_dummies(df, columns=['job', 'marital', 'education', 'month'])

    # Handle balance column outliers and missing values
    if "balance" in df.columns:
        print("\nBalance column before preprocessing:")
        print(df["balance"].head(10))

        df['balance'] = df['balance'].replace(0, np.nan)
        df["balance"] = pd.to_numeric(df["balance"], errors="coerce")

        median_balance = df["balance"].median()
        df["balance"] = df["balance"].fillna(median_balance)

        print("\nBalance column after filling missing values:")
        print(df["balance"].head(10))

        # Outlier Treatment (IQR Method)
        Q1 = df["balance"].quantile(0.25)
        Q3 = df["balance"].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df["balance"] = df["balance"].apply(
            lambda x: median_balance if x < lower_bound or x > upper_bound else x
        )
    
    print("\nDataset after Preprocessing (First 5 records):")
    print(df.head())

    return df

#-----------------------------------------------------------------
# Function Name  : split_dataset 
# Description    : Split the dataset into features and target, then train/test
# Parameters     : df, test_percentage
# Return         : train_x, test_x, train_y, test_y
# Author         : Yogiraj Khaladkar
# Date           : 06/05/2026
#-----------------------------------------------------------------
def split_dataset(df, test_percentage, random_state=42): 
    """Split dataset into train/test""" 

    X = df.drop("y", axis=1)
    Y = df["y"]

    train_x, test_x, train_y, test_y = train_test_split( 
        X, Y, 
        test_size=test_percentage, random_state=random_state
    ) 

    return train_x, test_x, train_y, test_y 

# ------------------------------------------------------------------
# Function Name : scale_data
# Description   : Scales the features using StandardScaler.
# Parameters    : train_x, test_x
# Returns       : train_x_scaled, test_x_scaled, scaler
# Author        : Yogiraj Khaladkar
# Date          : 06/05/2026
# ------------------------------------------------------------------
def scale_data(train_x, test_x):
    """Scale datasets using StandardScaler"""
    scaler = StandardScaler()

    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)

    print("Data scaled successfully.")
    return train_x_scaled, test_x_scaled, scaler

# ------------------------------------------------------------------
# Function Name : train_model
# Description   : Trains a Random Forest Classifier.
# Parameters    : train_x, train_y
# Returns       : Trained model
# Author        : Yogiraj Khaladkar
# Date          : 06/05/2026
# ------------------------------------------------------------------
def train_model(train_x, train_y):
    """Train the model"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(train_x, train_y)

    print("Random Forest Model trained successfully.")
    return model

#------------------------------------------------------------------
# Function name : plot_metrics 
# Description   : Displays the confusion matrix and ROC curve.    
# Author        : Yogiraj Khaladkar
# Date          : 06/05/2026
#------------------------------------------------------------------
def plot_metrics(Y_test, Y_pred, Y_prob):
    """Plot Confusion Matrix and ROC Curve"""
    
    cm = confusion_matrix(Y_test, Y_pred)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(Y_test, Y_prob)
    auc_score = roc_auc_score(Y_test, Y_prob)
    
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}", color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

#------------------------------------------------------------------
# Function name : save_csv 
# Description   : Save the final output csv with actual and predicted
# Author        : Yogiraj Khaladkar
# Date          : 06/05/2026
#------------------------------------------------------------------
def save_csv(test_x, test_y, predictions):
    """Save the final output csv"""
    # Assuming test_x is an array after scaling, convert back to df for saving
    result_df = pd.DataFrame(test_x)
    result_df['Actual_y'] = test_y.values
    result_df['Predicted_y'] = predictions

    result_df.to_csv(OUTPUT_PATH, index=False)
    print("\nPrediction results saved successfully to", OUTPUT_PATH)

#-----------------------------------------------------------------
# Function Name : save_artifacts
# Description   : Save the model and scaler using joblib
# Parameters    : model, scaler
# Return        : None
# Author        : Yogiraj Khaladkar
# Date          : 06/05/2026
#------------------------------------------------------------------
def save_artifacts(model, scaler):
    """ Save the model and scaler"""
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("Model Preserved successfully with", MODEL_PATH)
    print("Scaler Preserved successfully with", SCALER_PATH)

#-----------------------------------------------------------------
# Function Name  : main()
# Description    : Starting point of the application. Pipeline controller.
# Parameters     : None
# Return         : None
# Author         : Yogiraj Khaladkar
# Date           : 06/05/2026
#------------------------------------------------------------------
def main():
    """Main function from where execution starts"""
    
    # 1 Load CSV 
    display_Info("Step 1: Load Dataset")
    try:
        dataset = pd.read_csv(INPUT_PATH, sep=";") 
        print("Dataset loaded successfully")
    except FileNotFoundError:
        print(f"Error: Ensure '{INPUT_PATH}' is in the current directory.")
        return

    # 2 Basic stats & EDA
    display_Info("Step 2 : Dataset Statistics & Visualization")
    dataset_statistics(dataset) 

    # 3 Preprocessing
    display_Info("Step 3 : Preprocess Data")
    processed_df = preprocess_dataset(dataset)

    # 4 Split Data
    display_Info("Step 4 : Split Dataset")
    train_x, test_x, train_y, test_y = split_dataset(processed_df, test_percentage=0.20)
    
    print("Shape of X_train :", train_x.shape)
    print("Shape of X_test :", test_x.shape)
    print("Shape of Y_train :", train_y.shape)
    print("Shape of Y_test :", test_y.shape)

    # 5 Scale Data
    display_Info("Step 5 : Scale Features")
    train_x_scaled, test_x_scaled, scaler = scale_data(train_x, test_x)

    # 6 Build + Train Pipeline 
    display_Info("Step 6 : Train Model")
    trained_model = train_model(train_x_scaled, train_y) 
    
    # 7 Predictions 
    display_Info("Step 7 : Model Evaluation")
    predictions = trained_model.predict(test_x_scaled) 
    probabilities = trained_model.predict_proba(test_x_scaled)[:, 1]

    # Metrics computation
    accuracy = accuracy_score(test_y, predictions)
    auc_score = roc_auc_score(test_y, probabilities)
    
    print("Accuracy is : {:.2f}%".format(accuracy * 100))
    print("AUC Score is : {:.4f}".format(auc_score))
    print("\nClassification Report:\n", classification_report(test_y, predictions)) 

    # 8 Plot confusion matrix & ROC
    display_Info("Step 8 : Visualize Results")
    plot_metrics(test_y, predictions, probabilities)

    # 9 Save artifacts using joblib 
    display_Info("Step 9 : Save Model and Scaler")
    save_artifacts(trained_model, scaler) 

    # 10 Store the prediction results in a CSV file
    display_Info("Step 10 : Store the prediction results in a CSV file")
    save_csv(test_x_scaled, test_y, predictions)
    
    display_Info("Pipeline Execution Completed")

#------------------------------------------------------------------
# Application starter 
#------------------------------------------------------------------
if __name__ == "__main__":
    main()