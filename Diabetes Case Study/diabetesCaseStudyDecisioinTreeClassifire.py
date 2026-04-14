#-----------------------------------------------------------------
# Required Python Packages 
#-----------------------------------------------------------------

import pandas as pd
import numpy as np

from sklearn.metrics import (
accuracy_score,
confusion_matrix,
ConfusionMatrixDisplay,
classification_report
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
#-----------------------------------------------------------------
# File Paths
#-----------------------------------------------------------------

INPUT_FILE = "diabetes.csv"
OUTPUT_FILE = "final_diabetes_predictions.csv"
MODEL_PATH = "diabetes.joblib"

#-----------------------------------------------------------------
# Fucntion Name : display_Info
# Description   : It display the formated title
# Paramaters    : title(str)
# Return        : None
# Author        : Yogiraj Khaladkar 
# Date          : 10/04/2026
#------------------------------------------------------------------
def display_Info(title):
    """Display the message"""
    print("\n"+"="*70)
    print(title)
    print("="*70)

#-----------------------------------------------------------------
# Fucntion Name  : dataset_statistics
# Description    : It shows basic information about the dataset
# Paramaters     : Dataset(df)
#                  df -> pandas dataframe object
#                  message -> Heading text to display
# Return         : None
# Author         : Yogiraj Khaladkar
# Date           : 10/04/2026
#------------------------------------------------------------------
def dataset_statistics(df):
    """It shows basic information about the dataset"""

    print("\n First 5 rows of dataset")
    print(df.head)

    print("\n Shape of dataset")
    print(df.shape)

    print("\n Columns of dataset")
    print(df.columns.tolist())

    print("\n Statistics report")
    print(df.describe())

    print("\n Missing value in dataset")
    print(df.isnull().sum())

    sns.countplot(x='Outcome', data=df)
    plt.title("Distribution of Outcome")
    plt.show()

    df.hist(figsize=(12,10))
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,8))
    sns.boxplot(data=df)
    plt.xticks(rotation=45)
    plt.title("Boxplot for Outlier Detection")
    plt.show()

#-----------------------------------------------------------------
# Fucntion Name : clean_diabetes_data
# Description   : It does preprocessing
#                 It handle missing values
# Paramaters    : df -> Pandas dataframe
# Return        : df -> Clean Pandas dataframe
# Author        : Yogiraj Khaladkar
# Date          : 10/04/2026
#------------------------------------------------------------------
def clean_diabetes_data(df):
   
    # Handle SkinThickness columns

    if "SkinThickness" in df.columns:
        print("SkinThickness columns before filling missing value :")
        print(df["SkinThickness"].head(10))

        # replace the zero with nan
        df['SkinThickness'] = df['SkinThickness'].replace(0, np.nan)

        # coerce invalied value gets converted as NAN
        df["SkinThickness"] = pd.to_numeric(df["SkinThickness"],errors="coerce")

        SkinThickness_median=df["SkinThickness"].median()
        
        # Replace missing value
        df["SkinThickness"]=df["SkinThickness"].fillna(SkinThickness_median)

        print("SkinThickness columns after preprocessing")
        print(df["SkinThickness"].head(10))


    # Handle Insulin columns 
    if "Insulin" in df.columns :
        print("\n Insulin columns before preprocessing")
        print(df["Insulin"].head(10))

        df['Insulin'] = df['Insulin'].replace(0, np.nan)

        # coerce invalied value gets converted as NAN
        df["Insulin"] = pd.to_numeric(df["Insulin"],errors="coerce")

        fare_median=df["Insulin"].median()
    
        df["Insulin"]=df["Insulin"].fillna(fare_median)

        print("Insulin columns after preprocessing")
        print(df["Insulin"].head(10))

        median = df["Insulin"].median()
    
        Q1 = df["Insulin"].quantile(0.25)
        Q3 = df["Insulin"].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        df["Insulin"] = df["Insulin"].apply(lambda x: median if x < lower or x > upper else x)
    if "BMI" in df.columns:
        print("BMI before preprocessing:")
        print(df["BMI"].head(10))

        # Replace invalid zeros with NaN
        df["BMI"] = df["BMI"].replace(0, np.nan)

        # Convert to numeric (coerce errors)
        df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")

        # Median imputation
        bmi_median = df["BMI"].median()
        df["BMI"] = df["BMI"].fillna(bmi_median)

        print("BMI after preprocessing:")
        print(df["BMI"].head(10))

    if "BloodPressure" in df.columns:
        print("BloodPressure before preprocessing:")
        print(df["BloodPressure"].head(10))

        df["BloodPressure"] = df["BloodPressure"].replace(0, np.nan)
        df["BloodPressure"] = pd.to_numeric(df["BloodPressure"], errors="coerce")

        bp_median = df["BloodPressure"].median()
        df["BloodPressure"] = df["BloodPressure"].fillna(bp_median)

        print("BloodPressure after preprocessing:")
        print(df["BloodPressure"].head(10))

    if "Glucose" in df.columns:
        print("Glucose before preprocessing:")
        print(df["Glucose"].head(10))

        df["Glucose"] = df["Glucose"].replace(0, np.nan)
        df["Glucose"] = pd.to_numeric(df["Glucose"], errors="coerce")

        glucose_median = df["Glucose"].median()
        df["Glucose"] = df["Glucose"].fillna(glucose_median)

        print("Glucose after preprocessing:")
        print(df["Glucose"].head(10))

    return df

#-----------------------------------------------------------------
# Function Name  : split_dataset 
# Description    : Split the dataset with train_percentage 
# Parameters     : Dataset with related information 
# Return         : Dataset after splitting 
# Author         : Yogiraj Khaladkar
# Date           : 10/04/2026
#-----------------------------------------------------------------

def split_dataset(dataset, feature_headers,target_header, train_percentage ,random_state=42): 
    """Split dataset into train/test""" 

    train_x, test_x, train_y, test_y = train_test_split( 
        feature_headers,target_header, 
        train_size=train_percentage, random_state=random_state
    ) 

    return train_x, test_x, train_y, test_y 


# ------------------------------------------------------------------
# Function Name : train_diabetes_model
# Description   : Trains a Decision Tree Classifier using the training dataset.
# Parameters    :
#               : train_x (DataFrame): Training feature dataset.
#               : train_y (Series): Training target labels.
# Returns       : DecisionTreeClassifier: Trained Decision Tree model.
# Author        : Yogiraj Khaladkar
# Date          : 10/04/2026
# ------------------------------------------------------------------

def train_diabetes_model(train_x, train_y):
    """
    Train a Decision Tree Classifier on the diabetes dataset.

    Parameters:
        train_x (DataFrame): Training feature dataset.
        train_y (Series): Training target labels.

    Returns:
        DecisionTreeClassifier: Trained model.
    """
    model = DecisionTreeClassifier(
        criterion="gini",
        random_state=42,
        max_depth=2
    )

    model.fit(train_x, train_y)

    print("Model trained successfully")
    return model

#------------------------------------------------------------------
# Function name : plot_heatmap 
# Description   : Display the confusion matrix using a heatmap.    
# Author        : Yogiraj Khaladkar
# Date          : 10/04/2026
#------------------------------------------------------------------
def plot_heatmap(cm):
        """Display the confusion matrix using a heatmap."""
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

#------------------------------------------------------------------
# Function name : save_csv 
# Description   : Save the final output csv"
# Author        : Yogiraj Khaladkar
# Date          : 10/04/2026
#------------------------------------------------------------------
def save_csv(test_x,test_y,predictions):
        """Save the final output csv"""
        result_df = test_x.copy()

        # Add columns
        result_df['Actual'] = test_y
        result_df['Predicted'] = predictions

        # Save to CSV
        result_df.to_csv(OUTPUT_FILE, index=False)

#-----------------------------------------------------------------
# Fucntion Name : save_model
# Description   : Save the model  
# Paramaters    : model
#                 path
# Return        : None
# Author        : Yogiraj Khaladkar
# Date          : 10/04/2026
#------------------------------------------------------------------

def save_model(model,path= MODEL_PATH):
    """ Save the model"""
    joblib.dump(model,path)

    print("Model Preserve successfully with ",path)

#-----------------------------------------------------------------
# Fucntion Name : load_model
# Description   : Load the train model
# Paramaters    : path 
# Return        : model
# Author        : Yogiraj Khaladkar
# Date          : 10/04/2026
#------------------------------------------------------------------

def load_model(path = MODEL_PATH):
    """Load the train model"""

    loaded_model=joblib.load(path)

    print("Model successfully loaded")

    return loaded_model

#-----------------------------------------------------------------
# Fucntion Name  : main()
# Description    : Starting point of the application.This is main pipeline controller.
# Paramaters     : None
# Return         : None
# Author         : Yogiraj Khaladkar
# Date           : 10/04/2026
#------------------------------------------------------------------
def main():
    """Main function from where exexution starts"""
    
    # 1 Load CSV 
    display_Info("Step 1: Load dataset")
    print("Dataset loaded")
    dataset = pd.read_csv(INPUT_FILE) 

    # 2 Basic stats 
    display_Info("Step 2 : Print Basic data of dataset")
    dataset_statistics(dataset) 

    # 3 Clean dataaset
    display_Info("Step 3 : Clean dataset")
    dataset = clean_diabetes_data(dataset)
   
    feature_headers = dataset.drop("Outcome",axis=1)
    target_header =  dataset['Outcome']

    # 4 Split 
    display_Info("Step 4 : Split dataset")
    train_x, test_x, train_y, test_y = split_dataset(dataset,feature_headers,target_header, 0.7)
    
    print("Train_x Shape :: ", train_x.shape) 
    print("Train_y Shape :: ", train_y.shape) 
    print("Test_x Shape :: ", test_x.shape) 
    print("Test_y Shape :: ", test_y.shape) 

    # 5 Build + Train Pipeline 
    display_Info("Step 5 : Build + Train ")
    trained_model = train_diabetes_model(train_x, train_y) 
    print("Trained mode : ", trained_model) 
    
    # 6 Predictions 
    predictions = trained_model.predict(test_x) 

    # 7 Metrics 
    cm= confusion_matrix(test_y, predictions)
    display_Info("Step 7 : Display the Metrics")
    print("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x))) 
    print("Test Accuracy  :: ", accuracy_score(test_y, predictions)) 
    print("Classification Report:\n", classification_report(test_y, predictions)) 
    print("Confusion Matrix:\n",cm ) 

    # 8 Plot confusion matrix
    display_Info("Step 8 : Plot confusion matrix")
    plot_heatmap(cm)

    # 9 Save model  using joblib 
    display_Info("Step 9 : Save model using joblib")
    save_model(trained_model, MODEL_PATH) 

    # 10 Load model and test a sample 
    display_Info("Step 10 : Load model and test a sample")
    loaded = load_model(MODEL_PATH) 
    sample = test_x.iloc[[0]] 
    pred_loaded = loaded.predict(sample) 
    print(f"Loaded model prediction for first test sample: {pred_loaded[0]}") 

    # 11. Store the prediction results in a CSV file
    display_Info("Step 11 : Store the prediction results in a CSV file")
    save_csv(test_x, test_y, predictions)

#------------------------------------------------------------------
# Application starter 
#------------------------------------------------------------------
if __name__ =="__main__":
    main()
   