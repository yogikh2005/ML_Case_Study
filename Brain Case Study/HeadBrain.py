#----------------------------------------------------------------- 
# Required Python Packages 
#----------------------------------------------------------------- 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)

import joblib

#----------------------------------------------------------------- 
# File Paths 
#-----------------------------------------------------------------

INPUT_PATH = "HeadBrain.csv" 
OUTPUT_PATH = "HeadBrain_Output.csv" 
MODEL_PATH = "HeadBrain_DT_Model.joblib" 

#-----------------------------------------------------------------
# Function Name : display_Info
# Description   : It display the formated title
# Parameters    : title(str)
# Return        : None
# Author        : Yogiraj Khaladkar 
# Date          : 16/05/2026
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
# Date           : 16/05/2026
#------------------------------------------------------------------
def dataset_statistics(df):
    """It shows basic information and correlations of the dataset"""

    print("\n Few records from the dataset :")
    print(df.head())

    print("\n Shape of dataset :")
    print(df.shape)

    print("\n Columns of dataset :")
    print(df.columns.tolist())

    print("\n Missing values counts :")
    print(df.isnull().sum())

    print("\n Statistical report :")
    print(df.describe())

    print("\n Correlation matrix : ")
    # Selecting only numeric types avoids warnings during correlation calculation
    print(df.select_dtypes(include=[np.number]).corr())

#-----------------------------------------------------------------
# Function Name  : split_dataset 
# Description    : Split the dataset into features and target, then train/test
# Parameters     : df, test_percentage
# Return         : train_x, test_x, train_y, test_y, feature_columns
# Author         : Yogiraj Khaladkar
# Date           : 16/05/2026
#-----------------------------------------------------------------
def split_dataset(df, test_percentage, random_state=42): 
    """Split dataset into independent and dependent variables, then train/test""" 

    feature_cols = ['Age Range', 'Head Size(cm^3)', 'Brain Weight(grams)']
    X = df[feature_cols]
    Y = df['Gender']

    train_x, test_x, train_y, test_y = train_test_split( 
        X, Y, 
        test_size=test_percentage, random_state=random_state
    ) 

    return train_x, test_x, train_y, test_y, feature_cols

# ------------------------------------------------------------------
# Function Name : train_model
# Description   : Trains a Decision Tree Classifier model.
# Parameters    : train_x, train_y
# Returns       : Trained model
# Author        : Yogiraj Khaladkar
# Date          : 16/05/2026
# ------------------------------------------------------------------
def train_model(train_x, train_y):
    """Train the model"""
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42
    )
    model.fit(train_x, train_y)

    print("Decision Tree Model trained successfully.")
    return model

#------------------------------------------------------------------
# Function name : plot_evaluation 
# Description   : Displays the confusion matrix    
# Author        : Yogiraj Khaladkar
# Date          : 16/05/2026
#------------------------------------------------------------------
def plot_evaluation(cm, model):
    """Plot the Confusion Matrix"""
    data = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    data.plot(cmap='Blues')
    plt.title("Confusion Matrix of Brain Case Study")
    plt.show()

#------------------------------------------------------------------
# Function name : save_csv 
# Description   : Save the final output csv with actual and predicted
# Author        : Yogiraj Khaladkar
# Date          : 16/05/2026
#------------------------------------------------------------------
def save_csv(test_x, test_y, predictions):
    """Save the final output csv"""
    result_df = test_x.copy()
    result_df['Actual Gender'] = test_y.values
    result_df['Predicted Gender'] = predictions

    result_df.to_csv(OUTPUT_PATH, index=False)
    
    print("\nDataset with Actual and Predicted values (First 5 records):")
    print(result_df.head())
    print("\nDataset with Actual and Predicted values (Last 5 records):")
    print(result_df.tail())
    print("\nPrediction results saved successfully to", OUTPUT_PATH)

#-----------------------------------------------------------------
# Function Name : save_model
# Description   : Save the model using joblib
# Parameters    : model
# Return        : None
# Author        : Yogiraj Khaladkar
# Date          : 16/05/2026
#------------------------------------------------------------------
def save_model(model, path=MODEL_PATH):
    """ Save the trained model"""
    joblib.dump(model, path)
    print(f"Model Preserved successfully with {path}")

#-----------------------------------------------------------------
# Function Name : load_model
# Description   : Load the train model
# Parameters    : path 
# Return        : model
# Author        : Yogiraj Khaladkar
# Date          : 16/05/2026
#------------------------------------------------------------------
def load_model(path=MODEL_PATH):
    """Load the trained model"""
    loaded_model = joblib.load(path)
    print("Model successfully loaded")
    return loaded_model

#-----------------------------------------------------------------
# Function Name  : main()
# Description    : Starting point of the application. Pipeline controller.
# Parameters     : None
# Return         : None
# Author         : Yogiraj Khaladkar
# Date           : 16/05/2026
#------------------------------------------------------------------
def main():
    """Main function from where execution starts"""
    
    # 1 Load CSV 
    display_Info("Step 1: Load Dataset")
    try:
        dataset = pd.read_csv(INPUT_PATH) 
        print("Dataset loaded successfully")
    except FileNotFoundError:
        print(f"Error: Ensure '{INPUT_PATH}' is in the current directory.")
        return

    # 2 & 3 & 4 Basic stats, missing values & correlation
    display_Info("Step 2 & 3 & 4 : Check Missing Values and Statistical Summary")
    dataset_statistics(dataset) 

    # 5 & 6 Split Data
    display_Info("Step 5 & 6 : Split Dataset for Training and Testing")
    train_x, test_x, train_y, test_y, feature_cols = split_dataset(dataset, test_percentage=0.20)
    
    print("X_train shape : ", train_x.shape)    
    print("X_test shape : ", test_x.shape)    
    print("Y_train shape : ", train_y.shape)    
    print("Y_test shape : ", test_y.shape)

    # 7 Build + Train Pipeline 
    display_Info("Step 7 : Create and Train Model")
    trained_model = train_model(train_x, train_y) 
    
    # 8 Predictions 
    display_Info("Step 8 : Test the Model")
    predictions = trained_model.predict(test_x) 
    print("Model predictions generated successfully on Test Data.")

    # 9 Metrics evaluation
    display_Info("Step 9 : Evaluate the Model")
    accuracy = accuracy_score(test_y, predictions)
    cm = confusion_matrix(test_y, predictions)

    print("Model Accuracy   : {:.2f}%".format(accuracy * 100))
    print("Confusion matrix :\n", cm)

    # 10 Plot evaluation
    display_Info("Step 10 : Plot Confusion Matrix")
    plot_evaluation(cm, trained_model)

    # 11 Save model using joblib 
    display_Info("Step 11 : Save Model")
    save_model(trained_model) 

    # 12 Load model and test a sample 
    display_Info("Step 12 : Load Model and Test a Sample")
    loaded_model = load_model() 
    sample = test_x.iloc[[0]] 
    pred_loaded = loaded_model.predict(sample) 
    print(f"Loaded model prediction for first test sample: {pred_loaded[0]}") 

    # 13 Store the prediction results in a CSV file
    display_Info("Step 13 : Actual vs Predicted Values -> Save to CSV")
    save_csv(test_x, test_y, predictions)
    
    display_Info("Pipeline Execution Completed")

#------------------------------------------------------------------
# Application starter 
#------------------------------------------------------------------
if __name__ == "__main__":
    main()