#----------------------------------------------------------------- 
# Required Python Packages 
#----------------------------------------------------------------- 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import joblib

#----------------------------------------------------------------- 
# File Paths 
#-----------------------------------------------------------------

INPUT_PATH = "Advertising.csv" 
OUTPUT_PATH = "Advertising_Output.csv" 
MODEL_PATH = "Advertising_LR_Model.joblib" 

#-----------------------------------------------------------------
# Function Name : display_Info
# Description   : It display the formated title
# Parameters    : title(str)
# Return        : None
# Author        : Yogiraj Khaladkar 
# Date          : 11/05/2026
#------------------------------------------------------------------
def display_Info(title):
    """Display the message"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

#-----------------------------------------------------------------
# Function Name  : preprocess_dataset 
# Description    : Drops unnecessary columns before analysis
# Parameters     : df -> pandas dataframe object
# Return         : Cleaned DataFrame
# Author         : Yogiraj Khaladkar
# Date           : 11/05/2026
#-----------------------------------------------------------------
def preprocess_dataset(df):
    """Remove unwanted columns"""
    print("Shape of dataset before removal : ", df.shape)

    if "Unnamed: 0" in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    print("Shape of dataset after removal : ", df.shape)
    print("\nCleaned dataset (First 5 records):")
    print(df.head())
    
    return df

#-----------------------------------------------------------------
# Function Name  : dataset_statistics
# Description    : It shows basic information about the dataset
# Parameters     : df -> pandas dataframe object
# Return         : None
# Author         : Yogiraj Khaladkar
# Date           : 11/05/2026
#------------------------------------------------------------------
def dataset_statistics(df):
    """It shows basic information and correlations of the dataset"""

    print("\n Missing values counts :")
    print(df.isnull().sum())

    print("\n Statistical report :")
    print(df.describe())

    print("\n Correlation matrix : ")
    # Selecting only numeric types avoids warnings in newer Pandas versions
    print(df.select_dtypes(include=[np.number]).corr())

#-----------------------------------------------------------------
# Function Name  : split_dataset 
# Description    : Split the dataset into features and target, then train/test
# Parameters     : df, test_percentage
# Return         : train_x, test_x, train_y, test_y, feature_columns
# Author         : Yogiraj Khaladkar
# Date           : 11/05/2026
#-----------------------------------------------------------------
def split_dataset(df, test_percentage, random_state=42): 
    """Split dataset into independent and dependent variables, then train/test""" 

    feature_cols = ["TV", "radio", "newspaper"]
    X = df[feature_cols]
    Y = df['sales']

    train_x, test_x, train_y, test_y = train_test_split( 
        X, Y, 
        test_size=test_percentage, random_state=random_state
    ) 

    return train_x, test_x, train_y, test_y, feature_cols

# ------------------------------------------------------------------
# Function Name : train_model
# Description   : Trains a Linear Regression model.
# Parameters    : train_x, train_y
# Returns       : Trained model
# Author        : Yogiraj Khaladkar
# Date          : 11/05/2026
# ------------------------------------------------------------------
def train_model(train_x, train_y):
    """Train the model"""
    model = LinearRegression()
    model.fit(train_x, train_y)

    print("Linear Regression Model trained successfully.")
    return model

#------------------------------------------------------------------
# Function name : print_coefficients 
# Description   : Displays the model's coefficients and intercept
# Author        : Yogiraj Khaladkar
# Date          : 11/05/2026
#------------------------------------------------------------------
def print_coefficients(model, feature_cols):
    """Print the calculated model coefficients and intercept"""
    print("Model Coefficients:")
    for column, value in zip(feature_cols, model.coef_):
        print(f"  {column} : {value}")

    print("\nModel Intercept : ", model.intercept_)

#------------------------------------------------------------------
# Function name : plot_evaluation 
# Description   : Displays Actual vs Predicted scatter plot.    
# Author        : Yogiraj Khaladkar
# Date          : 11/05/2026
#------------------------------------------------------------------
def plot_evaluation(Y_test, Y_pred):
    """Plot Actual vs Predicted values"""
    plt.figure(figsize=(8, 5))
    plt.scatter(Y_test, Y_pred, alpha=0.7, color='blue')
    
    # Add a reference line for perfect predictions
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2)
    
    plt.xlabel("Actual sales")
    plt.ylabel("Predicted sales")
    plt.title("Actual sales vs Predicted sales")
    plt.grid(True)
    plt.show()

#------------------------------------------------------------------
# Function name : save_csv 
# Description   : Save the final output csv with actual and predicted
# Author        : Yogiraj Khaladkar
# Date          : 11/05/2026
#------------------------------------------------------------------
def save_csv(test_x, test_y, predictions):
    """Save the final output csv"""
    result_df = test_x.copy()
    result_df['Actual_Sales'] = test_y.values
    result_df['Predicted_Sales'] = predictions

    result_df.to_csv(OUTPUT_PATH, index=False)
    
    print("\nDataset with Actual and Predicted values (First 5 records):")
    print(result_df.head())
    print("\nPrediction results saved successfully to", OUTPUT_PATH)

#-----------------------------------------------------------------
# Function Name : save_model
# Description   : Save the model using joblib
# Parameters    : model
# Return        : None
# Author        : Yogiraj Khaladkar
# Date          : 11/05/2026
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
# Date          : 11/05/2026
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
# Date           : 11/05/2026
#------------------------------------------------------------------
def main():
    """Main function from where execution starts"""
    
    # 1 Load CSV 
    display_Info("Step 1: Load Dataset")
    try:
        dataset = pd.read_csv(INPUT_PATH) 
        print("Dataset loaded successfully")
        print("Few records from the dataset : ")
        print(dataset.head())
    except FileNotFoundError:
        print(f"Error: Ensure '{INPUT_PATH}' is in the current directory.")
        return

    # 2 Preprocess Data (Remove Unwanted Columns)
    display_Info("Step 2 : Remove Unwanted columns")
    processed_df = preprocess_dataset(dataset)

    # 3 & 4 Basic stats, missing values & correlation
    display_Info("Step 3 & 4 : Check Missing Values and Statistical Summary")
    dataset_statistics(processed_df) 

    # 5 & 6 Split Data
    display_Info("Step 5 & 6 : Split Dataset for Training and Testing")
    train_x, test_x, train_y, test_y, feature_cols = split_dataset(processed_df, test_percentage=0.20)
    
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
    MSE = mean_squared_error(test_y, predictions)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(test_y, predictions)

    print("Mean Squared Error      : ", MSE)
    print("Root Mean Squared Error : ", RMSE)
    print("R Square Score          : ", R2)

    # 10 Print Coefficients
    display_Info("Step 10 : Calculate Model Coefficients")
    print_coefficients(trained_model, feature_cols)

    # 11 Plot evaluation
    display_Info("Step 11 : Plot Actual vs Predicted values")
    plot_evaluation(test_y, predictions)

    # 12 Save model using joblib 
    display_Info("Step 12 : Save Model")
    save_model(trained_model) 

    # 13 Load model and test a sample 
    display_Info("Step 13 : Load Model and Test a Sample")
    loaded_model = load_model() 
    sample = test_x.iloc[[0]] 
    pred_loaded = loaded_model.predict(sample) 
    print(f"Loaded model prediction for first test sample: {pred_loaded[0]:.2f}") 

    # 14 Store the prediction results in a CSV file
    display_Info("Step 14 : Store the prediction results in a CSV file")
    save_csv(test_x, test_y, predictions)
    
    display_Info("Pipeline Execution Completed")

#------------------------------------------------------------------
# Application starter 
#------------------------------------------------------------------
if __name__ == "__main__":
    main()