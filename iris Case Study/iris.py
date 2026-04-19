#----------------------------------------------------------------- 
# Required Python Packages 
#----------------------------------------------------------------- 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

import joblib

#----------------------------------------------------------------- 
# File Paths 
#-----------------------------------------------------------------

INPUT_PATH = "california_housing.csv" 
OUTPUT_PATH = "california_housing_Output.csv" 
MODEL_PATH = "california_housing.joblib" 

#-----------------------------------------------------------------
# Function Name : display_Info
# Description   : It display the formated title
# Parameters    : title(str)
# Return        : None
# Author        : Yogiraj Khaladkar 
# Date          : 23/07/2026
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
# Date           : 23/07/2026
#------------------------------------------------------------------
def dataset_statistics(df):
    """It shows basic information about the dataset"""

    print("\n First 5 rows of dataset")
    print(df.head())

    print("\n Shape of dataset")
    print(df.shape)

    print("\n Columns of dataset")
    print(df.columns.tolist())

    print("\n Statistics report")
    print(df.describe())

    print("\n Missing value in dataset")
    print(df.isnull().sum())

    plt.figure(figsize=(7,5))
    
    # Using a histogram to visualize the continuous target distribution for regression
    sns.histplot(df["target"], kde=True, color="blue")
    
    plt.title("California Housing : Target Distribution")
    plt.xlabel("Target Value")
    plt.ylabel("Frequency")

    plt.grid()
    plt.show()

#-----------------------------------------------------------------
# Function Name  : split_dataset 
# Description    : Split the dataset with train_percentage 
# Parameters     : Dataset with related information 
# Return         : Dataset after splitting 
# Author         : Yogiraj Khaladkar
# Date           : 23/07/2026
#-----------------------------------------------------------------
def split_dataset(dataset, feature_headers, target_header, train_percentage, random_state=42): 
    """Split dataset into train/test""" 

    train_x, test_x, train_y, test_y = train_test_split( 
        feature_headers, target_header, 
        train_size=train_percentage, random_state=random_state
    ) 

    return train_x, test_x, train_y, test_y 

# ------------------------------------------------------------------
# Function Name : train_model
# Description   : Trains a Bagging Regressor using the training dataset.
# Parameters    :
#               : train_x (DataFrame): Training feature dataset.
#               : train_y (Series): Training target labels.
# Returns       : BaggingRegressor: Trained Ensemble model.
# Author        : Yogiraj Khaladkar
# Date          : 23/07/2026
# ------------------------------------------------------------------
def train_model(train_x, train_y):
    base_model = DecisionTreeRegressor(
        max_depth=15,
        min_samples_split=5,
        random_state=42
    )

    model = BaggingRegressor(
        estimator=base_model,
        n_estimators=100,
        random_state=42
    )

    model.fit(train_x, train_y)

    print("Model trained successfully")
    return model

#------------------------------------------------------------------
# Function name : save_csv 
# Description   : Save the final output csv
# Author        : Yogiraj Khaladkar
# Date          : 23/07/2026
#------------------------------------------------------------------
def save_csv(test_x, test_y, predictions):
    """Save the final output csv"""
    result_df = test_x.copy()

    # Add columns
    result_df['Actual'] = test_y
    result_df['Predicted'] = predictions

    # Save to CSV
    result_df.to_csv(OUTPUT_PATH, index=False)
    print("CSV saved successfully to", OUTPUT_PATH)

#-----------------------------------------------------------------
# Function Name : save_model
# Description   : Save the model  
# Parameters    : model, path
# Return        : None
# Author        : Yogiraj Khaladkar
# Date          : 23/07/2026
#------------------------------------------------------------------
def save_model(model, path=MODEL_PATH):
    """ Save the model"""
    joblib.dump(model, path)

    print("Model Preserved successfully with", path)

#-----------------------------------------------------------------
# Function Name : load_model
# Description   : Load the train model
# Parameters    : path 
# Return        : model
# Author        : Yogiraj Khaladkar
# Date          : 23/07/2026
#------------------------------------------------------------------
def load_model(path=MODEL_PATH):
    """Load the train model"""

    loaded_model = joblib.load(path)

    print("Model successfully loaded")

    return loaded_model

#------------------------------------------------------------------
# Function name : plot_evaluation 
# Description   : Display the actual vs predicted values using a scatter plot.    
# Author        : Yogiraj Khaladkar
# Date          : 23/07/2026
#------------------------------------------------------------------
def plot_evaluation(test_y, predictions):
    """Display the actual vs predicted values using a scatter plot."""
    plt.figure(figsize=(7,5))
    plt.scatter(test_y, predictions, alpha=0.5, color='blue')
    
    # Perfect prediction line
    plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'r--', lw=2)
    
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted")
    plt.grid()
    plt.show()

#-----------------------------------------------------------------
# Function Name  : main()
# Description    : Starting point of the application. This is main pipeline controller.
# Parameters     : None
# Return         : None
# Author         : Yogiraj Khaladkar
# Date           : 23/07/2026
#------------------------------------------------------------------
def main():
    """Main function from where execution starts"""
    
    # 1 Load CSV 
    display_Info("Step 1: Load dataset")
    dataset = pd.read_csv(INPUT_PATH) 
    print("Dataset loaded")

    # 2 Basic stats 
    display_Info("Step 2 : Print Basic data of dataset")
    dataset_statistics(dataset) 

    # 3 Select feature & target
    display_Info("Step 3 : Select the feature & target")
    feature_headers = dataset.drop("target", axis=1)
    target_header = dataset['target']

    # 4 Split 
    display_Info("Step 4 : Split dataset")
    # Using 0.80 to match the original script's test_size=0.20
    train_x, test_x, train_y, test_y = split_dataset(dataset, feature_headers, target_header, 0.80)
    
    print("Train_x Shape :: ", train_x.shape) 
    print("Train_y Shape :: ", train_y.shape) 
    print("Test_x Shape :: ", test_x.shape) 
    print("Test_y Shape :: ", test_y.shape) 

    # 5 Build + Train Pipeline 
    display_Info("Step 5 : Build + Train")
    trained_model = train_model(train_x, train_y) 
    print("Trained model : ", trained_model) 
    
    # 6 Predictions 
    predictions = trained_model.predict(test_x) 

    # 7 Metrics 
    display_Info("Step 7 : Display the Metrics")
    train_predictions = trained_model.predict(train_x)
    print("Train MSE :: ", mean_squared_error(train_y, train_predictions))
    print("Train R2  :: ", r2_score(train_y, train_predictions))
    
    print("\nTest MSE  :: ", mean_squared_error(test_y, predictions)) 
    print("Test R2   :: ", r2_score(test_y, predictions)) 

    # 8 Plot evaluation metrics
    display_Info("Step 8 : Plot Actual vs Predicted values")
    plot_evaluation(test_y, predictions)

    # 9 Save model using joblib 
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
if __name__ == "__main__":
    main()