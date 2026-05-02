#----------------------------------------------------------------- 
# Required Python Packages 
#----------------------------------------------------------------- 

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import joblib

#----------------------------------------------------------------- 
# File Paths 
#-----------------------------------------------------------------

INPUT_PATH = "PlayPredictor.csv" 
OUTPUT_PATH = "PlayPredictor_Output.csv" 
MODEL_PATH = "PlayPredictor_knn.joblib" 
ENCODER_PATH = "PlayPredictor_encoders.joblib" 

#-----------------------------------------------------------------
# Function Name : display_Info
# Description   : It display the formated title
# Parameters    : title(str)
# Return        : None
# Author        : Yogiraj Khaladkar 
# Date          : 25/04/2026
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
# Date           : 25/04/2026
#------------------------------------------------------------------
def dataset_statistics(df):
    """It shows basic information about the dataset"""

    print("\n First 5 rows of dataset")
    print(df.head())

    print("\n Shape of dataset")
    print(df.shape)

    print("\n Columns of dataset")
    print(df.columns.tolist())

    print("\n Missing values in dataset")
    print(df.isnull().sum())

#-----------------------------------------------------------------
# Function Name  : preprocess_dataset 
# Description    : Drops unnecessary columns and encodes categorical data
# Parameters     : df
# Return         : X, Y, encoders dict, transformed_df
# Author         : Yogiraj Khaladkar
# Date           : 25/04/2026
#-----------------------------------------------------------------
def preprocess_dataset(df):
    """Preprocess dataset: drop first column and apply Label Encoding"""
    
    # Drop first column
    df = df.drop(df.columns[0], axis=1)

    # Create encoders
    weather_encoder = LabelEncoder()
    temp_encoder = LabelEncoder()
    play_encoder = LabelEncoder()

    # Encode columns
    df['Whether'] = weather_encoder.fit_transform(df['Whether'])
    df['Temperature'] = temp_encoder.fit_transform(df['Temperature'])
    df['Play'] = play_encoder.fit_transform(df['Play'])

    encoders = {
        'weather': weather_encoder,
        'temperature': temp_encoder,
        'play': play_encoder
    }

    print("\nProcessed Dataset (First 5 records):")
    print(df.head())
    print("\nProcessed Dataset shape :", df.shape)

    features = ["Whether", "Temperature"]
    X = df[features]
    Y = df["Play"]

    return X, Y, encoders, df

#------------------------------------------------------------------
# Function Name : plot_data
# Description   : Visualizes the dataset features against the target.
# Parameters    :
#               : X (DataFrame): Feature dataset.
#               : Y (Series): Target dataset.
# Returns       : None
# Author        : Yogiraj Khaladkar
# Date          : 25/04/2026
# ------------------------------------------------------------------
def plot_data(X, Y):
    """Scatter plot visualization of the encoded data"""
    
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(X['Whether'], X['Temperature'], c=Y, cmap='coolwarm', edgecolors='k')
    
    plt.xlabel("Weather (Encoded)")
    plt.ylabel("Temperature (Encoded)")
    plt.title("Dataset Visualization (Weather vs Temperature)")
    
    # Create a legend
    cbar = plt.colorbar(scatter)
    cbar.set_label('Play (Encoded)')
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

#-----------------------------------------------------------------
# Function Name  : split_dataset 
# Description    : Split the dataset with train_percentage 
# Parameters     : X, Y, test_percentage
# Return         : Dataset after splitting 
# Author         : Yogiraj Khaladkar
# Date           : 25/04/2026
#-----------------------------------------------------------------
def split_dataset(X, Y, test_percentage, random_state=42): 
    """Split dataset into train/test""" 

    train_x, test_x, train_y, test_y = train_test_split( 
        X, Y, 
        test_size=test_percentage, random_state=random_state
    ) 

    return train_x, test_x, train_y, test_y 

# ------------------------------------------------------------------
# Function Name : train_model
# Description   : Trains a K-Nearest Neighbors Classifier.
# Parameters    :
#               : train_x (DataFrame): Training feature dataset.
#               : train_y (Series): Training target labels.
# Returns       : KNeighborsClassifier: Trained model.
# Author        : Yogiraj Khaladkar
# Date          : 25/04/2026
# ------------------------------------------------------------------
def train_model(train_x, train_y):
    """Train the model """

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(train_x, train_y)

    print("Model trained successfully")
    return model

#------------------------------------------------------------------
# Function name : save_csv 
# Description   : Save the final output csv with actual and predicted
# Author        : Yogiraj Khaladkar
# Date          : 25/04/2026
#------------------------------------------------------------------
def save_csv(test_x, test_y, predictions, play_encoder):
    """Save the final output csv"""
    result_df = test_x.copy()

    # Add columns (Decode back to original strings for readability)
    result_df['Actual_Encoded'] = test_y
    result_df['Predicted_Encoded'] = predictions
    result_df['Actual_Label'] = play_encoder.inverse_transform(test_y)
    result_df['Predicted_Label'] = play_encoder.inverse_transform(predictions)

    # Save to CSV
    result_df.to_csv(OUTPUT_PATH, index=False)
    
    print("\nOutput saved successfully to", OUTPUT_PATH)

#-----------------------------------------------------------------
# Function Name : save_artifacts
# Description   : Save the model and encoders using joblib
# Parameters    : model, encoders dict
# Return        : None
# Author        : Yogiraj Khaladkar
# Date          : 25/04/2026
#------------------------------------------------------------------
def save_artifacts(model, encoders):
    """ Save the model and encoders"""
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODER_PATH)

    print(f"Model Preserved successfully with {MODEL_PATH}")
    print(f"Encoders Preserved successfully with {ENCODER_PATH}")

#-----------------------------------------------------------------
# Function Name : load_artifacts
# Description   : Load the trained model and encoders
# Parameters    : None
# Return        : model, encoders dict
# Author        : Yogiraj Khaladkar
# Date          : 25/04/2026
#------------------------------------------------------------------
def load_artifacts():
    """Load the train model and encoders"""

    loaded_model = joblib.load(MODEL_PATH)
    loaded_encoders = joblib.load(ENCODER_PATH)

    print("Model and Encoders successfully loaded")

    return loaded_model, loaded_encoders

#------------------------------------------------------------------
# Function name : plot_heatmap 
# Description   : Display the confusion matrix using a heatmap.    
# Author        : Yogiraj Khaladkar
# Date          : 25/04/2026
#------------------------------------------------------------------
def plot_heatmap(cm, model):
    """Display the confusion matrix."""
    data = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    data.plot(cmap='Blues')
    plt.title("Confusion Matrix: Play Predictor")
    plt.show()

#-----------------------------------------------------------------
# Function Name  : main()
# Description    : Starting point of the application. This is main pipeline controller.
# Parameters     : None
# Return         : None
# Author         : Yogiraj Khaladkar
# Date           : 25/04/2026
#------------------------------------------------------------------
def main():
    """Main function from where execution starts"""
    
    # 1 Load CSV 
    display_Info("Step 1: Load dataset")
    dataset = pd.read_csv(INPUT_PATH) 
    print("Dataset loaded")

    # 2 Basic stats & Preprocessing
    display_Info("Step 2 : Data Analysis (EDA) & Preprocessing")
    dataset_statistics(dataset) 
    X, Y, encoders, processed_df = preprocess_dataset(dataset)

    # 3 Visualization
    display_Info("Step 3 : Data Visualization")
    plot_data(X, Y)
    
    # 4 Split Data
    display_Info("Step 4 : Split dataset")
    train_x, test_x, train_y, test_y = split_dataset(X, Y, test_percentage=0.30)
    
    print("X Shape :", X.shape)
    print("Y Shape :", Y.shape)
    print("X_train Shape :", train_x.shape)
    print("X_test Shape :", test_x.shape)
    print("Y_train Shape :", train_y.shape)
    print("Y_test Shape :", test_y.shape)

    # 5 Build + Train Pipeline 
    display_Info("Step 5 : Build and Train Model")
    trained_model = train_model(train_x, train_y) 
    print("Trained model : ", trained_model) 
    
    # 6 Predictions 
    display_Info("Step 6 : Model Testing")
    predictions = trained_model.predict(test_x) 

    decoded_predictions = encoders['play'].inverse_transform(predictions)
    
    print("Encoded Prediction :", predictions)
    print("Actual :", test_y.values)
    print("Decoded Prediction :", decoded_predictions)

    # 7 Performance Evaluation
    display_Info("Step 7 : Evaluate Model Performance")
    accuracy = accuracy_score(test_y, predictions)
    cm = confusion_matrix(test_y, predictions)
    
    print("Testing Accuracy of the model is : {:.2f}%".format(accuracy * 100))
    print("\nClassification Report:\n", classification_report(test_y, predictions)) 
    print("Confusion Matrix:\n", cm) 

    # 8 Plot confusion matrix
    display_Info("Step 8 : Plot Confusion Matrix")
    plot_heatmap(cm, trained_model)

    # 9 Save artifacts using joblib 
    display_Info("Step 9 : Save Model and Encoders")
    save_artifacts(trained_model, encoders) 

    # 10 Load model and test a sample 
    display_Info("Step 10 : Load Model and Test a Sample")
    loaded_model, loaded_encoders = load_artifacts() 
    
    sample = test_x.iloc[[0]] 
    pred_loaded = loaded_model.predict(sample) 
    pred_loaded_decoded = loaded_encoders['play'].inverse_transform(pred_loaded)
    print(f"Loaded model prediction for first test sample (Encoded): {pred_loaded[0]}") 
    print(f"Loaded model prediction for first test sample (Decoded): {pred_loaded_decoded[0]}") 

    # 11 Store the prediction results in a CSV file
    display_Info("Step 11 : Store the prediction results in a CSV file")
    save_csv(test_x, test_y, predictions, encoders['play'])

#------------------------------------------------------------------
# Application starter 
#------------------------------------------------------------------
if __name__ == "__main__":
    main()