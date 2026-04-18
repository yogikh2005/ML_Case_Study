#-----------------------------------------------------------------
# Required Python Packages 
#-----------------------------------------------------------------

import pandas  as pd 
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import joblib

#-----------------------------------------------------------------
# File Paths
#-----------------------------------------------------------------

INPUT_FILE = "Mall_Customers.csv"
OUTPUT_FILE = "Final_Mall_Customers_Predictions.csv"
MODEL_PATH = "Mall_Customers.joblib"
SCALER_PATH = "Mall_Customers_Scaler.joblib"

#-----------------------------------------------------------------
# Fucntion Name : display_Info
# Description   : It display the formated title
# Paramaters    : title(str)
# Return        : None
# Author        : Yogiraj Khaladkar 
# Date          : 13/04/2026
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
# Date           : 13/04/2026
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

# ------------------------------------------------------------------
# Function Name : scale_data
# Description   : Scales the input features using StandardScaler.
# Parameters    :
#               : X (DataFrame): Input feature dataset.
# Returns       :
#               : X_scaled (ndarray): Scaled dataset.
#               : scaler (StandardScaler): Trained scaler object.
# Author        : Yogiraj Khaladkar
# Date          : 13/04/2026
# ------------------------------------------------------------------
def scale_data(X):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    print("Data after scaling:")
    print(X_scaled[:5])

    return X_scaled, scaler

# ------------------------------------------------------------------
# Function Name : train_kmeans_model
# Description   : Trains a K-Means clustering model using the input dataset.
# Parameters    :
#               : X_scaled (DataFrame or ndarray): Scaled feature dataset.
# Returns       : KMeans: Trained K-Means clustering model.
# Author        : Yogiraj Khaladkar
# Date          : 13/04/2026
# ------------------------------------------------------------------
def train_kmeans_model(X_scaled):
    """Train the model """

    model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    return model

#------------------------------------------------------------------
# Function name : plot_Elbow
# Description   : Display the Elbow Method graph to find the optimal number of clusters
# Author        : Yogiraj Khaladkar
# Date          : 13/04/2026
#------------------------------------------------------------------
def plot_Elbow(X_scaled):
    """
    Plot the Elbow Method graph to determine the optimal number
    of clusters for the K-Means algorithm.
    """

    WCSS = []

    for i in range(1, 11):
        model = KMeans(
            n_clusters=i,
            random_state=42,
            n_init=10
        )
        model.fit(X_scaled)
        WCSS.append(model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), WCSS, marker='o')
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.grid(True)
    plt.show()

#------------------------------------------------------------------
# Function name : save_csv 
# Description   : Save the final output     
# Author        : Yogiraj Khaladkar
# Date          : 13/04/2026
#------------------------------------------------------------------
def save_csv(df,cluster):
        """Save the final output csv"""
     
        # Add columns
        df["cluster"] = cluster
       
        print("Dataset with cluster : \n")
        print(df.head(30))

        # Save to CSV
        df.to_csv(OUTPUT_FILE, index=False)

#-----------------------------------------------------------------
# Fucntion Name : save_model
# Description   : Save the model  
# Paramaters    : model
#                 path
# Return        : None
# Author        : Yogiraj Khaladkar
# Date          : 13/04/2026
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
# Date          : 13/04/2026
#------------------------------------------------------------------
def load_model(path = MODEL_PATH):
    """Load the train model"""

    loaded_model=joblib.load(path)

    print("Model successfully loaded")

    return loaded_model

# ------------------------------------------------------------------
# Function Name : save_scaler
# Description   : Save the trained StandardScaler object.
# Parameters    :
#               : scaler
#               : path
# Returns       : None
# Author        : Yogiraj Khaladkar
# Date          : 13/04/2026
# ------------------------------------------------------------------
def save_scaler(scaler, path=SCALER_PATH):

    joblib.dump(scaler, path)

    print("Scaler saved successfully:", path)

# ------------------------------------------------------------------
# Function Name : load_scaler
# Description   : Load the saved StandardScaler object.
# Parameters    :
#               : path
# Returns       : scaler
# Author        : Yogiraj Khaladkar
# Date          : 13/04/2026
# ------------------------------------------------------------------
def load_scaler(path=SCALER_PATH):

    scaler = joblib.load(path)

    print("Scaler loaded successfully")

    return scaler

#-----------------------------------------------------------------
# Fucntion Name  : main()
# Description    : Starting point of the application.This is main pipeline controller.
# Paramaters     : None
# Return         : None
# Author         : Yogiraj Khaladkar
# Date           : 13/04/2026
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

    # 3 : Scale the date 
    display_Info("Step 3 : Scale the data ")
    feature_headers = dataset[["AnnualIncome","SpendingScore"]]

    X_scaled,scaler= scale_data(feature_headers)
   
    # 4 Use elbow method 
    display_Info("Step 4 : Use Elbow method ")
    plot_Elbow(X_scaled)

    # 5 Build + Train model
    display_Info("Step 5 : Build + Train model")
    trained_model = train_kmeans_model(X_scaled) 
    print("Trained mode : ", trained_model) 
    
    # 6 Predictions 
    display_Info("Step 5 : Predictions ")
    predictions = trained_model.predict(X_scaled) 

    # 7 Save model  using joblib 
    display_Info("Step 7 : Save model using joblib")
    save_model(trained_model, MODEL_PATH) 
    save_scaler(scaler,SCALER_PATH)

    # 8 Load model and test a sample 
    display_Info("Step 8 : Load model and test a sample")
    loaded_model = load_model()
    loaded_scaler = load_scaler()

    new_customer = [[65, 40]]  # AnnualIncome, SpendingScore

    new_customer_scaled = loaded_scaler.transform(new_customer)

    cluster = loaded_model.predict(new_customer_scaled)

    print("Predicted Cluster:", cluster[0])

    # 9 Store the prediction results in a CSV file
    display_Info("Step 9 : Store the prediction results in a CSV file")
    save_csv(dataset,predictions)

#------------------------------------------------------------------
# Application starter 
#------------------------------------------------------------------
if __name__ =="__main__":
    main()
   