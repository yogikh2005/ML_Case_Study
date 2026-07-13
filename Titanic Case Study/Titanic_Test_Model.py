#-----------------------------------------------------------------
# Required Python Packages 
#-----------------------------------------------------------------

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.metrics import ( 
accuracy_score, 
confusion_matrix,  
classification_report, 
) 

from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.pipeline import Pipeline 

import joblib

#-----------------------------------------------------------------
# File Paths
#-----------------------------------------------------------------

INPUT_FILE = "TitanicDataset.csv"
MODEL_PATH = "Titanic_pipeline.joblib"

#-----------------------------------------------------------------
# Fucntion Name : display_Info
# Description   : It display the formated title
# Paramaters    : title(str)
# Author        : Yogiraj Khaladkar
# Date          : 14/03/2026
#------------------------------------------------------------------

def display_Info(title):
    print("\n"+"="*70)
    print(title)
    print("="*70)

#-----------------------------------------------------------------
# Fucntion Name : dataset_statistics
# Description   : It shows basic information about the dataset
# Paramaters    : Dataset(df),message
#                 df -> pandas dataframe object
#                 message -> Heading text to display
# Return        : None
# Date          : 14/03/2026
# Author        : Yogiraj Khaladkar
#------------------------------------------------------------------

def dataset_statistics(df):
    """ Print Basic data of dataset """
    
    print("\n First 5 rows of dataset")
    print(df.head)

    print("\n Shape of dataset")
    print(df.shape)

    print("\n Columns of dataset")
    print(df.columns.tolist())

    print("\n Missing value in dataset")
    print(df.isnull().sum())

    print("\ndataset describe" )
    print(df.describe())

#-----------------------------------------------------------------
# Fucntion Name : clean_Titanic_Data
# Description   : It does preprocessing
#                 It removed unnecessary columns
#                 It handle missing values
#                 It convert text data to numric format
#                 It does encodimg to  catagorial columns
# Paramaters    : df -> Pandas dataframe
# Return        : df -> Clean Pandas dataframe
# Author        : Yogiraj Khaladkar
# Date          : 14/03/2026 
#------------------------------------------------------------------
def clean_Titanic_Data(df):
    """Clean the dataset & Handle the missing the value"""

    print(df.head)

    # Remove unnecessary column
    drop_columns=["Passengerid","zero","Name","Cabin"]
    existing_columns=[col for col in drop_columns if col in df.columns]

    print("\n Columns to be droped : ")
    print(existing_columns)

    # drop unwanted colums
    df=df.drop(columns=existing_columns)

    display_Info("Data after the cleaning\n")
    print(df.head)

    # Handle age columns

    if "Age" in df.columns:
        print("Age columns before filling missing value :")
        print(df["Age"].head(10))

        # coerce invalied value gets converted as NAN
        df["Age"] = pd.to_numeric(df["Age"],errors="coerce")

        age_median=df["Age"].median()
        
        # Replace missing value
        df["Age"]=df["Age"].fillna(age_median)

        print("Age columns after preprocessing")
        print(df["Age"].head(10))


    # Handle fire columns 
    if "Fare" in df.columns :
        print("\n Fare columns before preprocessing")
        print(df["Fare"].head(10))

        # coerce invalied value gets converted as NAN
        df["Fare"] = pd.to_numeric(df["Fare"],errors="coerce")
        fare_median=df["Fare"].median()
        
        print("\n Mode of the Fare columns : ",fare_median)

        df["Fare"]=df["Fare"].fillna(fare_median)

        print("Fare columns after preprocessing")
        print(df["Fare"].head(10))

    # Handle Embarked columns
    if "Embarked" in df.columns :
        print("\n Embarked columns before preprocessing")
        print(df["Embarked"].head(10))

        # converted data into string
        df["Embarked"]=df["Embarked"].astype(str).str.strip()

        # Remove missing values

        df["Embarked"]=df["Embarked"].replace(['nan','None',''],np.nan)

        # get most frequnt_value
        embarked_mode = df["Embarked"].mode()[0]
        print("\n Mode of the Embarked columns : ",embarked_mode)

        df["Embarked"]=df["Embarked"].fillna(embarked_mode)


        print("Embarked columns after preprocessing")
        print(df["Embarked"].head(10))


    # Handle Sex columns 
    if "Sex" in df.columns :
        print("\n Sex columns before preprocessing")
        print(df["Sex"].head(10))

        # coerce invalied value gets converted as NAN
        df["Sex"] = pd.to_numeric(df["Sex"],errors="coerce")
        

        print("Sex columns after preprocessing")
        print(df["Sex"].head(10))
    
    print("Data after preprocessing")
    print(df.head())

    print("\n Missing value after preprocsiing")
    print(df.isnull().sum())
    
    # encode the embraked columns
    df = pd.get_dummies(df,columns=['Embarked'],drop_first=True)

    print("\nDate after the encoding")
    
    print(df.head())
    
    print("Shape of dataset : ",df.shape)

    # convert boolean columns into integer
    for col in df.columns:
        if df[col].dtype == bool:
            df[col]=df[col].astype(int)
    
    print("\nDate after the encoding")
    
    print(df.head())
    
    print("Shape of dataset : ",df.shape)
    
    return df

#-----------------------------------------------------------------
# Function Name     : split_dataset 
# Description       : Split the dataset with train_percentage 
# Parameters        : Dataset with related information 
# Return            : Dataset after splitting 
# Author            : Yogiraj Khaladkar
# Date              : 14/03/2026 
#-----------------------------------------------------------------

def split_dataset(dataset, feature_headers,target_header, train_percentage ,random_state=42): 
    """Split dataset into train/test""" 

    train_x, test_x, train_y, test_y = train_test_split( 
        feature_headers,target_header, 
        train_size=train_percentage, random_state=random_state
    ) 

    return train_x, test_x, train_y, test_y 

#-----------------------------------------------------------------
# Fucntion Name : bulid_pipeline
# Description   : Build a Pipeline
# Author        : Yogiraj Khaladkar
# Date          : 14/03/2026
#------------------------------------------------------------------

def build_pipeline():
    """Build a Pipeline """
    pipe = Pipeline(steps=[
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])
    return pipe

#-----------------------------------------------------------------
# Fucntion Name : train_pipeline
# Description   : Train a Pipeline
# Author        : Yogiraj Khaladkar
# Date          : 14/03/2026
#------------------------------------------------------------------

def train_pipeline(pipeline, X_train, y_train):
    '''Train a Pipeline'''
    pipeline.fit(X_train, y_train) 
    return pipeline

#------------------------------------------------------------------
# Function name : plot_confusion_matrix_matshow 
# Description   : Display Confusion Matrix     
# Author        : Yogiraj Khaladkar
# Date          : 14/03/2026
#------------------------------------------------------------------

def plot_confusion_matrix_matshow(y_true, y_pred, title="Confusion Matrix"): 
    """Display Confusion Matrix"""

    cm = confusion_matrix(y_true, y_pred) 
    fig, ax = plt.subplots() 
    cax = ax.matshow(cm) 
    fig.colorbar(cax) 

    for (i, j), v in np.ndenumerate(cm): 
        ax.text(j, i, str(v), ha='center', va='center') 
        ax.set_xlabel("Predicted") 
        ax.set_ylabel("Actual") 
        ax.set_title(title) 

    plt.tight_layout() 
    plt.show()

#-----------------------------------------------------------------
# Fucntion Name : save_model
# Description   : Save the model  
# Paramaters    : model
#                 path
# Return        : None
# Author        : Yogiraj Khaladkar
# Date          : 14/03/2026
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
# Date          : 14/03/2026
#------------------------------------------------------------------

def load_model(path = MODEL_PATH):
    """Load the train model"""

    loaded_model=joblib.load(path)

    print("Model successfully loaded")

    return loaded_model

#-----------------------------------------------------------------
# Fucntion Name : main
# Description   : Main function from where exexution starts
# Paramaters    : NOne
# Return        : None
# Author        : Yogiraj Khaladkar
# Date          : 14/03/2026
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
    dataset = clean_Titanic_Data(dataset)

    feature_headers = dataset.drop("Survived",axis=1)
    target_header =  dataset['Survived']

    # 4 Split 
    display_Info("Step 4 : Split dataset")
    train_x, test_x, train_y, test_y = split_dataset(dataset,feature_headers,target_header, 0.7)
    
    print("Train_x Shape :: ", train_x.shape) 
    print("Train_y Shape :: ", train_y.shape) 
    print("Test_x Shape :: ", test_x.shape) 
    print("Test_y Shape :: ", test_y.shape) 

    # 5 Build + Train Pipeline 
    display_Info("Step 5 : Build + Train Pipeline")
    pipeline = build_pipeline() 
    trained_model = train_pipeline(pipeline, train_x, train_y) 
    print("Trained Pipeline :: ", trained_model) 
    
    # 6 Predictions 
    predictions = trained_model.predict(test_x) 

    # 7 Metrics 
    display_Info("Step 7 : Display the Metrics")
    print("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x))) 
    print("Test Accuracy  :: ", accuracy_score(test_y, predictions)) 
    print("Classification Report:\n", classification_report(test_y, predictions)) 
    print("Confusion Matrix:\n", confusion_matrix(test_y, predictions)) 

    # 8 Plot confusion matrix
    display_Info("Step 8 : Plot confusion matrix")
    plot_confusion_matrix_matshow(test_y, predictions)

    # 9 Save model (Pipeline) using joblib 
    display_Info("Step 9 : Save model (Pipeline) using joblib")
    save_model(trained_model, MODEL_PATH) 

    # 10 Load model and test a sample 
    display_Info("Step 10 : Load model and test a sample")
    loaded = load_model(MODEL_PATH) 
    sample = test_x.iloc[[0]] 
    pred_loaded = loaded.predict(sample) 
    print(f"Loaded model prediction for first test sample: {pred_loaded[0]}") 

#------------------------------------------------------------------
# Application starter 
#------------------------------------------------------------------
if __name__ =="__main__":
    main()