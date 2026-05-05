#----------------------------------------------------------------- 
# Required Python Packages 
#----------------------------------------------------------------- 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import joblib

#----------------------------------------------------------------- 
# File Paths 
#-----------------------------------------------------------------

INPUT_FAKE_PATH = "Fake.csv"
INPUT_TRUE_PATH = "True.csv"
OUTPUT_PATH = "FakeNews_Output.csv" 
MODEL_SOFT_PATH = "FakeNews_SoftVoting.joblib" 
MODEL_HARD_PATH = "FakeNews_HardVoting.joblib" 
VECTORIZER_PATH = "FakeNews_Tfidf.joblib"

#-----------------------------------------------------------------
# Function Name : display_Info
# Description   : It display the formated title
# Parameters    : title(str)
# Return        : None
# Author        : Yogiraj Khaladkar 
# Date          : 01/05/2026
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
# Date           : 01/05/2026
#------------------------------------------------------------------
def dataset_statistics(df):
    """It shows basic information about the dataset"""

    print("\n First 5 rows of dataset")
    print(df.head())

    print("\n Shape of dataset")
    print(df.shape)

    print("\n Columns of dataset")
    print(df.columns.tolist())

    print("\n Statistical report")
    print(df.describe())

    print("\n Missing values in dataset")
    print(df.isnull().sum())

    plt.figure(figsize=(6,4))
    sns.countplot(x='label', data=df)
    plt.title("Distribution of Outcome")
    plt.show()

#-----------------------------------------------------------------
# Function Name  : preprocess_dataset 
# Description    : Concatenates fake and true news, removes extra columns, and encodes
# Parameters     : Fdf, Tdf
# Return         : Processed DataFrame
# Author         : Yogiraj Khaladkar
# Date           : 01/05/2026
#-----------------------------------------------------------------
def preprocess_dataset(Fdf, Tdf):
    """Preprocess dataset: label, concat, drop columns, and encode"""
    
    # Label the datasets
    Fdf['label'] = "Fake" 
    Tdf['label'] = "True" 

    # Concat
    df = pd.concat([Fdf, Tdf], ignore_index=True)

    # Drop the columns which values are unknown or unnecessary
    df.drop(['subject', 'date'], axis=1, inplace=True, errors='ignore')

    # Label encoding 
    df['label'] = df["label"].map({"Fake": 0, "True": 1})

    # Combine title and text into a single feature
    df['combined_text'] = df['title'].fillna('') + " " + df['text'].fillna('')

    print("\nProcessed Dataset (First 5 records):")
    print(df.head())
    print("\nProcessed Dataset shape :", df.shape)

    return df

#-----------------------------------------------------------------
# Function Name  : split_dataset 
# Description    : Split the dataset with test_percentage 
# Parameters     : df, test_percentage
# Return         : Dataset after splitting 
# Author         : Yogiraj Khaladkar
# Date           : 01/05/2026
#-----------------------------------------------------------------
def split_dataset(df, test_percentage, random_state=42): 
    """Split dataset into train/test""" 

    X = df['combined_text']
    Y = df['label']

    train_x, test_x, train_y, test_y = train_test_split( 
        X, Y, 
        test_size=test_percentage, random_state=random_state
    ) 

    return train_x, test_x, train_y, test_y 

#-----------------------------------------------------------------
# Function Name  : extract_features 
# Description    : Applies TF-IDF vectorization
# Parameters     : train_x, test_x
# Return         : train_x_tfidf, test_x_tfidf, tfidf_vectorizer
# Author         : Yogiraj Khaladkar
# Date           : 01/05/2026
#-----------------------------------------------------------------
def extract_features(train_x, test_x):
    """Extract TF-IDF features from text"""
    
    tfidf = TfidfVectorizer(stop_words='english')
    
    train_x_tfidf = tfidf.fit_transform(train_x) # Learn and transform
    test_x_tfidf = tfidf.transform(test_x)       # Only transform
    
    print("Feature extraction completed successfully.")
    
    return train_x_tfidf, test_x_tfidf, tfidf

# ------------------------------------------------------------------
# Function Name : train_models
# Description   : Trains Soft and Hard Voting Classifiers.
# Parameters    :
#               : train_x (Sparse Matrix): Training feature dataset.
#               : train_y (Series): Training target labels.
# Returns       : Soft Voting Model, Hard Voting Model
# Author        : Yogiraj Khaladkar
# Date          : 01/05/2026
# ------------------------------------------------------------------
def train_models(train_x, train_y):
    """Train the Voting models"""

    # Base models
    model_lr = LogisticRegression(max_iter=1000)
    model_dt = DecisionTreeClassifier(random_state=42)

    # Soft voting classification
    soft_model = VotingClassifier(
        estimators=[('lr', model_lr), ('dt', model_dt)],
        voting="soft"
    )

    # Hard voting classification
    hard_model = VotingClassifier(
        estimators=[('lr', model_lr), ('dt', model_dt)],
        voting="hard"
    )

    soft_model.fit(train_x, train_y)
    hard_model.fit(train_x, train_y)

    print("Both models trained successfully")
    return soft_model, hard_model

#------------------------------------------------------------------
# Function name : save_csv 
# Description   : Save the final output csv with actual and predicted
# Author        : Yogiraj Khaladkar
# Date          : 01/05/2026
#------------------------------------------------------------------
def save_csv(test_x, test_y, pred_soft, pred_hard):
    """Save the final output csv"""
    
    # Reconstructing dataframe for output
    result_df = pd.DataFrame({
        'Text': test_x,
        'Actual': test_y,
        'Predicted_Soft': pred_soft,
        'Predicted_Hard': pred_hard
    })

    # Save to CSV
    result_df.to_csv(OUTPUT_PATH, index=False)
    
    print("\nOutput saved successfully to", OUTPUT_PATH)

#-----------------------------------------------------------------
# Function Name : save_artifacts
# Description   : Save the models and vectorizer using joblib
# Parameters    : soft_model, hard_model, tfidf
# Return        : None
# Author        : Yogiraj Khaladkar
# Date          : 01/05/2026
#------------------------------------------------------------------
def save_artifacts(soft_model, hard_model, tfidf):
    """ Save the models and vectorizer"""
    joblib.dump(soft_model, MODEL_SOFT_PATH)
    joblib.dump(hard_model, MODEL_HARD_PATH)
    joblib.dump(tfidf, VECTORIZER_PATH)

    print("Soft Voting Model Preserved successfully")
    print("Hard Voting Model Preserved successfully")
    print("TF-IDF Vectorizer Preserved successfully")

#------------------------------------------------------------------
# Function name : plot_heatmap 
# Description   : Display the confusion matrix using a heatmap.    
# Author        : Yogiraj Khaladkar
# Date          : 01/05/2026
#------------------------------------------------------------------
def plot_heatmap(cm, title):
    """Display the confusion matrix."""
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()

#-----------------------------------------------------------------
# Function Name  : main()
# Description    : Starting point of the application. This is main pipeline controller.
# Parameters     : None
# Return         : None
# Author         : Yogiraj Khaladkar
# Date           : 01/05/2026
#------------------------------------------------------------------
def main():
    """Main function from where execution starts"""
    
    # 1 Load CSVs 
    display_Info("Step 1: Load Datasets")
    try:
        Fdf = pd.read_csv(INPUT_FAKE_PATH)
        Tdf = pd.read_csv(INPUT_TRUE_PATH)
        print("Datasets loaded successfully")
    except FileNotFoundError:
        print("Error: Ensure 'Fake.csv' and 'True.csv' are in the directory.")
        return

    # 2 Preprocessing & EDA
    display_Info("Step 2 : Concat and Preprocess Data")
    df = preprocess_dataset(Fdf, Tdf)
    
    display_Info("Step 3 : Dataset Statistics & Visualization")
    dataset_statistics(df) 
    
    # 4 Split Data
    display_Info("Step 4 : Split dataset")
    train_x, test_x, train_y, test_y = split_dataset(df, test_percentage=0.20)
    
    print("X_train Shape :", train_x.shape)
    print("X_test Shape :", test_x.shape)
    print("Y_train Shape :", train_y.shape)
    print("Y_test Shape :", test_y.shape)

    # 5 Feature Extraction (TF-IDF)
    display_Info("Step 5 : Feature Extraction (TF-IDF)")
    train_x_tfidf, test_x_tfidf, tfidf_vectorizer = extract_features(train_x, test_x)
    
    print("Shape of X_train_tfidf :", train_x_tfidf.shape)
    print("Shape of X_test_tfidf :", test_x_tfidf.shape)

    # 6 Build + Train Pipeline 
    display_Info("Step 6 : Build and Train Models")
    soft_model, hard_model = train_models(train_x_tfidf, train_y) 
    
    # 7 Predictions & Evaluation
    display_Info("Step 7 : Model Evaluation")
    
    # Soft Voting Evaluation
    display_Info("--> Soft Voting Model Evaluation")
    pred_soft = soft_model.predict(test_x_tfidf) 
    acc_soft = accuracy_score(test_y, pred_soft)
    cm_soft = confusion_matrix(test_y, pred_soft)
    
    print("Soft Voting Accuracy : {:.2f}%".format(acc_soft * 100))
    print("\nClassification Report (Soft Voting):\n", classification_report(test_y, pred_soft)) 
    plot_heatmap(cm_soft, "Confusion Matrix: Soft Voting Model")

    # Hard Voting Evaluation
    display_Info("--> Hard Voting Model Evaluation")
    pred_hard = hard_model.predict(test_x_tfidf) 
    acc_hard = accuracy_score(test_y, pred_hard)
    cm_hard = confusion_matrix(test_y, pred_hard)
    
    print("Hard Voting Accuracy : {:.2f}%".format(acc_hard * 100))
    print("\nClassification Report (Hard Voting):\n", classification_report(test_y, pred_hard)) 
    plot_heatmap(cm_hard, "Confusion Matrix: Hard Voting Model")

    # 8 Save artifacts using joblib 
    display_Info("Step 8 : Save Models and Vectorizer")
    save_artifacts(soft_model, hard_model, tfidf_vectorizer) 

    # 9 Store the prediction results in a CSV file
    display_Info("Step 9 : Store the prediction results in a CSV file")
    save_csv(test_x, test_y, pred_soft, pred_hard)
    
    display_Info("Pipeline Execution Completed")

#------------------------------------------------------------------
# Application starter 
#------------------------------------------------------------------
if __name__ == "__main__":
    main()