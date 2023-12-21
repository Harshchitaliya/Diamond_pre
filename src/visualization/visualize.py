import pandas as pd
import numpy as np
import sys 
import pathlib
import joblib
import mlflow
from sklearn.metrics import accuracy_score,precision_score
import yaml

def evaluate(model,X,y,params_file):
    prediction = model.predict(X)
    accuracy = accuracy_score(y,prediction )
    precision= precision_score(y,prediction,average="micro")

    mlflow.log_metric("accuracy",accuracy)
    mlflow.log_metric("precision",precision)
    mlflow.sklearn.log_model(
        sk_model = model,
        artifact_path="sklearn-model",
        
    )
    mlflow.log_param("n_estimators",params_file["n_estimators"])
    mlflow.log_param("random_state",params_file["random_state"])
    mlflow.log_param("max_depth",params_file["max_depth"])
    mlflow.log_param("learning_rate",params_file["learning_rate"])
    



def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent


    model_file = sys.argv[1]
    # Load the model.
    model = joblib.load(model_file)
    
    # Load the data.
    input_file = sys.argv[2]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/dvclive'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    params_file = home_dir.as_posix() + '/models/best_params.yaml'
    params = yaml.safe_load(open(params_file))
    
    
    
  

    
    X_test = pd.read_csv(data_path + '/X_test.csv')
    y_test = pd.read_csv(data_path + '/y_test.csv')

    # Evaluate train and test datasets.
    with mlflow.start_run() :
        
        
        evaluate(model, X_test, y_test,params)


        

if __name__ == "__main__":
    main()