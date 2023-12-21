import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import pathlib
import pandas as pd
import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier


curr_dir = pathlib.Path(__file__)
home_dir = curr_dir.parent.parent.parent
params_file = home_dir.as_posix() + '/params.yaml'


def train_models(params, X_train, X_test, y_train, y_test):
  """
  Trains all models specified in the parameters, evaluates them, and returns the best model.

  Args:
    params: A dictionary containing model parameters (including model names and their configurations).
    X_train, X_test, y_train, y_test: Split data for training and testing.

  Returns:
    best_model, best_score: The best performing model and its corresponding score.
  """
  with open(params_file, "r") as f:
   params = yaml.safe_load(f)
  y_train = np.ravel(y_train) 
  y_test = np.ravel(y_test)

  best_model = None
  best_score = 0
  for model_name, model_params in params["models"].items():
    # Import model class based on model name
    model_class = eval(model_name)

    # Instantiate and train the model
    model = model_class(**model_params)
    model.fit(X_train, y_train)

    # Evaluate and compare performance
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    if score > best_score:
      best_model = model
      best_score = score
      best_model_params = model_params

  return [best_model,best_model_params]


def save_model(model, output_path):
    # Save the trained model to the specified output path
    joblib.dump(model, output_path + '/model.joblib')


def save_params(params,output_path):
   
    with open(output_path+"/best_params.yaml", 'w') as file:
     yaml.dump(params, file)

     

def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'
    

    input_file = sys.argv[1]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/models'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
   
    X_train = pd.read_csv(data_path + 'X_train.csv')
    y_train = pd.read_csv(data_path + 'y_train.csv')
    X_test = pd.read_csv(data_path + 'X_test.csv')
    y_test = pd.read_csv(data_path + 'y_test.csv')

    trained_model = train_models(params_file,X_train,X_test,y_train,y_test)
    save_model(trained_model[0], output_path)
    save_params(trained_model[1], output_path)

    

if __name__ == "__main__":
    main()

 