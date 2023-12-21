# make_dataset.py
import pathlib
import yaml
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    color = ["D","E","F","G","H","I","J","K"]
    df = df[df["color"].isin(color)]

    return df

def split_data(df, test_split, seed):
    # Split the dataset into train and test sets
    X = df[["Max","Min","Minf","Maxf"]]
    y = df["color"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=seed)
    

    # Label encode the target
    le = LabelEncoder()
    le.fit(y_train)
    y_train=le.transform(y_train)
    y_test=le.transform(y_test)

    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    return X_train, X_test, y_train, y_test
    

def save_data(X_train,X_test,y_train,y_test, output_path):
    # Save the split datasets to the specified output path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    X_train.to_csv(output_path + 'X_train.csv', index=False)
    X_test.to_csv(output_path + 'X_test.csv', index=False)
    y_train.to_csv(output_path + 'y_train.csv', index=False)
    y_test.to_csv(output_path + 'y_test.csv', index=False)
    


def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["make_dataset"]

    input_file = sys.argv[1]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/data/processed/'
    
    data = load_data(data_path)
    X_train,X_test,y_train,y_test = split_data(data, params['test_split'], params['seed'])
    save_data(X_train,X_test,y_train,y_test, output_path)

if __name__ == "__main__":
    main()


