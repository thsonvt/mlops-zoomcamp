import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import mlflow
from mlflow.tracking import MlflowClient

# MLFLOW_TRACKING_URI = "sqlite:///mlflow-homework.db"

# client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# mlflow.set_experiment("nyc-taxi-experiment-2023")

# os.environ["AWS_PROFILE"] = "sonle" # fill in with your AWS profile. More info: https://docs.aws.amazon.com/sdk-for-java/latest/developer-guide/setup.html#setup-credentials

TRACKING_SERVER_HOST = "0.0.0.0" # fill in with the public DNS of the EC2 instance
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")

mlflow.set_experiment("nyc-taxi-2023-homework-2")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    mlflow.sklearn.autolog()

    with mlflow.start_run():

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)

        mlflow.log_metric("rmse", rmse)


if __name__ == '__main__':
    run_train()
