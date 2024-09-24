import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri('mlruns')
# Load the Boston Housing dataset from CSV
data = pd.read_csv('data/BostonHousing.csv')

# Display the first few rows of the dataset for verification
print("Dataset preview:")
print(data.head())

# Assume the target variable is 'medv' (Median value of owner-occupied homes)
X = data.drop('medv', axis=1)  # Replace 'medv' with the actual target column name if different
y = data['medv']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a new experiment
experiment_name = "Boston_Housing_Experiment"
mlflow.set_experiment(experiment_name)
print(f"Using experiment: {experiment_name}")

# Function to train and log model
def train_and_log_model(model, model_name):
    with mlflow.start_run():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        
        # Log parameters and metrics
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("mse", mse)
        
        # Log the model
        mlflow.sklearn.log_model(model, model_name)
        
        print(f"{model_name} MSE: {mse}")
        print(f"Logged {model_name} to MLflow.")

# Train Linear Regression
linear_model = LinearRegression()
train_and_log_model(linear_model, "Linear_Regression")

# Train Random Forest
rf_model = RandomForestRegressor()
train_and_log_model(rf_model, "Random_Forest")
