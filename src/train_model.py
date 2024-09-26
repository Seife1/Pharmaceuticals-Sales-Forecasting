import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
from datetime import datetime
from utils import evaluate_model

def train_random_forest(train_inputs, train_targets, val_inputs, val_targets):
    # Train the model
    random_forest_model = RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=10)
    random_forest_model.fit(train_inputs, train_targets)

    # Evaluate model
    evaluate_model(random_forest_model, train_inputs, train_targets, val_inputs, val_targets)

    # Serialize the model with timestamp
    timestamp = datetime.now().strftime('%d-%m-%Y-%H-%M-%S-%f')
    model_filename = f"../models/random_forest_model-{timestamp}.pkl"
    joblib.dump(random_forest_model, model_filename)

    print(f"Model saved as {model_filename}")
    return random_forest_model