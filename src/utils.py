import numpy as np
from sklearn.metrics import mean_squared_error

def calculate_rmspe(y_true, y_pred):
    
    percentage_error = (y_true - y_pred) / y_true
    percentage_error[y_true == 0] = 0
    squared_percentage_error = percentage_error ** 2
    return np.sqrt(np.mean(squared_percentage_error))

def evaluate_model(model, train_inputs, train_targets, val_inputs, val_targets):
    
    train_preds = model.predict(train_inputs)
    val_preds = model.predict(val_inputs)

    # RMSE
    train_rmse = np.round(mean_squared_error(train_targets, train_preds, squared=False), 5)
    val_rmse = np.round(mean_squared_error(val_targets, val_preds, squared=False), 5)

    # RMSPE
    train_rmspe = np.round(calculate_rmspe(train_targets, train_preds), 5)
    val_rmspe = np.round(calculate_rmspe(val_targets, val_preds), 5)

    print(f"Train RMSE: {train_rmse}, Val RMSE: {val_rmse}")
    print(f"Train RMSPE: {train_rmspe}, Val RMSPE: {val_rmspe}")