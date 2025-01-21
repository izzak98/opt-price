import json
import optuna

from varpi.QLSTM import LSTM_Model
with open("config.json", "r") as f:
    CONFIG = json.load(f)


def load_qlstm_model():
    """
    Fetch the best parameters from the Optuna study and load the specified model.

    Args:
    model_name (str): Name of the model to load. Either 'lstm' or 'dense'.

    Returns:
    model: The loaded model with best parameters.
    best_params (dict): The best parameters found by Optuna.
    """
    # Load the appropriate study
    study_name = "LSTM"
    study = optuna.load_study(
        study_name=study_name,
        storage=CONFIG["general"]["db_path"]
    )

    best_params = study.best_params

    # Create and load the model
    model = LSTM_Model(
        lstm_layers=best_params['raw_lstm_layers'],
        lstm_h=best_params['raw_lstm_h'],
        hidden_layers=[best_params[f'raw_hidden_layer_{i}']
                       for i in range(best_params['raw_hidden_layers'])],
        hidden_activation=best_params['hidden_activation'],
        market_lstm_layers=best_params['market_lstm_layers'],
        market_lstm_h=best_params['market_lstm_h'],
        market_hidden_layers=[best_params[f'market_hidden_layer_{i}'] for i in range(
            best_params['market_hidden_layers'])],
        market_hidden_activation=best_params['market_activation'],
        dropout=best_params['dropout'],
        layer_norm=best_params['use_layer_norm']
    )

    return model, best_params
