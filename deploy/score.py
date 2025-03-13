import json
import os
import joblib
import pandas as pd
from azureml.core.model import Model

def init():
    global model
    # Load the model from the registered model path
    #model_path = Model.get_model_path("mobile_price_predictor")
    print(os.getenv('AZUREML_MODEL_DIR'))
    model_dir = os.path.join(os.getenv('AZUREML_MODEL_DIR'))
    print(model_path)
    model_path = os.path.join(model_dir, "mobile_price_predictor.pkl")
    model = joblib.load(model_path)
    #model = joblib.load(model_path + "/1/mobile_price_predictor.pkl")
    print(f"Model loaded from {model_path}")

def run(raw_data):
    try:
        # Parse the input JSON data
        data = json.loads(raw_data)
        
        # Convert input to DataFrame
        # Expecting input like: {"data": [[174.0, 6.0, 3600.0, 6.1, 2024], ...]}
        input_data = pd.DataFrame(
            data["data"],
            columns=["Mobile Weight", "RAM", "Battery Capacity", "Screen Size", "Launched Year"]
        )

        # Make predictions
        predictions = model.predict(input_data)

        # Return predictions as JSON
        return json.dumps({"predictions": predictions.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})
