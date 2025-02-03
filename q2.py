import requests
import numpy as np

model_endpoints = [
    "https://5fc9-89-30-29-68.ngrok-free.app/predict",
    "https://50da-89-30-29-68.ngrok-free.app/predict"
]

def get_predictions(sepal_length, sepal_width, petal_length, petal_width):
    predictions = []

    for url in model_endpoints:
        try:
            params = {
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width
            }
            response = requests.get(url, params=params)
            data = response.json()
            if data["status"] == "success":
                predictions.append(data["prediction"]["class_index"])
        except Exception as e:
            print(f"Error connecting to {url}: {e}")

    return predictions

def aggregate_prediction(predictions):
    if not predictions:
        return None, "No valid predictions received"
    
    unique, counts = np.unique(predictions, return_counts=True)
    consensus_index = unique[np.argmax(counts)]
    
    return int(consensus_index), "Consensus achieved"

# Input
sepal_length, sepal_width, petal_length, petal_width = 5.1, 3.5, 1.4, 0.2

predictions = get_predictions(sepal_length, sepal_width, petal_length, petal_width)

consensus, message = aggregate_prediction(predictions)

print(f"Aggregated Prediction: {consensus} ({message})")
