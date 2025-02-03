import requests
import numpy as np

# List of group members' API endpoints (replace with actual ngrok URLs)
model_endpoints = [
    "https://5fc9-89-30-29-68.ngrok-free.app/predict",
    "https://50da-89-30-29-68.ngrok-free.app/predict"
]

# Initialize model weights (starting with equal influence)
model_weights = {url: 1.0 for url in model_endpoints}

def get_predictions(sepal_length, sepal_width, petal_length, petal_width):
    predictions = {}
    
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
                predictions[url] = data["prediction"]["class_index"]
        except Exception as e:
            print(f"Error connecting to {url}: {e}")

    return predictions

def weighted_consensus(predictions):
    if not predictions:
        return None, "No valid predictions received"

    # Aggregate weighted votes
    weighted_votes = {}
    for url, pred in predictions.items():
        weight = model_weights[url]
        if pred not in weighted_votes:
            weighted_votes[pred] = 0
        weighted_votes[pred] += weight  # Accumulate weighted votes

    # Determine the class with the highest weighted vote
    consensus_class = max(weighted_votes, key=weighted_votes.get)

    return int(consensus_class), "Weighted consensus achieved"

def update_weights(predictions, consensus_class):
    learning_rate = 0.1  # Controls how fast weights change

    for url, pred in predictions.items():
        if pred == consensus_class:
            model_weights[url] = min(1.0, model_weights[url] + learning_rate)  # Increase weight
        else:
            model_weights[url] = max(0.1, model_weights[url] - learning_rate)  # Decrease weight

    print(f"Updated Weights: {model_weights}")

def run_consensus_round(sepal_length, sepal_width, petal_length, petal_width):
    predictions = get_predictions(sepal_length, sepal_width, petal_length, petal_width)
    consensus, message = weighted_consensus(predictions)
    update_weights(predictions, consensus)

    print(f"Final Consensus Prediction: {consensus} ({message})")
