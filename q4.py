import json
import requests
import numpy as np

# Load model data (balances & weights)
def load_models():
    with open("models.json", "r") as f:
        return json.load(f)

# Save updated model data
def save_models(models):
    with open("models.json", "w") as f:
        json.dump(models, f, indent=4)

def get_predictions(sepal_length, sepal_width, petal_length, petal_width, models):
    predictions = {}

    for url in models.keys():
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

def weighted_consensus(predictions, models):
    if not predictions:
        return None, "No valid predictions received"

    weighted_votes = {}

    for url, pred in predictions.items():
        weight = models[url]["weight"]
        if pred not in weighted_votes:
            weighted_votes[pred] = 0
        weighted_votes[pred] += weight  # Weighted vote based on trust

    consensus_class = max(weighted_votes, key=weighted_votes.get)

    return int(consensus_class), "Weighted consensus achieved"

def update_stakes(predictions, consensus_class, models):
    reward = 10  # Euros for correct predictions
    penalty = 50  # Slashing penalty for incorrect predictions

    for url, pred in predictions.items():
        if pred == consensus_class:
            models[url]["balance"] += reward  # Reward accurate models
        else:
            models[url]["balance"] -= penalty  # Slash inaccurate models

            # Reduce trust weight if balance decreases
            models[url]["weight"] = max(0.1, models[url]["balance"] / 1000)

        # If balance reaches 0, remove model
        if models[url]["balance"] <= 0:
            print(f"Model {url} has been removed due to insufficient balance.")
            del models[url]

    save_models(models)  # Save updated balances
    print(f"Updated Balances: {models}")

def run_consensus_round(sepal_length, sepal_width, petal_length, petal_width):
    models = load_models()  # Load latest stake data

    predictions = get_predictions(sepal_length, sepal_width, petal_length, petal_width, models)
    
    if not predictions:
        print("No models responded.")
        return
    
    consensus, message = weighted_consensus(predictions, models)
    
    update_stakes(predictions, consensus, models)  # Apply PoS penalties/rewards

    print(f"Final Consensus Prediction: {consensus} ({message})")