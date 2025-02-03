from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.datasets import load_iris

model = joblib.load("iris_model.pkl")

iris = load_iris()

app = Flask(__name__)

@app.route('/')
def home():
    return "Iris Prediction API"

@app.route('/predict', methods=['GET'])
def predict():
    try:
        sepal_length = float(request.args.get('sepal_length'))
        sepal_width = float(request.args.get('sepal_width'))
        petal_length = float(request.args.get('petal_length'))
        petal_width = float(request.args.get('petal_width'))

        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        prediction = model.predict(features)
        class_name = iris.target_names[prediction[0]]

        response = {
            "status": "success",
            "prediction": {
                "class_index": int(prediction[0]),
                "class_name": class_name
            }
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
