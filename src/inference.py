import json
import pickle
import requests

def simple_predict(model_pickle_path: str):
    # Loading model to compare the results
    #
    with open(model_pickle_path,'rb') as f:
        model = pickle.load(f)

    predictions = model.predict([[2, 9, 6], [1, 4, 4]])

    print(f"My prediction: ${ [ round(prediction,2) for prediction in predictions] }")


def predict_using_api(url: str):
    input_features = {
        "experience_score": 4,
        "test_score": 7,
        "interview_score": 8
    }
    headers = {"Content-Type": "application/json"}
    payload = json.dumps(input_features)
    response = requests.post(url, data=payload, headers=headers)
    prediction = response.json()
    print(f"My prediction: {prediction}")

if __name__ == "__main__":
    #simple_predict('models/model.pkl')
    predict_using_api("http://127.0.0.1:5000/predict2")
