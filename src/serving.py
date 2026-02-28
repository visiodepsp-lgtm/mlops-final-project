import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask("My Model API") #Initialize the flask App
model = pickle.load(open('models/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

@app.route('/predict2',methods=['POST'])
def predict2():
    features_dict = request.get_json()
    print(f"features dict: {features_dict}")
    input_features = [ features_dict.get("experience_score"), features_dict.get("test_score"), features_dict.get("interview_score")]
    #int_features = [int(x) for x in ]
    final_features = [np.array(input_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    return jsonify({"prediction": output})
    #return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


if __name__ == "__main__":
    #app.run(debug=True)
    app.run()
