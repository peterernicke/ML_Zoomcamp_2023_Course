# 5.4 Serving the churn model with Flask
import pickle

from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model1.bin'
dv_file = 'dv.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)

app = Flask('score')

# Here we use the POST method, because we want to send information to the web service
@app.route('/predict', methods=['POST'])
def predict():
    # json = Python dictionary
    client = request.get_json()

    X = dv.transform([client])
    model.predict_proba(X)
    y_pred = model.predict_proba(X)[0,1] 
    y_pred = round(y_pred,3)
    score = y_pred >= 0.5

    result = {
        # the next line raises an error so we need to change it
        #'churn_probability': y_pred,
        'probability': float(y_pred),
        # the next line raises an error so we need to change it
        #'churn': churn
        'application_accepted': bool(score)
    }

    return jsonify(result) 

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)