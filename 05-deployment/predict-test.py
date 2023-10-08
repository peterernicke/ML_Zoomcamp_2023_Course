# 5.4 Serving the churn model with Flask
#!/usr/bin/env python
# coding: utf-8

import requests

## change because of EB
#host = 'churn-serving-env.eba-gg58yj4v.eu-west-1.elasticbeanstalk.com'
#url = f'http://{host}/predict'

url = 'http://localhost:9696/predict'


customer_id = 'xyz-123'
customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 29.85,
    "totalcharges": 29.85
}

response = requests.post(url, json=customer).json()
print(response)

# On Windows you can use 
# pip install waitress
# That is similar to gunicorn
# waitress-serve --listen=0.0.0.0:9696 predict:app

if response['churn'] == True:
    print('sending promo email to %s' % customer_id)
else:
    print('not sending promo email to %s' % customer_id)



