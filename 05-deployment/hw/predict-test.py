#!/usr/bin/env python
# coding: utf-8

import requests

url = 'http://localhost:9696/predict'

client_id = 'xyz-123'
client = {"job": "unknown", 
          "duration": 270, 
          "poutcome": "failure"}

#client = {"job": "retired", "duration": 445, "poutcome": "success"}

response = requests.post(url, json=client).json()
print(response)

# On Windows you can use 
# pip install waitress
# That is similar to gunicorn
# waitress-serve --listen=0.0.0.0:9696 predict:app

#if response['churn'] == True:
#    print('sending promo email to %s' % client_id)
#else:
#    print('not sending promo email to %s' % client_id)



