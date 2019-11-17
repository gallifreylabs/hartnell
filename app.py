# import os
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from flask_cors import CORS

import joblib

import urllib.request
import sys
import importlib.util

app = Flask(__name__)
api = Api(app)
CORS(app, support_credentials=True, resources={r"/*": {"origins": "*"}})

class MakePrediction(Resource):
    @staticmethod
    def post():
        posted_data = request.get_json()
        sepal_length = posted_data['sepal_length']
        sepal_width = posted_data['sepal_width']
        petal_length = posted_data['petal_length']
        petal_width = posted_data['petal_width']
        token = posted_data['token']

        model = joblib.load(token+'.model')

        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
        if prediction == 0:
            predicted_class = 'Iris-setosa'
        elif prediction == 1:
            predicted_class = 'Iris-versicolor'
        else:
            predicted_class = 'Iris-virginica'

        return jsonify({
            'Prediction': predicted_class
        })

class FileUpload(Resource):
    @staticmethod
    def post():
        posted_data = request.get_json()
        train_file_url = posted_data['train_file_url']
        model_file_name = posted_data['model_file_name']
        token = train_file_url.split("token=")[1]

        # Load in memory the file from `url`
        response = urllib.request.urlopen(train_file_url)
        data = response.read()      # a `bytes` object
        text = data.decode('utf-8') # a `str`; this step can't be used if data is binary

        # Replace model file name
        text = text.replace(model_file_name,token+'.model')
        
        # Save training file locally under token as file name
        file_path = token + '.py'
        file = open(file_path, 'w')
        file.write(text)
        file.close()
        
        # Load training module dynamically
        spec = importlib.util.spec_from_file_location(token, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Train model
        module.train_model()

        return jsonify({
            'url': 'https://gallifreylabs-hartnell.herokuapp.com/predict',
            'status': 'success',
            'token': token
        })

# Declare a route, the part of URL which will be used to handle requests:
api.add_resource(MakePrediction, '/predict')
api.add_resource(FileUpload, '/file_upload')

# Run the app in debug mode:
if __name__ == '__main__':
    app.run(debug=True)
