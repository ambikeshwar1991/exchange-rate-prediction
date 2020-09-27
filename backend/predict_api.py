
import os
import joblib

from flask import Flask, jsonify, request
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)

model = joblib.load('../fx_predict_model.model')

class MakePrediction(Resource):
    @staticmethod
    def post():
        # posted_data = request.get_json()
        # print(posted_data)
        # ird = posted_data['ird']
        # cpi_diff = posted_data['cpi_diff']
        # gdp_diff = posted_data['gdp_diff']

        prediction = model.predict([[1, 1, 1]])[0]

        return jsonify({
            'Prediction': prediction
        })


api.add_resource(MakePrediction, '/predict')


if __name__ == '__main__':
    app.run(debug=True)

