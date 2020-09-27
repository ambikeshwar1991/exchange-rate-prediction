
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
        posted_data = request.get_json()
        ird = float(posted_data["ird"])
        cpi_diff = float(posted_data["cpi_diff"])
        gdp_diff = float(posted_data["gdp_diff"])

        prediction = model.predict([[ird, cpi_diff, gdp_diff]])[0]

        return jsonify({
            'Prediction': prediction
        })


api.add_resource(MakePrediction, '/predict')


if __name__ == '__main__':
    app.run(debug=True)

