
import os
import joblib
from flask_cors import CORS, cross_origin
from flask import Flask, jsonify, request
from flask_restful import Api, Resource

app = Flask(__name__)
CORS(app)
api = Api(app)

model = joblib.load('../fx_predict_model.model')

class MakePrediction(Resource):
    @staticmethod
    def post():
        posted_data = request.get_json()
        print(posted_data)
        ird = round(float(posted_data["interestRateDifferential"])) #interestRateDifferential
        cpi_diff = round(float(posted_data["cpiDifference"]))
        gdp_diff = round(float(posted_data["gdpDifference"]))

        prediction = model.predict([[ird, cpi_diff, gdp_diff]])[0]
        #score_m = model.score([[ird, cpi_diff, gdp_diff]])
        return jsonify({
            'Prediction': prediction,
        })


api.add_resource(MakePrediction, '/predict')


if __name__ == '__main__':
    app.run(host='0.0.0.0')

