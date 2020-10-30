import os
import joblib
import json
import datetime as dt

from flask_cors import CORS, cross_origin
from flask import Flask, jsonify, request
from flask_restful import Api, Resource

app = Flask(__name__)
CORS(app)
api = Api(app)

eurJpy_plot = joblib.load('../EUR_JPY_plot_data.model')
eurJpy_pred = joblib.load('../EUR_JPY_predict_model.model')
eurJpyCpiInfl = joblib.load('../EUR_JPY_cpi_inflation_model.model')
eurJpyCpi = joblib.load('../EUR_JPY_cpi_model.model')
eurJpyFdi = joblib.load('../EUR_JPY_fdi_model.model')
eurJpyGdp = joblib.load('../EUR_JPY_gdp_model.model')
eurJpyIrd = joblib.load('../EUR_JPY_ird_model.model')

eurUsd_plot = joblib.load('../EUR_USD_plot_data.model')
eurUsd_pred = joblib.load('../EUR_USD_predict_model.model')
eurUsdCpiInfl = joblib.load('../EUR_USD_cpi_inflation_model.model')
eurUsdCpi = joblib.load('../EUR_USD_cpi_model.model')
eurUsdFdi = joblib.load('../EUR_USD_fdi_model.model')
eurUsdGdp = joblib.load('../EUR_USD_gdp_model.model')
eurUsdIrd = joblib.load('../EUR_USD_ird_model.model')

eurAud_plot = joblib.load('../EUR_AUD_plot_data.model')
eurAud_pred = joblib.load('../EUR_AUD_predict_model.model')
eurAudCpiInfl = joblib.load('../EUR_AUD_cpi_inflation_model.model')
eurAudCpi = joblib.load('../EUR_AUD_cpi_model.model')
eurAudFdi = joblib.load('../EUR_AUD_fdi_model.model')
eurAudGdp = joblib.load('../EUR_AUD_gdp_model.model')
eurAudIrd = joblib.load('../EUR_AUD_ird_model.model')

eurGbp_plot = joblib.load('../EUR_GBP_plot_data.model')
eurGbp_pred = joblib.load('../EUR_GBP_predict_model.model')
eurGbpCpiInfl = joblib.load('../EUR_GBP_cpi_inflation_model.model')
eurGbpCpi = joblib.load('../EUR_GBP_cpi_model.model')
eurGbpFdi = joblib.load('../EUR_GBP_fdi_model.model')
eurGbpGdp = joblib.load('../EUR_GBP_gdp_model.model')
eurGbpIrd = joblib.load('../EUR_GBP_ird_model.model')

gbpJpy_plot = joblib.load('../GBP_JPY_plot_data.model')
gbpJpy_pred = joblib.load('../GBP_JPY_predict_model.model')
gbpJpyCpiInfl = joblib.load('../GBP_JPY_cpi_inflation_model.model')
gbpJpyCpi = joblib.load('../GBP_JPY_cpi_model.model')
gbpJpyFdi = joblib.load('../GBP_JPY_fdi_model.model')
gbpJpyGdp = joblib.load('../GBP_JPY_gdp_model.model')
gbpJpyIrd = joblib.load('../GBP_JPY_ird_model.model')

gbpUsd_plot = joblib.load('../GBP_USD_plot_data.model')
gbpUsd_pred = joblib.load('../GBP_USD_predict_model.model')
gbpUsdCpiInfl = joblib.load('../GBP_USD_cpi_inflation_model.model')
gbpUsdCpi = joblib.load('../GBP_USD_cpi_model.model')
gbpUsdFdi = joblib.load('../GBP_USD_fdi_model.model')
gbpUsdGdp = joblib.load('../GBP_USD_gdp_model.model')
gbpUsdIrd = joblib.load('../GBP_USD_ird_model.model')

gbpEur_plot = joblib.load('../GBP_EUR_plot_data.model')
gbpEur_pred = joblib.load('../GBP_EUR_predict_model.model')
gbpEurCpiInfl = joblib.load('../GBP_EUR_cpi_inflation_model.model')
gbpEurCpi = joblib.load('../GBP_EUR_cpi_model.model')
gbpEurFdi = joblib.load('../GBP_EUR_fdi_model.model')
gbpEurGdp = joblib.load('../GBP_EUR_gdp_model.model')
gbpEurIrd = joblib.load('../GBP_EUR_ird_model.model')

gbpAud_plot = joblib.load('../GBP_AUD_plot_data.model')
gbpAud_pred = joblib.load('../GBP_AUD_predict_model.model')
gbpAudCpiInfl = joblib.load('../GBP_AUD_cpi_inflation_model.model')
gbpAudCpi = joblib.load('../GBP_AUD_cpi_model.model')
gbpAudFdi = joblib.load('../GBP_AUD_fdi_model.model')
gbpAudGdp = joblib.load('../GBP_AUD_gdp_model.model')
gbpAudIrd = joblib.load('../GBP_AUD_ird_model.model')

jpyAud_plot = joblib.load('../JPY_AUD_plot_data.model')
jpyAud_pred = joblib.load('../JPY_AUD_predict_model.model')
jpyAudCpiInfl = joblib.load('../JPY_AUD_cpi_inflation_model.model')
jpyAudCpi = joblib.load('../JPY_AUD_cpi_model.model')
jpyAudFdi = joblib.load('../JPY_AUD_fdi_model.model')
jpyAudGdp = joblib.load('../JPY_AUD_gdp_model.model')
jpyAudIrd = joblib.load('../JPY_AUD_ird_model.model')

jpyGbp_plot = joblib.load('../JPY_GBP_plot_data.model')
jpyGbp_pred = joblib.load('../JPY_GBP_predict_model.model')
jpyGbpCpiInfl = joblib.load('../JPY_GBP_cpi_inflation_model.model')
jpyGbpCpi = joblib.load('../JPY_GBP_cpi_model.model')
jpyGbpFdi = joblib.load('../JPY_GBP_fdi_model.model')
jpyGbpGdp = joblib.load('../JPY_GBP_gdp_model.model')
jpyGbpIrd = joblib.load('../JPY_GBP_ird_model.model')


jpyUsd_plot = joblib.load('../JPY_USD_plot_data.model')
jpyUsd_pred = joblib.load('../JPY_USD_predict_model.model')
jpyUsdCpiInfl = joblib.load('../JPY_USD_cpi_inflation_model.model')
jpyUsdCpi = joblib.load('../JPY_USD_cpi_model.model')
jpyUsdFdi = joblib.load('../JPY_USD_fdi_model.model')
jpyUsdGdp = joblib.load('../JPY_USD_gdp_model.model')
jpyUsdIrd = joblib.load('../JPY_USD_ird_model.model')


jpyEur_plot = joblib.load('../JPY_EUR_plot_data.model')
jpyEur_pred = joblib.load('../JPY_EUR_predict_model.model')
jpyEurCpiInfl = joblib.load('../JPY_EUR_cpi_inflation_model.model')
jpyEurCpi = joblib.load('../JPY_EUR_cpi_model.model')
jpyEurFdi = joblib.load('../JPY_EUR_fdi_model.model')
jpyEurGdp = joblib.load('../JPY_EUR_gdp_model.model')
jpyEurIrd = joblib.load('../JPY_EUR_ird_model.model')


usdAud_plot = joblib.load('../USD_AUD_plot_data.model')
usdAud_pred = joblib.load('../USD_AUD_predict_model.model')
usdAudCpiInfl = joblib.load('../JPY_AUD_cpi_inflation_model.model')
usdAudCpi = joblib.load('../JPY_AUD_cpi_model.model')
usdAudFdi = joblib.load('../JPY_AUD_fdi_model.model')
usdAudGdp = joblib.load('../JPY_AUD_gdp_model.model')
usdAudIrd = joblib.load('../JPY_AUD_ird_model.model')

usdGbp_plot = joblib.load('../USD_GBP_plot_data.model')
usdGbp_pred = joblib.load('../USD_GBP_predict_model.model')
usdGbpCpiInfl = joblib.load('../USD_AUD_cpi_inflation_model.model')
usdGbpCpi = joblib.load('../USD_AUD_cpi_model.model')
usdGbpFdi = joblib.load('../USD_AUD_fdi_model.model')
usdGbpGdp = joblib.load('../USD_AUD_gdp_model.model')
usdGbpIrd = joblib.load('../USD_AUD_ird_model.model')

usdJpy_plot = joblib.load('../USD_JPY_plot_data.model')
usdJpy_pred = joblib.load('../USD_JPY_predict_model.model')
usdJpyCpiInfl = joblib.load('../USD_JPY_cpi_inflation_model.model')
usdJpyCpi = joblib.load('../USD_JPY_cpi_model.model')
usdJpyFdi = joblib.load('../USD_JPY_fdi_model.model')
usdJpyGdp = joblib.load('../USD_JPY_gdp_model.model')
usdJpyIrd = joblib.load('../USD_JPY_ird_model.model')

usdEur_plot = joblib.load('../USD_EUR_plot_data.model')
usdEur_pred = joblib.load('../USD_EUR_predict_model.model')
usdEurCpiInfl = joblib.load('../USD_EUR_cpi_inflation_model.model')
usdEurCpi = joblib.load('../USD_EUR_cpi_model.model')
usdEurFdi = joblib.load('../USD_EUR_fdi_model.model')
usdEurGdp = joblib.load('../USD_EUR_gdp_model.model')
usdEurIrd = joblib.load('../USD_EUR_ird_model.model')

# {currencyPair: "USD/EUR", gdpDifference: 0, cpiDifference: 0, cpiInflationDifferential: 0, interestRateDifferential: 0, …}
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class MakePrediction(Resource):
    @staticmethod
    def post():
        posted_data = request.get_json()
        print(posted_data)
        ird = round(float(posted_data["interestRateDifferential"])) #interestRateDifferential
        cpi_diff = round(float(posted_data["cpiDifference"]))
        gdp_diff = round(float(posted_data["gdpDifference"]))
        cpi_inflation_diff = round(float(posted_data["cpiInflationDifferential"]))
        fdi_diff = round(float(posted_data["fdiDifference"]))
        if posted_data["currencyPair"] == 'EUR/JPY':
            model = eurJpy_pred
            plot_array = eurJpy_plot
            cpiInf = eurJpyCpiInfl
            cpi = eurJpyCpi
            fdi = eurJpyFdi
            gdp = eurJpyGdp
            irdm = eurJpyIrd
        elif posted_data["currencyPair"] == 'EUR/USD':
            model = eurUsd_plot
            plot_array = eurUsd_plot
            cpiInf = eurUsdCpiInfl
            cpi = eurUsdCpi
            fdi = eurUsdFdi
            gdp = eurUsdGdp
            irdm = eurUsdIrd
        elif posted_data["currencyPair"] == 'EUR/AUD':
            model = eurAud_plot
            plot_array = eurAud_plot
            cpiInf = eurUsdCpiInfl
            cpi = eurAudCpi
            fdi = eurAudFdi
            gdp = eurAudGdp
            irdm = eurAudIrd
        elif posted_data["currencyPair"] == 'EUR/GBP':
            model = eurGbp_pred
            plot_array = eurGbp_plot
            cpiInf = eurGbpCpiInfl
            cpi = eurGbpCpi
            fdi = eurGbpFdi
            gdp = eurGbpGdp
            irdm = eurGbpIrd
        elif posted_data["currencyPair"] == 'GBP/JPY':
            model = gbpJpy_pred
            plot_array = gbpJpy_plot
            cpiInf = gbpJpyCpiInfl
            cpi = gbpJpyCpi
            fdi = gbpJpyFdi
            gdp = gbpJpyGdp
            irdm = gbpJpyIrd
        elif posted_data["currencyPair"] == 'GBP/USD':
            model = gbpUsd_pred
            plot_array = gbpUsd_plot
            cpiInf = gbpUsdCpiInfl
            cpi = gbpUsdCpi
            fdi = gbpUsdFdi
            gdp = gbpUsdGdp
            irdm = gbpUsdIrd
        elif posted_data["currencyPair"] == 'GBP/EUR':
            model = gbpEur_pred
            plot_array = gbpEur_plot
            cpiInf = gbpEurCpiInfl
            cpi = gbpEurCpi
            fdi = gbpEurFdi
            gdp = gbpEurGdp
            irdm = gbpEurIrd
        elif posted_data["currencyPair"] == 'GBP/AUD':
            model = gbpAud_pred
            plot_array = gbpAud_plot
            cpiInf = gbpAudCpiInfl
            cpi = gbpAudCpi
            fdi = gbpAudFdi
            gdp = gbpAudGdp
            irdm = gbpAudIrd
        elif posted_data["currencyPair"] == 'USD/EUR':
            model = usdEur_pred
            plot_array =usdEur_plot
            cpiInf = usdEurCpiInfl
            cpi = usdEurCpi
            fdi = usdEurFdi
            gdp = usdEurGdp
            irdm = usdEurIrd
        elif posted_data["currencyPair"] == 'USD/AUD':
            model = usdAud_pred
            plot_array = usdAud_plot
            cpiInf = usdAudCpiInfl
            cpi = usdAudCpi
            fdi = usdAudFdi
            gdp = usdAudGdp
            irdm = usdAudIrd
        elif posted_data["currencyPair"] == 'USD/JPY':
            model = usdJpy_pred
            plot_array = usdJpy_plot
            cpiInf = usdJpyCpiInfl
            cpi = usdJpyCpi
            fdi = usdJpyFdi
            gdp = usdJpyGdp
            irdm = usdJpyIrd
        elif posted_data["currencyPair"] == 'USD/GBP':
            model = usdGbp_pred
            plot_array = usdGbp_plot
            cpiInf = usdGbpCpiInfl
            cpi = usdGbpCpi
            fdi = usdGbpFdi
            gdp = usdGbpGdp
            irdm = usdGbpIrd
        elif posted_data["currencyPair"] == 'JPY/EUR':
            model = jpyEur_pred
            plot_array = jpyEur_plot
            cpiInf = jpyEurCpiInfl
            cpi = jpyEurCpi
            fdi = jpyEurFdi
            gdp = jpyEurGdp
            irdm = jpyEurIrd
        elif posted_data["currencyPair"] == 'JPY/AUD':
            model = jpyAud_pred
            plot_array = jpyAud_plot
            cpiInf = jpyAudCpiInfl
            cpi = jpyAudCpi
            fdi = jpyAudFdi
            gdp = jpyAudGdp
            irdm = jpyAudIrd
        elif posted_data["currencyPair"] == 'JPY/USD':
            model = jpyUsd_pred
            plot_array = jpyUsd_plot
            cpiInf = jpyUsdCpiInfl
            cpi = jpyUsdCpi
            fdi = jpyUsdFdi
            gdp = jpyUsdGdp
            irdm = jpyUsdIrd
        elif posted_data["currencyPair"] == 'JPY/GBP':
            model = jpyGbp_pred
            plot_array = jpyGbp_plot
            cpiInf = jpyGbpCpiInfl
            cpi = jpyGbpCpi
            fdi = jpyGbpFdi
            gdp = jpyGbpGdp
            irdm = jpyGbpIrd
        prediction = model.predict([[gdp_diff, cpi_diff, ird, cpi_inflation_diff, fdi_diff]])[0]
        import pandas as pd
        from collections import defaultdict
        res = defaultdict(list)
        for sub in plot_array['predictedValue']:
            for key in sub:
                res[key].append(sub[key])
        # df = pd.DataFrame(res).T
        # df = df.reindex(df.index.values.tolist() + ['2021-01', '2021-02', '2021-03'])
        # df = df.interpolate(method='linear', order=2)
        # dct = df.to_dict()
        # plot_array['predictedValue'] = [{key: dct[0][key]} for key in dct[0].keys()]

        res = defaultdict(list)
        for sub in plot_array['actualValue']:
            for key in sub:
                res[key].append(sub[key])
        df = pd.DataFrame(res).T
        df = df.drop('2020-12')
        df = df.drop('2020-11')
        df = df.drop('2020-10')
        dct = df.to_dict()
        plot_array['actualValue'] = [{key: dct[0][key]} for key in dct[0].keys()]

        predArr = ['2020-10-01', '2020-11-01', '2020-12-01', '2021-01-01', '2021-02-01', '2021-03-01']

        for i in predArr:
            d = dt.datetime.strptime(i, '%Y-%m-%d').date()
            d = d.toordinal()
            plot_array['predictedValue'].append({i[:-3]: model.predict([[gdp.predict(d)[0], cpi.predict(d)[0],
                                                                        irdm.predict(d)[0], cpiInf.predict(d)[0],
                                                                        fdi.predict(d)[0]]])[0]})

        return jsonify({
            'Prediction': float(prediction)
            ,
            'plot_array': plot_array
        })

api.add_resource(MakePrediction, '/predict')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
