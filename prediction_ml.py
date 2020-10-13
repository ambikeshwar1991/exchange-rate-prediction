'''
@author : Ambikeshwar Srivastava
ML model for foreign exchange prediction
'''

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
import joblib

def getFxRatesForPairs(pairName):
    df = pd.read_csv("data_source/fx_rates_aud-USD.csv")
    df = df.replace('ND', np.nan)
    df = df.dropna().reset_index(drop=True)
    df.isna().sum()
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Time Series'] = pd.to_datetime(df['Time Series'])
    df['month'] = df['Time Series'].dt.month
    df['year'] = df['Time Series'].dt.year
    df['month_year'] = df['Time Series'].dt.to_period('M')
    return df.groupby('month_year').AUD_USD.mean().reset_index()

def getIrdData(pairName):
    ir_df = pd.read_csv("data_source/aud-usd-ird.csv")
    ir_df = ir_df[(ir_df['Date'] >= '2016-03-01') &
                  (ir_df['Date'] <= '2020-04-02')]
    ir_df = ir_df['Long Carry'].astype(str)
    ir_df.reindex(index=ir_df.index[::-1])
    ir_df = ir_df.replace({'%': ''}, regex=True)
    ir_df = ir_df.astype(float)

    return np.array(ir_df).reshape(-1, 1)

def getGdpDiff(pairName):
    aus_gdp = pd.read_csv("data_source/aus-gdp-rate.csv")
    usa_gdp = pd.read_csv("data_source/usd-gdp-rate.csv")

    aus_gdp['DATE'] = pd.to_datetime(aus_gdp['DATE']).dt.to_period('M')
    aus_gdp = aus_gdp.set_index('DATE').resample('M').interpolate()
    aus_gdp['month_year'] = aus_gdp.index

    usa_gdp['DATE'] = pd.to_datetime(usa_gdp['DATE']).dt.to_period('M')
    usa_gdp = usa_gdp.set_index('DATE').resample('M').interpolate()
    usa_gdp['month_year'] = usa_gdp.index

    aus_gdp = aus_gdp.rename(columns={'GDP': 'AUS_GDP'})
    aus_usa_gdp = pd.merge(aus_gdp, usa_gdp, on="month_year", how="inner")
    aus_usa_gdp = aus_usa_gdp.rename(columns={'GDP': 'USA_GDP'})
    aus_usa_gdp['GDP_diff'] = aus_usa_gdp['AUS_GDP'] - aus_usa_gdp['USA_GDP']

    aus_usa_gdp = aus_usa_gdp[(aus_usa_gdp['month_year'] >= '2016-03') &
                  (aus_usa_gdp['month_year'] <= '2020-04')].reset_index(drop=True)

    gdp_diff = ["%.4f" % num for num in aus_usa_gdp['GDP_diff']]
    return gdp_diff

def getCPIDiff(pairName):
    aus_cpi = pd.read_csv("data_source/AUS-CPI.csv")
    usa_cpi = pd.read_csv("data_source/USA-CPI.csv")

    aus_cpi['DATE'] = pd.to_datetime(aus_cpi['DATE']).dt.to_period('M')
    aus_cpi = aus_cpi.set_index('DATE').resample('M').interpolate()
    aus_cpi['month_year'] = aus_cpi.index
    aus_cpi = aus_cpi[['month_year', 'AUS_CPI']]

    usa_cpi['DATE'] = pd.to_datetime(usa_cpi['DATE']).dt.to_period('M')
    usa_cpi = usa_cpi.set_index('DATE').resample('M').interpolate()
    usa_cpi['month_year'] = usa_cpi.index
    usa_cpi = usa_cpi[['month_year', 'USA_CPI']]

    aus_usa_cpi = pd.merge(aus_cpi, usa_cpi, on="month_year", how="inner")
    aus_usa_cpi = aus_usa_cpi.rename(columns={'CPI': 'USA_CPI'})
    aus_usa_cpi['CPI_diff'] = aus_usa_cpi['AUS_CPI'] - aus_usa_cpi['USA_CPI']

    aus_usa_cpi = aus_usa_cpi[(aus_usa_cpi['month_year'] >= '2016-03') &
                              (aus_usa_cpi['month_year'] <= '2020-04')].reset_index(drop=True)

    cpi_diff = ["%.4f" % num for num in aus_usa_cpi['CPI_diff']]
    return cpi_diff

def createMultiLinearModel():
    pairName = "AUD-USD"
    x_ir = getIrdData(pairName)
    gdp_diff = getGdpDiff(pairName)
    cpi_diff = getCPIDiff(pairName)

    x_ir_gdp_cpi = np.array(list(zip(x_ir, gdp_diff, cpi_diff)))
    x_ir_gdp_cpi = x_ir_gdp_cpi.astype(np.float)

    audUsdFxRates = getFxRatesForPairs(pairName)
    y_fx = audUsdFxRates[(audUsdFxRates['month_year'] >= '2016-03') &
                                  (audUsdFxRates['month_year'] <='2020-04')].reset_index(drop=True)
    y_fx['CPI_Diff'] = cpi_diff
    y_fx['GDP_Diff'] = gdp_diff
    y_fx['IRD'] = x_ir
    y_fx = y_fx['AUD_USD']

    model = LinearRegression()
    model = model.fit(x_ir_gdp_cpi, y_fx)
    y_fx_predict_4 = model.predict(x_ir_gdp_cpi)
    print(model.score(x_ir_gdp_cpi, y_fx))
    joblib.dump(model, 'fx_predict_model.model')
    cpi_diff = [{i: float(j)} for i, j in zip(y_fx['month_year'], cpi_diff)]
    gdp_diff = [{i: float(j)} for i, j in zip(y_fx['month_year'], gdp_diff)]
    x_ir  = [{i: float(j)} for i, j in zip(y_fx['month_year'], x_ir)]
    y_fx = [{i: float(j)} for i, j in zip(y_fx['month_year'], y_fx)]
    y_fx_predict_4 = [{i: float(j)} for i, j in zip(y_fx['month_year'], y_fx_predict_4)]
    joblib.dump({'cpi_diff': cpi_diff, 'gdp_diff': gdp_diff, 'x_ir': x_ir, 'y_fx': y_fx,
                 'y_fx_predict_4': y_fx_predict_4}
                , 'array.dump')
if __name__ == '__main__':
    createMultiLinearModel()
