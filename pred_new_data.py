
import joblib

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#pwd:P@ssw0rd
def getDataFromCsv():
    df = pd.read_csv('data_source/prediction_tool_data.csv')
    df = df.interpolate(method='linear', axis=0).ffill().bfill()
    return df

def getPairFromdf(df, pairName):
    firstCurr = pairName.split('/')[0]
    secondCurr = pairName.split('/')[1]

    dfFirstCurr = df[df['Code'] == firstCurr].reset_index(drop=True)
    dfSecondCurr = df[df['Code'] == secondCurr].reset_index(drop=True)

    dfFinal = pd.DataFrame()

    dfFinal['Date'] = pd.to_datetime(dfFirstCurr[['Year', 'Month']].assign(DAY=1))
    dfFinal['Date'] = dfFinal['Date'].dt.to_period('M')
    dfFinal['GDP Diff'] = dfFirstCurr['GDP billion currency units'] - \
                                            dfSecondCurr['GDP billion currency units']
    dfFinal['CPI Diff'] = dfFirstCurr['Consumer Price Index (CPI)'] - \
                                            dfSecondCurr['Consumer Price Index (CPI)']
    dfFinal['CPI Inflation Diff'] = dfFirstCurr['Inflation monthly percent change in the CPI'] -\
                                                             dfSecondCurr['Inflation monthly percent change in the CPI']
  
    dfFinal['IRD'] = dfFirstCurr['Interest rates'] - dfSecondCurr['Interest rates']

    dfFinal['FDI Diff'] = dfFirstCurr['Foreign direct investment million currency units'] - \
                          dfSecondCurr['Foreign direct investment million currency units']

    dfFinal['Exchange rate '+str(firstCurr)] = dfSecondCurr['Exchange rate '+str(firstCurr)]
    dfFinal['GDP '+str(firstCurr)] = dfFirstCurr['GDP billion currency units']
    dfFinal['GDP '+str(secondCurr)] = dfSecondCurr['GDP billion currency units']
    dfFinal['CPI '+str(firstCurr)] = dfFirstCurr['Consumer Price Index (CPI)']
    dfFinal['CPI ' + str(secondCurr)] = dfSecondCurr['Consumer Price Index (CPI)']
    dfFinal['CPI Inflation '+str(firstCurr)] = dfFirstCurr['Inflation monthly percent change in the CPI']
    dfFinal['CPI Inflation ' + str(secondCurr)] = dfSecondCurr['Inflation monthly percent change in the CPI']
    dfFinal['Interest Rate '+str(firstCurr)] = dfFirstCurr['Interest rates']
    dfFinal['Interest Rate '+str(secondCurr)] = dfSecondCurr['Interest rates']
    dfFinal['FDI '+str(firstCurr)] = dfFirstCurr['Foreign direct investment million currency units']
    dfFinal['FDI ' + str(firstCurr)] = dfFirstCurr['Foreign direct investment million currency units']

    return firstCurr, secondCurr, dfFinal
global dct
dct=dict()
def createLinearMLmodel(df, pair):
    firstCurr, secondCurr, dfFinal = getPairFromdf(df, pair)
    gdpDiff = np.array(list(dfFinal['GDP Diff'])).reshape(-1, 1)
    cpiDiff = np.array(list(dfFinal['CPI Diff'])).reshape(-1, 1)
    ird = np.array(list(dfFinal['IRD'])).reshape(-1,1)
    cpiInflationDiff = np.array(list(dfFinal['CPI Inflation Diff'])).reshape(-1,1)
    fdiDiff = np.array(list(dfFinal['FDI Diff'])).reshape(-1,1)
    xAxis = np.array(list(zip(gdpDiff, cpiDiff, ird, cpiInflationDiff, fdiDiff)))
    xAxis = xAxis.astype(np.float).reshape(-1,5)
#     print('hhhh=', xAxis.shape, xAxis)
        
    yFx = np.array(list(dfFinal['Exchange rate '+str(firstCurr)])).reshape(-1,1)
    model = LinearRegression()
#     print('hema=',len(yFx),yFx.shape, yFx)
#     print('milind=',len(xAxis),xAxis.shape, xAxis)
    model = model.fit(xAxis, yFx)
    y = model.predict(xAxis)
    joblib.dump(model, str(firstCurr)+'_'+str(secondCurr)+'_predict_model.model')

    gdpDiff = [{i: float(j)} for i, j in zip(dfFinal['Date'].astype(str), dfFinal['GDP Diff'])]
    gdpFirstCurr = [{i: float(j)} for i, j in zip(dfFinal['Date'].astype(str), dfFinal['GDP '+str(firstCurr)])]
    gdpSecondCurr = [{i: float(j)} for i, j in zip(dfFinal['Date'].astype(str), dfFinal['GDP '+str(secondCurr)])]
    
    cpiDiff = [{i: float(j)} for i, j in zip(dfFinal['Date'].astype(str), dfFinal['CPI Diff'])]
    cpiFirstCurr = [{i: float(j)} for i, j in zip(dfFinal['Date'].astype(str), dfFinal['CPI ' + str(firstCurr)])]
    cpiSecondCurr = [{i: float(j)} for i, j in zip(dfFinal['Date'].astype(str), dfFinal['CPI ' + str(secondCurr)])]
    
    ird = [{i: float(j)} for i, j in zip(dfFinal['Date'].astype(str), dfFinal['IRD'])]
    irFirstCurr = [{i: float(j)} for i, j in zip(dfFinal['Date'].astype(str), dfFinal['Interest Rate '+ str(firstCurr)])]
    irSecondCurr = [{i: float(j)} for i, j in zip(dfFinal['Date'].astype(str), dfFinal['Interest Rate ' + str(secondCurr)])]

    cpiInflationDiff = [{i: float(j)} for i, j in zip(dfFinal['Date'].astype(str), dfFinal['CPI Inflation Diff'])]
    cpiInflationFirst = [{i: float(j)} for i, j in
                         zip(dfFinal['Date'].astype(str), dfFinal['CPI Inflation ' + str(firstCurr)])]
    cpiInflationSecond = [{i: float(j)} for i, j in
                         zip(dfFinal['Date'].astype(str), dfFinal['CPI Inflation ' + str(secondCurr)])]

    fdiDiff = [{i: float(j)} for i, j in zip(dfFinal['Date'].astype(str), dfFinal['FDI Diff'])]
    fdiFirstCurr = [{i: float(j)} for i, j in zip(dfFinal['Date'].astype(str), dfFinal['FDI '+str(firstCurr)])]
    fdiSecondCurr = [{i: float(j)} for i, j in zip(dfFinal['Date'].astype(str), dfFinal['FDI '+str(firstCurr)])]
    yy = []
    for i in y:
        yy.append(i[0])
    dct = {'gdpDiff': gdpDiff, 'cpiDiff':cpiDiff, 'ird': ird, 'cpiInflationDiff':cpiInflationDiff, 'fdiDiff':fdiDiff,
           'predictedValue': yy, 'GDP '+str(firstCurr): gdpFirstCurr, 'GDP '+str(secondCurr): gdpSecondCurr,
           'CPI '+str(secondCurr): cpiFirstCurr, 'CPI '+str(secondCurr): cpiSecondCurr, 'IRD '+ str(firstCurr):irFirstCurr,
           'IRD ' + str(secondCurr): irSecondCurr, 'CPI Inflation ' + str(firstCurr): cpiInflationFirst,
           'CPI Inflation ' + str(secondCurr): cpiInflationSecond, 'FDI '+str(firstCurr): fdiFirstCurr,
           'FDI '+str(secondCurr): fdiSecondCurr}
    
    joblib.dump(dct, str(firstCurr)+'_'+str(secondCurr)+'_plot_data.model')

if __name__ == '__main__':
    df = getDataFromCsv()
    createLinearMLmodel(df, 'JPY/AUD')
