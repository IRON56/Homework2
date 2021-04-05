import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.layers import LSTM
from keras.wrappers.scikit_learn import KerasRegressor
import statsmodels.api as sm
from pickle import dump
from pickle import load
import pandas as pd
import numpy as np
import pandas_datareader as pdr

def strategy (size, md):
    df = pdr.DataReader('IVV', data_source='yahoo', start='2020-04-01')
    bond_10 = pdr.DataReader('^TNX',data_source='yahoo', start='2020-04-01')
    bond_30 = pdr.DataReader('^TYX',data_source='yahoo', start='2020-04-01')
    bond_5 = pdr.DataReader('^FVX',data_source='yahoo', start='2020-04-01')
    final = len(df)
    eia = pdr.DataReader('EIA.F',data_source='yahoo',start='2020-04-01')
    df['IVV'] = (df['Close'] - df['Open']) / df['Open']
    df['mvg30'] = df['IVV'].rolling(window=size).mean()
    df['std30'] = df['IVV'].rolling(window=size).std()
    bond_10['bond10'] = (bond_10['Close'] - bond_10['Open']) / bond_10['Open']
    bond_5['bond5'] = (bond_5['Close'] - bond_5['Open']) / bond_5['Open']
    bond_30['bond30'] = (bond_30['Close'] - bond_30['Open']) / bond_30['Open']
    ivv_response = []
    for i in range(1, len(df)):
        ivv_response.append(int(df["IVV"][i] > 0))
    ivv_response.append(np.nan)
    df['Response'] = ivv_response
    eia_return = []
    eia_return.append(np.nan)
    for i in range(1, len(eia)):
        eia_return.append((eia['Close'][i] - eia['Close'][i - 1]) / eia['Close'][i - 1])
    eia['eia'] = eia_return
    kp_df = ['Date', 'IVV', 'mvg30', 'std30', 'Response','Close','Open']
    dp_df = [col for col in df.columns if col not in kp_df]
    df.drop(labels=dp_df, axis=1, inplace=True)
    kp_5 = ['Date', 'bond5']
    dp_5 = [col for col in bond_5.columns if col not in kp_5]
    bond_5.drop(labels=dp_5, axis=1, inplace=True)
    kp_10 = ['Date', 'bond10']
    dp_10 = [col for col in bond_10.columns if col not in kp_10]
    bond_10.drop(labels=dp_10, axis=1, inplace=True)
    kp_30 = ['Date', 'bond30']
    dp_30 = [col for col in bond_30.columns if col not in kp_30]
    bond_30.drop(labels=dp_30, axis=1, inplace=True)
    kp_e = ['Date', 'eia']
    dp_e = [col for col in eia.columns if col not in kp_e]
    eia.drop(labels=dp_e, axis=1, inplace=True)
    model_data = df.join(bond_5, how='outer')
    model_data = model_data.join(bond_10, how='outer')
    model_data = model_data.join(bond_30, how='outer')
    model_data = model_data.join(eia, how='outer')
    test_data = model_data[-1:]
    model_data = model_data.dropna(axis=0)
    Y = model_data['Response']
    X = model_data.loc[:, model_data.columns!='Response']
    validation_size = 0.2
    seed = 3
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
    if md =='Decision Tree':
        model = DecisionTreeClassifier()
    if md =='Loglinear':
        model = LogisticRegression()
    if md =='KNN':
        model = KNeighborsClassifier()
    model.fit(X_train, Y_train)
    filename = 'finalized_model.sav'
    dump(model, open(filename, 'wb'))
    return model_data, test_data


def backtest (model_data,test):
    filename = 'finalized_model.sav'
    loaded_model = load(open(filename, 'rb'))
    test_data = model_data[-31:]
    Result = loaded_model.predict(test_data.loc[:, test_data.columns != 'Response'])
    amt = 0
    ror = []
    df = pd.DataFrame()
    df['Date'] = test_data.index[1:]
    df['ID'] = test_data['Close'][1:]
    df['Type'] = test_data['Close'][1:]
    df['actn'] = test_data['Close'][1:]
    df['Price'] = test_data['Close'][1:]
    df['size'] = test_data['Close'][1:]
    df['symb'] = test_data['Close'][1:]
    count = 1
    last = 0.0
    cost = 0
    revenue = 0.0
    for i in range(1, 31):
        df['ID'][i-1] = count
        if Result[i - 1] > 0.5:
            df['Type'][i-1] = 'MKT'
            df['actn'][i-1] = 'BUY'
            df['symb'][i-1] = 'IVV'
            df['size'][i-1] = 100
            df['Price'][i-1] = test_data['Open'][i].round(2)
            amt = amt + 1
            cost = cost + 100*test_data['Open'][i]
            revenue = revenue + amt * 100 * test_data['Close'][i] / cost - 1
            ror.append(revenue.round(4))
            last = revenue
            revenue = 0.0
        else:
            df['Type'][i-1] = 'LMT'
            df['actn'][i-1] = 'SELL'
            df['symb'][i-1] = 'IVV'
            df['size'][i-1] = amt*100
            df['Price'][i-1] = test_data['Close'][i-1].round(2)
            ror.append(round(last,4))
            amt = 0
            revenue = last
            cost = 0.0
        count = count +1
    test['Response'] =  loaded_model.predict(test.loc[:, test.columns != 'Response'])
    test_data[:]['actn'] = 'BUY'
    df["Rate of Return(%)"] = ror
    return df, test
