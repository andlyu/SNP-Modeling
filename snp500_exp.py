#Download SNP500 data from 1973 to 2023
#use Yahoo Finance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import yfinance as yf
import pandas_datareader.data as web


pred_preiod = 20
dist = 10

for pred_preiod in [7,30,60,100]:
    for dist in [0,3,7,15]:
        # Download data for the S&P 500 index from 2020 to 2022
        ticker = "SPY"
        start_date = "1973-01-01"
        end_date = "2023-05-01"
        sp500_data = yf.download(ticker, start=start_date, end=end_date)

        # Print the first 5 rows of the data
        print(sp500_data.head())

        #get percent growth from start to end
        start_price = sp500_data["Adj Close"][0]
        end_price = sp500_data["Adj Close"][-1]
        percent_growth = (end_price-start_price)/start_price

        dataset = pd.DataFrame(sp500_data["Adj Close"])

        #get the most important technical indicators
        ma_50 = sp500_data["Adj Close"].rolling(window=50).mean()
        ma_200 = sp500_data["Adj Close"].rolling(window=200).mean()
        ema_50 = sp500_data["Adj Close"].ewm(span=50, adjust=False).mean()
        ema_200 = sp500_data["Adj Close"].ewm(span=200, adjust=False).mean()

        #get the RSI (relative strength index)
        #RSI = 100 - 100/(1 + RS)
        rsi = ta.momentum.RSIIndicator(sp500_data["Adj Close"], window=14).rsi()

        # Download VIX data from Yahoo Finance
        vix = yf.download('^VIX', start=start_date, end=end_date)

        dataset['ma_50'] = ma_50
        dataset['ma_200'] = ma_200
        dataset['ema_50'] = ema_50
        dataset['ema_200'] = ema_200
        dataset['rsi'] = rsi

        #plot vix
        vix["Adj Close"].plot()

        gdp = web.DataReader('GDPC1', 'fred', start_date, end_date)
        inflation = web.DataReader('CPIAUCSL', 'fred', start_date, end_date)
        unemployment = web.DataReader('UNRATE', 'fred', start_date, end_date)

        #convert inflation to inflatin rate
        inflation_rate = inflation.pct_change()
        #add smoothed inflation rate
        inflation_rate['inflation_rate_12m'] = inflation_rate['CPIAUCSL'].rolling(window=12).mean()

        dataset['daily_returns'] = sp500_data["Adj Close"].pct_change().shift(-1)

        #calculate moving average of GDP
        gdp['gdp_7m'] = gdp['GDPC1'].rolling(window=7).mean()
        gdp['gdp_28m'] = gdp['GDPC1'].rolling(window=28).mean()
        gdp['gdp_ratio'] = (gdp['gdp_7m']/gdp['gdp_28m'])

        dataset['gdp_ratio'] = gdp['gdp_ratio']
        dataset['unemployment'] = unemployment['UNRATE']
        dataset['inflation_rate'] = inflation_rate['inflation_rate_12m']

        #investor Intelligence Sentiment Index
        consumer_sentimet = web.DataReader('UMCSENT', 'fred', start_date, end_date)
        equity_volatility = web.DataReader('EMVMACROBUS', 'fred', start_date, end_date)
        #plot iis
        equity_volatility.plot()

        dataset['consumer_sentimet'] = consumer_sentimet['UMCSENT']
        dataset['equity_volatility'] = equity_volatility['EMVMACROBUS']

        #ema50 - ema200
        dataset['norm_ema_ratio'] = (dataset['ema_50'] - dataset['ema_200'])/dataset['ema_200']
        plt.plot(dataset['norm_ema_ratio'])

        # Create a new column that identifies when the EMAs cross over each other
        dataset["EMA_crossover"] = ((dataset['norm_ema_ratio'].shift(1) < 0) & (dataset['norm_ema_ratio'] > 0)).astype(int)

        #rsa moving average
        dataset['rsi_ma_50'] = dataset['rsi'].rolling(window=50).mean()
        dataset['rsi_ma_200'] = dataset['rsi'].rolling(window=200).mean()

        # Calculate the rolling mean over a window of thirty days
        rolling_mean = dataset['Adj Close'].rolling(window=120).mean()
        dataset['percent_increase_smooth'] = (dataset['Adj Close'].shift(-1*pred_preiod) - rolling_mean) / rolling_mean * 100

        print(dataset.columns)
        cols_to_model = ['norm_ema_ratio',  'gdp_ratio', 'EMA_crossover', 'rsi', 
            'unemployment', 'consumer_sentimet','inflation_rate',
            'equity_volatility', 'rsi_ma_50']
            #'rsi_ma_200']
        #percent_inc = (dataset['Adj Close'].shift(-1)/dataset['Adj Close'] - 1)
        #find the percent increase over the next 30 days
        percent_inc_30 = dataset['percent_increase_smooth']
        #percent_inc = (dataset['Adj Close'].shift(-7) / dataset['Adj Close'] - 1) * 100 / 7
        y = percent_inc_30.shift(-1)
        dataset['pct_growth'] = y
        dataset = dataset[:-31]

        #get num na values in dataset['pct_growth']
        dataset['pct_growth'].isna().sum()
        dataset['y'] = .5
        dataset.loc[dataset['pct_growth'] > dataset['pct_growth'].median(), 'y'] = 1
        dataset.loc[dataset['pct_growth'] < dataset['pct_growth'].median(), 'y'] = 0
        dataset['y'] = dataset['pct_growth']

        X = dataset[cols_to_model].fillna(method='ffill')
        y = dataset['y']
        pct_change =  dataset['pct_growth']

        X = X[212:]
        y = y[212:]
        dates = dataset.index[212:]
        pct_change = pct_change[212:]
        pct_returns = dataset['daily_returns'][212:]

        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import KFold
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression

        scaler = StandardScaler()

        seed = 42

        kf = KFold(n_splits=5, shuffle=False)

        #split the data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, shuffle=False)
        # # Fit the scaler to your data to calculate the mean and standard deviation

        scaler.fit(X_train)

        #get nan values in dataset
        print(X_train.isna().sum())

        # # Apply the scaler to your data to transform it to a normalized form
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        #create use XGBoost model for regression
        import xgboost as xgb
        from sklearn.metrics import mean_squared_error

        scaler = StandardScaler()

        seed = 42

        kf = KFold(n_splits=5, shuffle=False)

        #split the data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, shuffle=False)

        # # Fit the scaler to your data to calculate the mean and standard deviation
        scaler.fit(X_train)

        #get nan values in dataset
        print(X_train.isna().sum())

        # # Apply the scaler to your data to transform it to a normalized form
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        #remove values that don't matter much
        mean_dist = np.mean(y_train)
        print(mean_dist)

        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import uniform, randint

        param_distributions = {
            'learning_rate': [0.01, 0.1, 0.5, 1],
            'max_depth': [1, 2, 3, 5, 7, 9, 13],
            'n_estimators': [1,2, 3, 5,10, 50, 100],
            'alpha': [0, 1, 10, 100, 1000],
            'colsample_bytree': [0.3, 0.5, 0.7, .9]
        }

        #y_train = np.where(y_train < 0, y_train, y_train)
        xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        xgb_cv = RandomizedSearchCV(xgb_reg, param_distributions, cv=kf, n_iter=1000, random_state=42, n_jobs=-1)


        trn_idxs = (y_train < mean_dist-dist) | (y_train > mean_dist+dist)
        X_train_scaled_extreme, train_y_extreme = X_train_scaled[trn_idxs], y_train[trn_idxs]
        xgb_cv.fit(X_train_scaled_extreme, train_y_extreme)

        print(xgb_cv.best_params_)

        configs = xgb_cv.best_params_


        train_scores = []
        scores = []
        correlations = []
        preds = []
        gt = []
        for train_idx, val_idx in kf.split(X_train_scaled):

            min_val_idx = min(val_idx)
            max_val_idx = max(val_idx)
            train_idx = [i for i in train_idx if i < min_val_idx-pred_preiod or i > max_val_idx]

            # split the data into training and validation sets for this fold
            train_X, val_X = X_train_scaled[train_idx], X_train_scaled[val_idx]
            train_y, val_y = y_train[train_idx], y_train[val_idx]
            #add more weight to values < 0

            #trainX and Y where percent_change < 0 or > 10
            trn_idxs = (train_y < mean_dist-dist) | (train_y > mean_dist+dist)
            train_X, train_y = train_X[trn_idxs], train_y[trn_idxs]
            
            #load xgb model from config
            xgb_model = xgb.XGBRegressor(objective='reg:squarederror', **configs)

            xgb_model.fit(train_X, train_y)

            #predict for this fold
            pred = xgb_model.predict(val_X)
            preds.append(pred)
            gt.append(val_y)

            train_rmse = np.sqrt(mean_squared_error(train_y, xgb_model.predict(train_X)))
            train_scores.append(train_rmse)

            #calculate rmse for this fold
            rmse = np.sqrt(np.mean((xgb_model.predict(val_X) - val_y) ** 2))
            scores.append(rmse)

            #calculate correlation for this fold
            corr = np.corrcoef(xgb_model.predict(val_X), val_y)[0, 1]
            correlations.append(corr)

        print(train_scores)
        print(scores)
        print(correlations)

        print('HERE IS THE RESULT:', pred_preiod, dist, np.mean(correlations))

