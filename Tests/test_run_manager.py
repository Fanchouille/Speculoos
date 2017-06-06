from Run import RunManager as RunManag
from Models import DeepModels as dl
import pandas as pd

# .PA means PARIS STOCK EXCHANGE
# Load file of all eurolist stocks
stocklist_df = pd.read_csv('/Users/fanch/Desktop/Titres/eurolist_nom.csv', header=0, sep=';')

# Params to get data
stocklist = sorted(stocklist_df[stocklist_df['EUROLIST'].isin(['A', 'B'])].loc[:, 'StockSymbol'].unique())
stocklist = [stock for stock in stocklist if stock not in ['APAM.PA', 'ASIT.PA']]

PARAMS = {'homepath': '/Users/fanch/Desktop/Titres',
          'stocklist': stocklist,
          # 'stockparams':
          #    {'SGO.PA': {'features': {'pattern': None}}}
          }

# Launch Run manager
runM = RunManag.RunManager(PARAMS)
print stocklist

# Test Run
# stock = 'SGO.PA'
# df_pred, targets = runM.compute_targets_duration_on_stock_data(stock, '2012-01-01')
# df_pred.to_csv('test_stock_duration.csv', sep=';', index=False)

# TEST DEEP LEARNING
# df = runM.DataHand.get_data(stock)
# df_feat, features = runM.featurize_stock_data(df, stock, '2012-01-01')
# df_train, targets = runM.compute_targets_on_stock_data(df_feat, stock, '2012-01-01')

# model = KerasClassifier(build_fn=dl.construct_deep_model(len(features)))
# scaler, fittedmodels = dl.fit_models(df_train, features, targets)
# dl.save(fittedmodels,scaler)


# DAILY RUN
# print runM.get_predictions_on_stock('AC.PA', iModelDate=None, iNumDays=1, iType='deep')
runM.daily_run(iSleepRange=(10, 15), iTrainingFromDate='2013-01-01', iRetrain=False, iType='classifier')

# print runM.DataHand.get_stock_info_from_stocklist([stock.replace('.PA','') for stock in stocklist[:5]], pd.to_datetime('2017-05-11').date()).dtypes

# print runM.DataHand.get_stock_info_from_stocklist([stock.replace('.PA','') for stock in stocklist[:5]], pd.to_datetime('2017-05-11').date())
