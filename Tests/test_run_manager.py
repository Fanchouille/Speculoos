from Run import RunManager as RunManag
from Models import DeepModels as dl
import pandas as pd

# .PA means PARIS STOCK EXCHANGE
# Load file of all eurolist stocks
stocklist_df = pd.read_csv('/Users/fanch/Desktop/Titres/eurolist_nom.csv', header=0, sep=';')

# Params to get data
stocklist = sorted(stocklist_df[stocklist_df['EUROLIST'] == 'A'].loc[:, 'StockSymbol'].unique())
PARAMS = {'homepath': '/Users/fanch/Desktop/Titres',
          'stocklist': stocklist,
          # 'stockparams':
          #    {'SGO.PA': {'features': {'pattern': None}}}
          }

# Launch Run manager
runM = RunManag.RunManager(PARAMS)

# Test Run
# stock = 'SGO.PA'
# df_pred, targets = runM.compute_targets_duration_on_stock_data(stock, '2012-01-01')
# df_pred.to_csv('test_stock_duration.csv', sep=';', index=False)

# TEST DEEP LEARNING
# df = runM.DataHand.get_data(stock)
# df_feat, features = runM.featurize_stock_data(df, stock, '2012-01-01')
#df_train, targets = runM.compute_targets_on_stock_data(df_feat, stock, '2012-01-01')

# model = KerasClassifier(build_fn=dl.construct_deep_model(len(features)))
# scaler, fittedmodels = dl.fit_models(df_train, features, targets)
#dl.save(fittedmodels,scaler)


# DAILY RUN
runM.daily_run(iSleepRange=(1, 1), iTrainingFromDate='2013-01-01', iRetrain=False, iType='classifier')
runM.train_and_save_models(iFromDate='2012-01-01', iRetrain=False, iType='deep')
# runM.daily_run(iSleepRange=(1, 1), iTrainingFromDate='2015-01-01', iRetrain=False, iType='regressor')
