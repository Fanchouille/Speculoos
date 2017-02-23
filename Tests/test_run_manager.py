from Run import RunManager as RunManag
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

# fitted_models = runM.train_models_on_stock_data(stock, '2015-01-01')
# runM.save_fitted_models(fitted_models, stock)
# df_pred = runM.apply_models_on_data(stock, '2016-01-01')

# runM.train_and_save_models('2015-01-01')

# runDate = '2017-01-01'
# runM.get_predictions_on_stocklist(runDate).to_csv(PARAMS['homepath']+'/results_'+runDate+'.csv', sep=';', index=False)

runM.daily_run(iSleepRange=(2, 5), iTrainingFromDate='2015-01-01', iRetrain=False)
