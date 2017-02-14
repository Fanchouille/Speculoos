from DataManagement import DataHandler as datHand
import pandas as pd

# .PA means PARIS STOCK EXCHANGE
# Load file of all eurolist stocks
stocklist_df = pd.read_csv('/Users/fanch/Desktop/Titres/eurolist_nom.csv', header=0, sep=';')

# Params to get data
stocklist = stocklist_df[stocklist_df['EUROLIST'] == 'A'].loc[:, 'StockSymbol'].unique()
PARAMS = {'homepath': '/Users/fanch/Desktop/Titres', 'stocklist': stocklist}

# Get data
handData = datHand.DataHandler(PARAMS)
handData.save_all_stocks(iSleepRange=(1, 3))

