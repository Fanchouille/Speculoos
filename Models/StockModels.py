from DataManagement import PathHandler as ph
from DataManagement import DataHandler as dhand
from DataManagement import DataFeaturizer as dfeat
import requests
from io import StringIO
import pandas as pd
import datetime as dt
import os
import glob

import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')


class StockModels:
    """
        iParamsDict = {'homepath': '/Users/fanch/Desktop/Titres/', 'stocklist' : [], 'stockparams' : {}}
    """

    def __init__(self, iParamsDict):
        self.Params = iParamsDict
        # Create logger
        self.logfile = self.create_logger()
        self.dath = dhand.DataHandler({k: self.Params for k in ('homepath', 'stocklist') if k in self.Params})
        self.create_logger()

    def create_logger(self):
        """
        Creates log file
        :return:
        """
        file_handler = RotatingFileHandler(self.Paths['LogsPath'] + 'models_logfile.log', 'a', 1000000, 1)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        self.file_handler = file_handler
        return

    def featurize_stock_data(self, df, iFromDate=None, **kwargs):
        """

        :param iStockSymbol: Stock Symbol
        :param iFromDate: iFromDate to filter the dataset
        :param kwargs: { 'overlap' : {} , 'momentum':{} , 'volatility':{}, 'pattern':[] , 'volume': {} , 'cycle':{}}
        In principal dict :
        put None as value to remove indicators 'pattern': None => remove chandeliers indicators
        don't put key to get all indicators by default
        In Sub dict :
        don't put key to get default specific indicator
        put Key + None to remove specific indicator
        put Key + Values to get custom specific indicator
        examples :
        'volume': {'AD': **kwargs: , 'ADOSC': **kwargs, 'OBV': **kwargs}
        'pattern' : choose which chandelier to get in following list
        ['CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE', 'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY', 'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU', 'CDLCONCEALBABYSWALL', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI', 'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS', 'CDLINNECK', 'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU', 'CDLMATCHINGLOW', 'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLONNECK', 'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP', 'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS']

        :return:
        """
        if df is not None:
            df.columns = [s.lower() for s in df.columns]

            # get 1st features (before featuring)
            bef_feat = [col for col in df.columns if col not in ['stock', 'date']]
            cur_feat = bef_feat.copy()

            # IF WE SPECIFY PATTERN INDICATORS
            if kwargs.has_key('pattern'):
                # IF A LIST OF CHANDELIERS IS PROVIDED
                if kwargs['pattern'] is not None:
                    df = dfeat.pattern_chandeliers(df, kwargs['pattern'])
            # ELSE USE DEFAULT (ALL) LIST
            else:
                df = dfeat.pattern_chandeliers(df)

            # IF WE SPECIFY VOLUME INDICATORS
            if kwargs.has_key('volume'):
                # IF A LIST OF VOLUME INDICATOR (AND POSSIBLY PARAMETERS IS PROVIDED)
                # AD
                if kwargs['volume'].has_key('AD'):
                    if kwargs['volume']['AD'] is not None:
                        df = dfeat.volume_ad(df)  # No parameter for AD
                # ADOSC
                if kwargs['volume'].has_key('ADOSC'):
                    if kwargs['volume'][
                        'ADOSC'] is not None:  # ADOSC default parameters : iFastPeriod=3, iSlowPeriod=10
                        df = dfeat.volume_adosc(df, kwargs['volume']['ADOSC'])
                    else:
                        df = dfeat.volume_adosc(df)
                # OBV
                if kwargs['volume'].has_key('OBV'):
                    if kwargs['volume']['OBV'] is not None:
                        df = dfeat.volume_obv(df, kwargs['volume'][
                            'OBV'])  # OBV default parameters : iAddSMA=True, iPeriod=20
                    else:
                        df = dfeat.volume_obv(df)
            # ELSE USE ALL
            else:
                df = dfeat.volume_ad(df)
                df = dfeat.volume_adosc(df)
                df = dfeat.volume_obv(df)

            # IF WE SPECIFY VOLATILITY INDICATORS
            if kwargs.has_key('volatility'):
                # IF A LIST OF VOLATILITY INDICATOR (AND POSSIBLY PARAMETERS IS PROVIDED)
                # ATR
                if kwargs['volatility'].has_key('ATR'):
                    if kwargs['volatility']['ATR'] is not None:
                        df = dfeat.volatility_ATR(df,
                                                  kwargs['volatility']['ATR'])  # ATR default parameters : iPeriod=14
                    else:
                        df = dfeat.volatility_ATR(df)
            # ELSE USE ALL
            else:
                df = dfeat.volatility_ATR(df)

            # IF WE SPECIFY VOLATILITY INDICATORS
            if kwargs.has_key('overlap'):
                # IF A LIST OF VOLATILITY INDICATOR (AND POSSIBLY PARAMETERS IS PROVIDED)
                # SMA CROSS
                if kwargs['overlap'].has_key('SMA_CROSS'):
                    if kwargs['overlap']['SMA_CROSS'] is not None:
                        df = dfeat.overlap_sma_cross(df, kwargs['overlap'][
                            'SMA_CROSS'])  # SMA_CROSS default parameters : iPeriod1=20, iPeriod2=50
                    else:
                        df = dfeat.overlap_sma_cross(df)
                # Bollinger Bands
                if kwargs['overlap'].has_key('BBANDS'):
                    if kwargs['overlap']['BBANDS'] is not None:
                        df = dfeat.overlap_bbands(df, kwargs['overlap'][
                            'BBANDS'])  # BBANDS default parameters : iTimePeriod=10, iStdUp=2, iStdDwn=2
                    else:
                        df = dfeat.overlap_bbands(df)
            # ELSE USE ALL
            else:
                df = dfeat.overlap_sma_cross(df)
                df = dfeat.overlap_bbands(df)

            # IF WE SPECIFY MOMENTUM INDICATORS
            if kwargs.has_key('momentum'):
                # IF A LIST OF MOMENTUM INDICATOR (AND POSSIBLY PARAMETERS IS PROVIDED)
                # MACD
                if kwargs['momentum'].has_key('MACD'):
                    if kwargs['momentum']['MACD'] is not None:
                        df = dfeat.momentum_macd(df, kwargs['momentum'][
                            'MACD'])  # MACD default parameters : iFastPeriod=12, iSlowPeriod=26, iSignalPeriod=9, iThresPeriod=14
                    else:
                        df = dfeat.momentum_macd(df)
                # PPO
                if kwargs['momentum'].has_key('PPO'):
                    if kwargs['momentum']['PPO'] is not None:
                        df = dfeat.momentum_ppo(df, kwargs['momentum'][
                            'PPO'])  # PPO default parameters : iFastPeriod=12, iSlowPeriod=26
                    else:
                        df = dfeat.momentum_ppo(df)
            # ELSE USE ALL
            else:
                df = dfeat.momentum_macd(df)
                df = dfeat.momentum_ppo(df)

            new_feat = [col for col in df.columns if col not in cur_feat]
            cur_feat.append(new_feat)

        else:
            return None, None
        if iFromDate is not None:
            df = df[df['date'] >= pd.to_datetime(iFromDate, format='%Y-%m-%d')].copy()
        return df, cur_feat

    # Put this in ordonancer
    def get_data(self, iStockSymbol):
        if self.dath.is_up_to_date(iStockSymbol):
            file_list = glob.glob(self.Paths['DataPath'] + iStockSymbol + '/*.csv')
            # We must have only one file
            if len(file_list) == 1:
                df = pd.read_csv(file_list[0], header=0, sep=';')
                df.loc[:, 'Date'] = pd.to_datetime(df.loc[:, 'Date'], format='%Y-%m-%d')
                df.drop_duplicates(inplace=True)
                logging.info('Data for stock ' + iStockSymbol + ' and date >= ' + df['Date'].min().strftime(
                    format='%Y-%m-%d') + ' returned.')
            else:
                logging.warning('Multiple data files for stock ' + iStockSymbol + ' were found. Please Check.')
                return None
        else:
            logging.warning('Data for stock ' + iStockSymbol + ' is not usable. Please Check.')
            return None
        return df

    # df = self.get_data(iStockSymbol)

    def create_model_path(self, iStockSymbol):
        if os.path.exists(self.Paths['DataPath'] + iStockSymbol + '/'):
            ph.create_path(self.Paths['DataPath'] + iStockSymbol + '/Models/')
            return
