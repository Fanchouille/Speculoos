import PathHandler as ph
import requests
from io import StringIO
import pandas as pd
import datetime as dt
import os
import glob
import time
import random

import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')


class DataHandler:
    """
    Params_MagInt = {'homepath': '/Users/fanch/Desktop/Titres/', 'stocklist : []'}
    """

    def __init__(self, iParamsDict):
        self.Params = iParamsDict
        # Create and save  paths (Data / Logs)
        ph.create_all_paths(self.Params['homepath'])
        self.Paths = ph.get_all_paths(self.Params['homepath'])
        # Create logger
        self.logfile = self.create_logger()

    def create_logger(self):
        """
        Creates log file
        :return:
        """
        file_handler = RotatingFileHandler(self.Paths['LogsPath'] + 'logfile.log', 'a', 1000000, 1)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        self.file_handler = file_handler
        return

    def download_stock_data(self, iStockSymbol, iFromDate=None, iToDate=None):
        """

        :param iStockSymbol: Stock Symbol (SLB-PA for instance)
        :param iDate: start Date
        :return: historical data from Yahoo as a pandas DF
        source : https://github.com/mdengler/stockquote/blob/master/stockquote.py
        """
        if iFromDate is None:
            lUrl = ("http://ichart.finance.yahoo.com/table.csv?"
                    "s=%s" % (iStockSymbol,))
        else:
            lUrl = ("http://ichart.finance.yahoo.com/table.csv?"
                    "s=%s&"
                    "a=%s&"
                    "b=%s&"
                    "c=%s&"
                    "d=%s&"
                    "e=%s&"
                    "f=%s&"
                    "g=d&"
                    "ignore=.csv" % (
                        iStockSymbol, iFromDate.month - 1, iFromDate.day, iFromDate.year, iToDate.month - 1,
                        iToDate.day,
                        iToDate.year,))
        # print lUrl
        lS = requests.get(lUrl).text
        oDf = pd.read_csv(StringIO(lS))
        if oDf.shape[0] > 0:
            oDf.loc[:, 'Stock'] = iStockSymbol
            oDf.loc[:, 'Date'] = pd.to_datetime(oDf.loc[:, 'Date'], format='%Y-%m-%d')
            oDf.sort_values('Date', inplace=True)
            return oDf.loc[:, ['Stock', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
        else:
            return None

    def save_stock_data(self, iStockSymbol):
        """

        :param iStockSymbol: Stock symbol
        :return: save file as CSV
        """
        # If data dir does not exists, we don't have historical data
        if not os.path.exists(self.Paths['DataPath'] + iStockSymbol + '/'):
            lDf = self.download_stock_data(iStockSymbol)
            if lDf is not None:
                ph.create_path(self.Paths['DataPath'] + iStockSymbol + '/')
                max_date = lDf.loc[:, 'Date'].max()
                lDf.to_csv(self.Paths['DataPath'] + iStockSymbol + '/' + iStockSymbol + '_' + max_date.strftime(
                    format='%Y-%m-%d') + '.csv', index=False, sep=';')
                logging.info(
                    'File already exists : Historical data for stock ' + iStockSymbol + ' was saved.')
            else:
                logging.warning(
                    'Historical data for stock ' + iStockSymbol + ' was not saved : data was not found.')
        # There is already a file
        else:
            file_list = glob.glob(self.Paths['DataPath'] + iStockSymbol + '/*.csv')
            # We must have only one file
            if len(file_list) == 1:
                file_name = file_list[0].replace(self.Paths['DataPath'] + iStockSymbol + '/', '')
                # Get last maj date (from filename)
                date_last_maj = pd.to_datetime(file_name.replace('.csv', '').replace(iStockSymbol + '_', ''),
                                               format='%Y-%m-%d').date()
                # if last available data was not yesterday
                if (dt.date.today() - date_last_maj).days > 1:
                    # get last available data
                    lDf = self.download_stock_data(iStockSymbol, date_last_maj + dt.timedelta(days=1), dt.date.today())
                    if lDf is not None:
                        # append last data to histo file
                        # with open(self.Paths['DataPath'] + iStockSymbol + '/' + file_name, 'a') as f:
                        max_date = lDf.loc[:, 'Date'].max()
                        lDf.to_csv(self.Paths['DataPath'] + iStockSymbol + '/' + file_name, mode='a', header=False,
                                   index=False, sep=';')
                        os.rename(self.Paths['DataPath'] + iStockSymbol + '/' + file_name,
                                  self.Paths['DataPath'] + iStockSymbol + '/' + iStockSymbol + '_' + max_date.strftime(
                                      format='%Y-%m-%d') + '.csv')
                        logging.info(
                            'Historical data for stock ' + iStockSymbol + ' was updated.')
                    else:
                        logging.warning(
                            'Historical data for stock ' + iStockSymbol + ' was not updated :  : data was not found.')
                else:
                    logging.info(
                        'Historical data for stock ' + iStockSymbol + ' was not updated : data is already up to date.')
            return

    def save_all_stocks(self, iSleepRange=None):
        if iSleepRange is not None:
            for stock in self.Params['stocklist']:
                logging.info('Try to fetch stock data for : ' + stock)
                sleep = random.randint(iSleepRange[0], iSleepRange[1])
                logging.info('Now sleeping for : %s seconds.' % (sleep,))
                # Sleeps btw iSleepRange not to get tracked by Yahoo!
                time.sleep(sleep)
                self.save_stock_data(stock)
        else:
            for stock in self.Params['stocklist']:
                logging.info('Try to fetch stock data for : ' + stock)
                self.save_stock_data(stock)
        return

    def check_consistency(self, iStockSymbol):
        if os.path.exists(self.Paths['DataPath'] + iStockSymbol + '/'):
            file_list = glob.glob(self.Paths['DataPath'] + iStockSymbol + '/*.csv')
            if len(file_list) == 1:
                df = pd.read_csv(file_list[0], header=0, sep=';')
                df.loc[:, 'Date'] = pd.to_datetime(df.loc[:, 'Date'], format='%Y-%m-%d')
                calendar_df = pd.DataFrame(pd.date_range(df.loc[:, 'Date'].min(), df.loc[:, 'Date'].max(), freq='D'),
                                   columns=['FullDate'])
                calendar_df.loc[:, 'DayOfWeek'] = calendar_df.loc[:, 'FullDate'].dt.dayofweek
                num_days = calendar_df[calendar_df['DayOfWeek'] < 5].shape[0]
                return df.shape[0]/float(num_days)
        return 0.0