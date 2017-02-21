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
    iParamsDict = {'homepath': '/Users/fanch/Desktop/Titres/', 'stocklist : []'}
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
        r = requests.get(lUrl)
        if r.status_code == 200:
            lS = r.text
            oDf = pd.read_csv(StringIO(lS))
            if oDf.shape[0] > 0:
                oDf.loc[:, 'Stock'] = iStockSymbol
                oDf.loc[:, 'Date'] = pd.to_datetime(oDf.loc[:, 'Date'], format='%Y-%m-%d')
                oDf.sort_values('Date', inplace=True)
                return oDf.loc[:, ['Stock', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
            else:
                return None
        elif r.status_code == 404:
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
                    'Historical data for stock ' + iStockSymbol + ' was saved.')
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
                if (((dt.date.today() - date_last_maj).days > 1) & (dt.date.today().weekday() > 0)) | (
                    (dt.date.today().weekday() == 0) & (
                            (dt.date.today() - date_last_maj).days > 3)):
                    # get last available data
                    lDf = self.download_stock_data(iStockSymbol, date_last_maj + dt.timedelta(days=1), dt.date.today())
                    if lDf is not None:
                        # append last data to histo file (mode = 'a')
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
        """

        :param iSleepRange: min / max in sec to sleep between each call
        :return:
        """
        i = 1
        if iSleepRange is not None:
            for stock in self.Params['stocklist']:
                # print ('Pct completed : %s') % str(100.0*i/float(len(self.Params['stocklist'])))
                pct = 100.0 * i / float(len(self.Params['stocklist']))
                print 'MAJ of stock data... [%f%%]\r' % pct
                logging.info('Try to fetch stock data for : ' + stock)
                # Don't sleep if data is already up to date
                if self.is_up_to_date(stock):
                    self.save_stock_data(stock)
                else:
                    sleep = random.randint(iSleepRange[0], iSleepRange[1])
                    logging.info('Now sleeping for : %s seconds.' % (sleep,))
                    # Sleeps btw iSleepRange not to get tracked by Yahoo!
                    time.sleep(sleep)
                    self.save_stock_data(stock)
                logging.info(
                    'Data for stock ' + stock + ' whole consistency = ' + str(self.check_consistency(stock)))
                logging.info(
                    'Data for stock ' + stock + ' consistency since 2015 = ' + str(
                        self.check_consistency(stock, '2015-01-01')))
                i += 1
        else:
            for stock in self.Params['stocklist']:
                pct = 100.0 * i / float(len(self.Params['stocklist']))
                print 'MAJ of stock data... [%f%%]\r' % pct
                # print str(100.0*i/float(len(self.Params['stocklist'])))
                logging.info('Try to fetch stock data for : ' + stock)
                self.save_stock_data(stock)
                logging.info(
                    'Data for stock ' + stock + ' whole consistency = ' + str(self.check_consistency(stock)))
                logging.info(
                    'Data for stock ' + stock + ' consistency since 2015= ' + str(
                        self.check_consistency(stock, '2015-01-01')))
                i += 1
        return

    def check_consistency(self, iStockSymbol, iFromDate=None):
        """

        :param iStockSymbol: Stock symbol
        :param iFromDate: filter on date
        :return: consistency of DF :
        """
        if os.path.exists(self.Paths['DataPath'] + iStockSymbol + '/'):
            file_list = glob.glob(self.Paths['DataPath'] + iStockSymbol + '/*.csv')
            if len(file_list) == 1:
                df = pd.read_csv(file_list[0], header=0, sep=';')
                df.loc[:, 'Date'] = pd.to_datetime(df.loc[:, 'Date'], format='%Y-%m-%d')
                df.drop_duplicates(inplace=True)
                if iFromDate is not None:
                    if df['Date'].max() >= pd.to_datetime(iFromDate, format='%Y-%m-%d'):
                        df = df[df['Date'] >= pd.to_datetime(iFromDate, format='%Y-%m-%d')].copy()
                    else:
                        return 0.0
                calendar_df = pd.DataFrame(pd.date_range(df.loc[:, 'Date'].min(), df.loc[:, 'Date'].max(), freq='D'),
                                           columns=['FullDate'])
                calendar_df.loc[:, 'DayOfWeek'] = calendar_df.loc[:, 'FullDate'].dt.dayofweek
                df.loc[:, 'DayOfWeek'] = df.loc[:, 'Date'].dt.dayofweek
                num_days_cal = calendar_df[calendar_df['DayOfWeek'] < 5].shape[0]
                num_days_df = df[df['DayOfWeek'] < 5].shape[0]
                return num_days_df / float(num_days_cal)
        return 0.0

    def is_up_to_date(self, iStockSymbol):
        """

        :param iStockSymbol: Stock Symbol
        :return: True if data is up-to-date
        """
        if os.path.exists(self.Paths['DataPath'] + iStockSymbol + '/'):
            file_list = glob.glob(self.Paths['DataPath'] + iStockSymbol + '/*.csv')
            if len(file_list) == 1:
                file_name = file_list[0].replace(self.Paths['DataPath'] + iStockSymbol + '/', '')
                # Get last maj date (from filename)
                date_last_maj = pd.to_datetime(file_name.replace('.csv', '').replace(iStockSymbol + '_', ''),
                                               format='%Y-%m-%d').date()
                # if last available data was not yesterday or if today is monday and last available data is last friday
                if ((dt.date.today() - date_last_maj).days <= 1) | ((dt.date.today().weekday() == 0) & (
                            (dt.date.today() - date_last_maj).days == 3)):
                    return True
        return False

    def is_usable(self, iStockSymbol, iFromDate):
        if self.is_up_to_date(iStockSymbol) & (self.check_consistency(iStockSymbol, iFromDate) == 1.0):
            return True
        return False

    def get_data(self, iStockSymbol):
        file_list = glob.glob(self.Paths['DataPath'] + iStockSymbol + '/*.csv')
        # We must have only one file
        if len(file_list) == 1:
            df = pd.read_csv(file_list[0], header=0, sep=';')
            df.drop_duplicates(inplace=True)
            df.loc[:, 'Date'] = pd.to_datetime(df.loc[:, 'Date'], format='%Y-%m-%d')
            df.loc[:, 'Volume'] = df.loc[:, 'Volume'].astype(float)
            logging.info('Data for stock ' + iStockSymbol + ' and date >= ' + df['Date'].min().strftime(
                format='%Y-%m-%d') + ' returned.')
            df.columns = [s.lower() for s in df.columns]
        else:
            logging.warning('Multiple data files for stock ' + iStockSymbol + ' were found. Please Check.')
            return None

        return df
