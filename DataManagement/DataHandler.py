# coding: utf-8
import PathHandler as ph
import requests
from io import StringIO
import pandas as pd
import datetime as dt
import os
import glob
import time
import random
from lxml import html
import numpy as np
import fix_yahoo_finance as yf

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

    def download_stock_data(self, iStockSymbol, iFromDate=None, iToDate=None, iUse_Fix_Yahoo=True):
        """

        :param iStockSymbol: Stock Symbol (SLB-PA for instance)
        :param iDate: start Date
        :return: historical data from Yahoo as a pandas DF
        source : https://github.com/mdengler/stockquote/blob/master/stockquote.py
        """

        if iUse_Fix_Yahoo:
            if iFromDate is not None:
                try:
                    oDf = yf.download(iStockSymbol, start=iFromDate.strftime("%Y-%m-%d"),
                                      end=iToDate.strftime("%Y-%m-%d"))
                except:
                    print iStockSymbol + ' data was not fetched.'
                    oDf = None
                if oDf is not None:
                    oDf.reset_index(inplace=True)
                    oDf.loc[:, 'Stock'] = iStockSymbol
                    oDf.loc[:, 'Date'] = pd.to_datetime(oDf.loc[:, 'Date'], format='%Y-%m-%d')
                    oDf.sort_values('Date', inplace=True)
                    return oDf[oDf['Volume'] > 0].loc[:,
                           ['Stock', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
                else:
                    return None
            else:
                # If no date is provided, dowload from 1st jan 2000
                oDf = yf.download(iStockSymbol, start='2000-01-01', end=dt.date.today().strftime("%Y-%m-%d"))
                if oDf is not None:
                    oDf.reset_index(inplace=True)
                    oDf.loc[:, 'Stock'] = iStockSymbol
                    oDf.loc[:, 'Date'] = pd.to_datetime(oDf.loc[:, 'Date'], format='%Y-%m-%d')
                    oDf.sort_values('Date', inplace=True)
                    return oDf[oDf['Volume'] > 0].loc[:,
                           ['Stock', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]

        else:
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
                    return oDf[oDf['Volume'] > 0].loc[:,
                           ['Stock', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
                else:
                    return None
            elif r.status_code == 404:
                return None

    def check_abc_advise(self, t):
        if ('achat' in t.lower()) & ('baisse' not in t.lower()):
            return 'Achat'
        elif ('nuage' in t.lower()) | ('consolidation' in t.lower()):
            return 'Conserver'
        elif u'sécurité' in t.lower():
            return 'Vente_Faible'
        elif (u'baissier' in t.lower()) | ('patience' in t.lower()):
            return 'Vente_Forte'
        else:
            return 'Pas de conseil.'

    def get_stock_info(self, iStockSymbol, iFromDate):
        url = "https://www.abcbourse.com/marches/events.aspx?s=" + iStockSymbol + "p"
        try:
            r = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'})
        except requests.exceptions.ConnectionError:
            # r.status_code = "Connection refused"
            r = None

        if r is not None:
            tree = html.fromstring(r.content)

            liste_fields = tree.xpath('//td/text()')
            stock_info = pd.DataFrame([unicode(field.encode('latin1'), 'utf-8') for field in liste_fields],
                                      columns=['RawText'])
            stock_info.loc[:, 'RealDates'] = stock_info.loc[:, 'RawText'].map(
                lambda x: pd.to_datetime(x, format='%d/%m/%Y', errors='coerce').date())
            stock_info.loc[:, 'EventType'] = stock_info.loc[:, 'RawText'].map(
                lambda x: x if (u'Résultat' in x) | (u'Chiffre' in x) else np.nan)
            stock_info.loc[:, 'EventDates'] = stock_info.loc[:, 'RealDates'].shift(1)
            stock_info.loc[:, 'EventComments'] = stock_info.loc[:, 'RawText'].shift(-1)
            stock_info.loc[:, 'EventComments'] = stock_info.loc[:, 'EventComments'].map(
                lambda x: 'Before' if u'Avant' in unicode(x) else 'After' if u'Après' in unicode(x) else np.nan)
        else:
            stock_info = pd.DataFrame()

        if stock_info.shape[0] > 0:
            stock_info.loc[:, 'EventAlertDates'] = stock_info.apply(
                lambda x: (x['EventDates'] - dt.timedelta(days=1)) if (x['EventComments'] == 'Before') else x[
                    'EventDates'],
                axis=1)
            stock_info.loc[:, 'stock'] = iStockSymbol

            stock_info.loc[:, 'dateInfo'] = iFromDate

            if (stock_info.loc[:, 'EventDates'].dropna().max() - iFromDate).days > 0:
                cleaned_stock_info = stock_info[stock_info.loc[:, 'EventDates'] > iFromDate].loc[:,
                                     ['stock', 'dateInfo', 'EventType', 'EventDates', 'EventAlertDates']]
            else:
                cleaned_stock_info = stock_info[
                                         stock_info.loc[:, 'EventDates'] == stock_info.loc[:,
                                                                            'EventDates'].dropna().max()].loc[:,
                                     ['stock', 'dateInfo', 'EventType', 'EventDates', 'EventAlertDates']]

            cleaned_stock_info.loc[:, 'TimeToEvent'] = (
                cleaned_stock_info.loc[:, 'EventDates'] - cleaned_stock_info.loc[:, 'dateInfo'])
            cleaned_stock_info.loc[:, 'TimeToEvent'] = cleaned_stock_info.loc[:, 'TimeToEvent'].map(
                lambda x: x.days if x.days > 0 else 0)

            # Advise of ABC bourse
            url = "https://www.abcbourse.com/analyses/conseil.aspx?s=" + iStockSymbol + "p"
            r = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'})
            tree = html.fromstring(r.content)
            liste_fields = tree.xpath('//td/text()')
            text = [unicode(field.encode('latin1').replace('\t', '').replace('\r\n', '').strip(), 'utf8') \
                    for field in liste_fields if
                    field.encode('latin1').replace('\t', '').replace('\r\n', '').strip() != '']

            cleaned_stock_info.loc[:, 'Conseil'] = self.check_abc_advise(text[2])
            cleaned_stock_info.loc[:, 'stock'] = cleaned_stock_info.loc[:, 'stock'].map(lambda x: x + '.PA')
            sleep = random.randint(3, 5)
            time.sleep(sleep)

            return cleaned_stock_info.sort_values('TimeToEvent').iloc[[0]]
        else:
            return None

    def get_stock_info_from_stocklist(self, iStockList, iFromDate):
        return pd.concat([self.get_stock_info(StockSymbol, iFromDate) for StockSymbol in iStockList])

    def clean_stock_data(self, iStockSymbol):
        if os.path.exists(self.Paths['DataPath'] + iStockSymbol + '/'):
            file_list = glob.glob(self.Paths['DataPath'] + iStockSymbol + '/*.csv')
            # We must have only one file
            if len(file_list) == 1:
                file_name = file_list[0].replace(self.Paths['DataPath'] + iStockSymbol + '/', '')
                df = pd.read_csv(self.Paths['DataPath'] + iStockSymbol + '/' + file_name, header=0, sep=';')
                df[df['Volume'] > 0].to_csv(self.Paths['DataPath'] + iStockSymbol + '/' + file_name,
                                            index=False, sep=';')
        return

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
                if ((dt.date.today() - date_last_maj).days > 1):
                    # get last available data
                    # print date_last_maj
                    lDf = self.download_stock_data(iStockSymbol, date_last_maj + dt.timedelta(days=1), dt.date.today())
                    if (lDf is not None):
                        if (lDf.shape[0] > 0):
                            # append last data to histo file (mode = 'a')
                            max_date = lDf.loc[:, 'Date'].max()
                            lDf.to_csv(self.Paths['DataPath'] + iStockSymbol + '/' + file_name, mode='a', header=False,
                                       index=False, sep=';')
                            os.rename(self.Paths['DataPath'] + iStockSymbol + '/' + file_name,
                                      self.Paths[
                                          'DataPath'] + iStockSymbol + '/' + iStockSymbol + '_' + max_date.strftime(
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
                if ((dt.date.today() - date_last_maj).days <= 1) | ((date_last_maj.weekday() == 4) & (
                            (dt.date.today() - date_last_maj).days <= 3)):
                    return True
        return False

    def is_usable(self, iStockSymbol, iFromDate):
        # 0.95 not to have to handle  days
        # print self.is_up_to_date(iStockSymbol)
        # print self.check_consistency(iStockSymbol, iFromDate)
        if self.is_up_to_date(iStockSymbol) & (self.check_consistency(iStockSymbol, iFromDate) >= 0.95):
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

    def get_moves_histo(self):
        file_list = glob.glob(self.Paths['PFPath'] + 'Movements/' + '*.csv')
        if len(file_list) == 1:
            oDf = pd.read_csv(file_list[0], header=0, sep=';')
            oDf.loc[:, 'Date'] = pd.to_datetime(oDf.loc[:, 'Date'], format='%Y-%m-%d')

        else:
            logging.warning('No moves histo was found. Please Check.')
            return None
        return oDf.sort_values(['date', 'stock'], ascending=[1, 1])

    def get_portfolio(self):
        file_list = glob.glob(self.Paths['PFPath'] + 'PF/' + '*.csv')
        if len(file_list) == 1:
            oDf = pd.read_csv(file_list[0], header=0, sep=';')

        else:
            logging.warning('No portfolio was found. Please Check.')
            return None
        return oDf.sort_values(['stock'], ascending=[1])

    def create_move(self, iDate, iMoveType, iStockSymbol, iPrice, iQty):
        oDf = pd.DataFrame([iDate, iMoveType, iStockSymbol, iPrice, iQty],
                           columns=['date', 'movetype', 'stock', 'price', 'qty'])
        oDf.loc[:, 'Date'] = pd.to_datetime(oDf.loc[:, 'Date'], format='%Y-%m-%d')
        return oDf

    def add_move(self, iDate, iMoveType, iStockSymbol, iPrice, iQty):
        file_list = glob.glob(self.Paths['PFPath'] + 'Movements/' + '*.csv')
        mDf = self.create_move(iDate, iMoveType, iStockSymbol, iPrice, iQty)
        if len(file_list) == 0:
            mDf.to_csv(self.Paths['movesPath'] + 'moves_histo_' + iDate + '.csv', header=True,
                       index=False, sep=';')
        else:
            file_name = file_list[0].replace(self.Paths['PFPath'] + 'Movements/', '')
            date_last_maj = pd.to_datetime(file_name.replace('.csv', '').replace('moves_histo_', ''),
                                           format='%Y-%m-%d').date()
            iDate = pd.to_datetime(iDate, format='%Y-%m-%d').date()
            mDf.to_csv(file_list[0], mode='a', header=False,
                       index=False, sep=';')
            if iDate > date_last_maj:
                os.rename(file_list[0],
                          self.Paths['PFPath'] + 'Movements/' + 'moves_histo_' + iDate.strftime(
                              format='%Y-%m-%d') + '.csv')
        return

    def save_stock_info_sql_lite(self, iFromDate):

        return
