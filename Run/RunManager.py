from DataManagement import PathHandler as ph
from DataManagement import DataHandler as dhand
from DataManagement import DataFeaturizer as dfeat
import datetime as dt
import pandas as pd

import os
import glob

import logging
from logging.handlers import RotatingFileHandler

from Models import StockModels as stoMod

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')


class RunManager:
    """
        iParamsDict = {'homepath': '/Users/fanch/Desktop/Titres/', 'stocklist' : [],
         'stockparams' : { 'SGO.PA' :{'features' :{},'targets': {}, 'models' :{}}, 'BNP.PA' : etc...}
    """

    def __init__(self, iParamsDict):
        self.Params = iParamsDict
        self.Paths = ph.get_all_paths(self.Params['homepath'])
        self.DataHand = dhand.DataHandler(
            {k: self.Params[k] for k in self.Params.keys() if k in ['homepath', 'stocklist']})
        # Create logger
        self.logfile = self.create_logger()

    def create_logger(self):
        """
        Creates log file
        :return:
        """
        file_handler = RotatingFileHandler(self.Paths['LogsPath'] + 'run_logfile.log', 'a', 1000000, 1)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        self.file_handler = file_handler
        return

    def create_models_path(self, iStockSymbol):
        if not os.path.exists(self.Paths['ModelsPath'] + iStockSymbol + '/'):
            ph.create_path(self.Paths['ModelsPath'] + iStockSymbol + '/')
            return

    def create_results_path(self):
        if not os.path.exists(self.Paths['ResultsPath']):
            ph.create_path(self.Paths['ResultsPath'])
            return

    def featurize_stock_data(self, df, iStockSymbol, iFromDate=None):
        """

        :param df:
        :param iStockSymbol:
        :param iFromDate:
        :return:
        """
        # Featurize it - Check if parameters are there
        if self.Params.has_key('stockparams'):
            if self.Params['stockparams'].has_key(iStockSymbol):
                if self.Params['stockparams'][iStockSymbol].has_key('features'):
                    df_feat, features = dfeat.featurize_stock_data(df, iFromDate,
                                                                   **self.Params['stockparams'][iStockSymbol][
                                                                       'features'])
                else:
                    df_feat, features = dfeat.featurize_stock_data(df, iFromDate)
            else:
                df_feat, features = dfeat.featurize_stock_data(df, iFromDate)
        else:
            df_feat, features = dfeat.featurize_stock_data(df, iFromDate)

        return df_feat, features

    def compute_targets_on_stock_data(self, df_feat, iStockSymbol, iFromDate):

        if self.Params.has_key('stockparams'):
            if self.Params['stockparams'].has_key(iStockSymbol):
                if self.Params['stockparams'][iStockSymbol].has_key('targets'):
                    df_train, targets = dfeat.compute_target(df_feat, iFromDate,
                                                             **self.Params['stockparams'][iStockSymbol][
                                                                 'targets'])
                else:
                    df_train, targets = dfeat.compute_target(df_feat, iFromDate)
            else:
                df_train, targets = dfeat.compute_target(df_feat, iFromDate)
        else:
            df_train, targets = dfeat.compute_target(df_feat, iFromDate)
        return df_train, targets

    def create_models_for_stock(self, iStockSymbol, targets):
        if self.Params.has_key('stockparams'):
            if self.Params['stockparams'].has_key(iStockSymbol):
                if self.Params['stockparams'][iStockSymbol].has_key('models'):
                    models = stoMod.create_models(targets, **self.Params['stockparams'][iStockSymbol][
                        'models'])
                else:
                    models = stoMod.create_models(targets)
            else:
                models = stoMod.create_models(targets)
        else:
            models = stoMod.create_models(targets)

        return models

    def train_models_on_stock_data(self, iStockSymbol, iFromDate=None, iRetrain=False):
        """

        :param iStockSymbol:
        :param iFromDate:
        :return:
        """
        # Steps : check if data is usable / featurize / compute targets / train models / save them !
        logging.info('*' * 50)
        logging.info('Train model for stock ' + iStockSymbol + ' : ')

        if ((not os.path.exists(self.Paths['ModelsPath'] + iStockSymbol + '/')) | (iRetrain)):

            if self.DataHand.is_usable(iStockSymbol, iFromDate):
                # Get data if usable
                df = self.DataHand.get_data(iStockSymbol)

                # Featurize
                df_feat, features = self.featurize_stock_data(df, iStockSymbol, iFromDate)
                if iFromDate is not None:
                    logging.info('Data was filtered with date >= ' + iFromDate + ' and featurized')
                else:
                    logging.info('Data for stock ' + iStockSymbol + ' was featurized.')

                # Computes targets and filter - Check if parameters are there
                df_train, targets = self.compute_targets_on_stock_data(df_feat, iStockSymbol, iFromDate)
                logging.info('Target data for stock ' + iStockSymbol + ' were computed.')

                # Create models (GridSearchCV RFC & GBC)
                models = self.create_models_for_stock(iStockSymbol, targets)

                # Fit models
                fitted_models = stoMod.fit_models(models, df_train.dropna(), features, targets)
                logging.info('Models for stock ' + iStockSymbol + ' were trained.')

            else:
                logging.warning(
                    'Data for stock ' + iStockSymbol + ' is not usable. Models were not trained. Please check.')
                return None

        else:
            logging.info(
                'Models for stock ' + iStockSymbol + ' were already trained. No update were done. Use iRetrain if needed.')
            return None

        return fitted_models

    def save_fitted_models(self, fitted_models, iStockSymbol):
        # Save models
        self.create_models_path(iStockSymbol)

        if self.Params.has_key('stockparams'):
            if self.Params['stockparams'].has_key(iStockSymbol):
                stoMod.save_models(fitted_models, self.Paths['ModelsPath'] + iStockSymbol + '/',
                                   dt.date.today().strftime(format='%Y-%m-%d'),
                                   self.Params['stockparams'][iStockSymbol])
            else:
                stoMod.save_models(fitted_models, self.Paths['ModelsPath'] + iStockSymbol + '/',
                                   dt.date.today().strftime(format='%Y-%m-%d'))
        else:
            stoMod.save_models(fitted_models, self.Paths['ModelsPath'] + iStockSymbol + '/',
                               dt.date.today().strftime(format='%Y-%m-%d'))

        logging.info('Models for stock ' + iStockSymbol + ' were saved.')
        return

    def train_and_save_models_on_stock(self, iStockSymbol, iFromDate=None, iRetrain=False):
        fitted_models = self.train_models_on_stock_data(iStockSymbol, iFromDate=iFromDate, iRetrain=iRetrain)
        if fitted_models is not None:
            self.save_fitted_models(fitted_models, iStockSymbol)
        return

    def train_and_save_models(self, iFromDate=None, iRetrain=False):
        for stock in self.Params['stocklist']:
            # logging.info('Training model for STOCK : ' + stock)
            self.train_and_save_models_on_stock(stock, iFromDate, iRetrain)
        return

    def load_stock_models(self, iStockSymbol, iDate=None):
        """

        :param iStockSymbol:
        :param iDate:
        :return:
        """
        return stoMod.load_models(self.Paths['ModelsPath'] + iStockSymbol, iDate)

    def apply_models_on_data(self, iStockSymbol, iFromDate=None, iModelDate=None):
        """

        :param iStockSymbol:
        :param iFromDate:
        :param iModelDate:
        :return:
        """
        if self.DataHand.is_usable(iStockSymbol, iFromDate):
            # Get data if usable
            df = self.DataHand.get_data(iStockSymbol)

            modelsAndParams = self.load_stock_models(iStockSymbol, iModelDate)
            # Check if specific config is there or not
            if modelsAndParams[1] is None:
                pass
            else:
                self.Params['stockparams'][iStockSymbol] = modelsAndParams[1]

            df_feat, features = self.featurize_stock_data(df, iStockSymbol, iFromDate)

            if modelsAndParams[0] is not None:
                targets = set([target.replace('_gbr', '').replace('_rf', '') for target in modelsAndParams[0].keys()])
            else:
                return None, None

            df_pred = stoMod.apply_models(modelsAndParams[0], df_feat, features, targets)
        else:
            if iFromDate is not None:
                logging.warning(
                    'Data for stock ' + iStockSymbol + ' and date >= ' + iFromDate + ' was not usable. Please check')
            else:
                logging.warning(
                    'Data for stock ' + iStockSymbol + ' was not usable. Please check.')
            return None, None

        return df_pred, targets

    def get_predictions_on_stocklist(self, iFromDate=None, iModelDate=None, iNumDays=1):
        """

        :param iFromDate:
        :param iModelDate:
        :param iNumDays:
        :return:
        """
        results_per_stock = []
        targets_final = []
        for stock in self.Params['stocklist']:

            current_stock_df, targets = self.apply_models_on_data(stock, iFromDate=iFromDate, iModelDate=iModelDate)
            if current_stock_df is not None:
                results_per_stock.append(current_stock_df.tail(iNumDays))
                targets_final = [target + '_p' for target in targets] + [target + '_ps' for target in targets]
                logging.info('Prediction for stock ' + stock + ' computed.')
            else:
                logging.warning('Prediction for stock ' + stock + ' NOT computed. Please check.')

        oDf = pd.concat(results_per_stock).loc[:, ['stock', 'date'] + targets_final]

        # Filter out rows with all targets at 0
        oDf = oDf[oDf[targets_final].values.sum(axis=1) != 0]

        if oDf.shape[0] == 0:
            logging.warning('No move to do today.')
            return None
        else:
            return oDf

    def save_predictions_on_stocklist(self, iFromDate=None, iModelDate=None, iNumDays=1):
        self.create_results_path()
        pred_df = self.get_predictions_on_stocklist(iFromDate, iModelDate, iNumDays)
        if pred_df is not None:
            pred_df.to_csv(
                self.Paths['ResultsPath'] + 'results_' + dt.date.today().strftime(format='%Y-%m-%d') + '.csv',
                sep=';', index=False)
        return

    def get_last_predictions_on_stocklist(self):
        if os.path.exists(self.Paths['ResultsPath']):
            file_list = glob.glob(self.Paths['ResultsPath'] + '*.csv')
            if len(file_list) > 0:
                file_list = [file.replace(self.Paths['ResultsPath'] + 'results_', '').replace('.csv', '') for file in
                             file_list]
                data_list = sorted([pd.to_datetime(file, format='%Y-%m-%d').date() for file in file_list], reverse=True)
                last_date_folder = data_list[0]
                return last_date_folder
            else:
                return None
        else:
            return None

    def daily_run(self, iSleepRange, iTrainingFromDate=None, iModelDate=None, iRetrain=False):
        """

        :param iSleepRange:
        :param iFromDate:
        :param iModelDate:
        :param iNumDays:
        :return:
        """

        # current day
        current_day = dt.date.today()

        # Save stock data
        # self.DataHand.save_all_stocks(iSleepRange)

        # Train models if needed
        self.train_and_save_models(iTrainingFromDate, iRetrain)

        # Check last predictions
        last_pred = self.get_last_predictions_on_stocklist()
        if last_pred is not None:
            num_days = (current_day - last_pred).days
            if num_days == 0:
                num_days = 1
        else:
            num_days = 1

        iFromDate = (current_day - dt.timedelta(days=num_days)).strftime(format='%Y-%m-%d')

        # Compute predictions for days not since last prediction
        self.save_predictions_on_stocklist(iFromDate, iModelDate, iNumDays=num_days)

        return
