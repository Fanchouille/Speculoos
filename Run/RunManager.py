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
from Models import DeepModels as dl

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

    def create_results_path(self, iType):
        if not os.path.exists(self.Paths['ResultsPath'] + iType):
            ph.create_path(self.Paths['ResultsPath'] + iType)
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

    def compute_targets_on_stock_data(self, df_feat, iStockSymbol, iFromDate, iType='classifier'):
        if iType in ['classifier', 'deep']:
            if self.Params.has_key('stockparams'):
                if self.Params['stockparams'].has_key(iStockSymbol):
                    if self.Params['stockparams'][iStockSymbol].has_key('targets'):
                        df_train, targets = dfeat.compute_classifier_target(df_feat, iFromDate,
                                                                            **self.Params['stockparams'][iStockSymbol][
                                                                                'targets'])
                    else:
                        df_train, targets = dfeat.compute_classifier_target(df_feat, iFromDate)
                else:
                    df_train, targets = dfeat.compute_classifier_target(df_feat, iFromDate)
            else:
                df_train, targets = dfeat.compute_classifier_target(df_feat, iFromDate)

        elif iType == 'regressor':
            if self.Params.has_key('stockparams'):
                if self.Params['stockparams'].has_key(iStockSymbol):
                    if self.Params['stockparams'][iStockSymbol].has_key('targets'):
                        df_train, targets = dfeat.compute_regressor_target(df_feat, iFromDate,
                                                                           **self.Params['stockparams'][iStockSymbol][
                                                                               'targets'])
                    else:
                        df_train, targets = dfeat.compute_regressor_target(df_feat, iFromDate)
                else:
                    df_train, targets = dfeat.compute_regressor_target(df_feat, iFromDate)
            else:
                df_train, targets = dfeat.compute_regressor_target(df_feat, iFromDate)
        return df_train, targets

    def create_models_for_stock(self, iStockSymbol, targets, iType='classifier'):
        if iType == 'classifier':
            if self.Params.has_key('stockparams'):
                if self.Params['stockparams'].has_key(iStockSymbol):
                    if self.Params['stockparams'][iStockSymbol].has_key('models'):
                        models = stoMod.create_classifier_models(targets, **self.Params['stockparams'][iStockSymbol][
                            'models'])
                    else:
                        models = stoMod.create_classifier_models(targets)
                else:
                    models = stoMod.create_classifier_models(targets)
            else:
                models = stoMod.create_classifier_models(targets)

        elif iType == 'regressor':
            if self.Params.has_key('stockparams'):
                if self.Params['stockparams'].has_key(iStockSymbol):
                    if self.Params['stockparams'][iStockSymbol].has_key('models'):
                        models = stoMod.create_classifier_models(targets, **self.Params['stockparams'][iStockSymbol][
                            'models'])
                    else:
                        models = stoMod.create_regressor_models(targets)
                else:
                    models = stoMod.create_regressor_models(targets)
            else:
                models = stoMod.create_regressor_models(targets)

        return models

    def compute_targets_duration_on_stock_data(self, iStockSymbol, iFromDate=None):
        """

        :param iStockSymbol:
        :param iFromDate:
        :return:
        """

        if self.DataHand.is_usable(iStockSymbol, iFromDate):
            # Get data if usable
            df = self.DataHand.get_data(iStockSymbol)
            # Featurize
            df_feat, features = self.featurize_stock_data(df, iStockSymbol, iFromDate)

            # Computes targets and filter - Check if parameters are there
            df_train, targets = self.compute_targets_on_stock_data(df_feat, iStockSymbol, iFromDate)
            return df_train, targets
        return None, None

    def train_models_on_stock_data(self, iStockSymbol, iFromDate=None, iRetrain=False, iType='classifier'):
        """

        :param iStockSymbol:
        :param iFromDate:
        :return:
        """
        # Steps : check if data is usable / featurize / compute targets / train models / save them !
        logging.info('*' * 50)
        logging.info('Train model for stock ' + iStockSymbol + ' : ')

        if ((not os.path.exists(self.Paths['ModelsPath'] + iStockSymbol + '/' + iType + '/')) | (iRetrain)):

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
                df_train, targets = self.compute_targets_on_stock_data(df_feat, iStockSymbol, iFromDate, iType)
                logging.info('Target data for stock ' + iStockSymbol + ' were computed.')

                # Create models (GridSearchCV RFC & GBC)
                if iType in ['classifier', 'regressor']:
                    # 2 steps here : creation and fitting
                    models = self.create_models_for_stock(iStockSymbol, targets, iType)
                    # Fit models
                    fitted_models = stoMod.fit_models(models, df_train.dropna(), features, targets)
                elif iType == 'deep':
                    # here fit also creates the model
                    fitted_models = dl.fit_models(df_train.dropna(), features, targets)
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

    def save_fitted_models(self, fitted_models, iStockSymbol, iType='classifier'):
        # Save models
        self.create_models_path(iStockSymbol)

        if iType in ['classifier', 'regressor']:
            if self.Params.has_key('stockparams'):
                if self.Params['stockparams'].has_key(iStockSymbol):
                    stoMod.save_models(fitted_models, self.Paths['ModelsPath'] + iStockSymbol + '/',
                                       dt.date.today().strftime(format='%Y-%m-%d'), iType,
                                       self.Params['stockparams'][iStockSymbol])
                else:
                    stoMod.save_models(fitted_models, self.Paths['ModelsPath'] + iStockSymbol + '/',
                                       dt.date.today().strftime(format='%Y-%m-%d'), iType)
            else:
                stoMod.save_models(fitted_models, self.Paths['ModelsPath'] + iStockSymbol + '/',
                                   dt.date.today().strftime(format='%Y-%m-%d'), iType)

            logging.info('Models for stock ' + iStockSymbol + ' were saved.')
        elif iType == 'deep':
            # If deep we have as scaler in 1st place
            dl.save_models(fitted_models[1], fitted_models[0], self.Paths['ModelsPath'] + iStockSymbol + '/',
                           dt.date.today().strftime(format='%Y-%m-%d'), iType)
        return

    def train_and_save_models_on_stock(self, iStockSymbol, iFromDate=None, iRetrain=False, iType='classifier'):
        fitted_models = self.train_models_on_stock_data(iStockSymbol, iFromDate=iFromDate, iRetrain=iRetrain,
                                                        iType=iType)

        if fitted_models is not None:
            self.save_fitted_models(fitted_models, iStockSymbol, iType)
        return

    def train_and_save_models(self, iFromDate=None, iRetrain=False, iType='classifier'):
        for stock in self.Params['stocklist']:
            # print 'Training Stock : ' + stock + ' with supervised model type : ' + iType
            # logging.info('Training model for STOCK : ' + stock)
            self.train_and_save_models_on_stock(stock, iFromDate, iRetrain, iType)
        return

    def load_stock_models(self, iStockSymbol, iDate=None, iType='classifier'):
        """

        :param iStockSymbol:
        :param iDate:
        :return:
        """
        return stoMod.load_models(self.Paths['ModelsPath'] + iStockSymbol, iDate, iType)

    def apply_models_on_data(self, iStockSymbol, iFromDate=None, iModelDate=None, iType='classifier'):
        """

        :param iStockSymbol:
        :param iFromDate:
        :param iModelDate:
        :return:
        """
        if self.DataHand.is_up_to_date(iStockSymbol):
            # Get data if usable
            df = self.DataHand.get_data(iStockSymbol)

            if iType in ['classifier', 'regressor']:
                modelsAndParams = self.load_stock_models(iStockSymbol, iModelDate, iType)
                # Check if specific config is there or not
                if modelsAndParams[1] is None:
                    pass
                else:
                    self.Params['stockparams'][iStockSymbol] = modelsAndParams[1]
                if modelsAndParams[0] is not None:
                    targets = set(
                        [target.replace('_gbr', '').replace('_rf', '') for target in modelsAndParams[0].keys()])
                else:
                    return None, None
            elif iType == 'deep':
                lModelDict, lScaler = dl.load_models(self.Paths['ModelsPath'] + iStockSymbol, iModelDate, iType='deep')
                if lModelDict is not None:
                    targets = set(
                        [target.replace('_deep', '') for target in lModelDict.keys()])
                else:
                    return None, None

            df_feat, features = self.featurize_stock_data(df, iStockSymbol, iFromDate)
            # print features

            if iType == 'classifier':
                df_pred = stoMod.apply_classifier_models(modelsAndParams[0], df_feat, features, targets)
            elif iType == 'regressor':
                df_pred = stoMod.apply_regressor_models(modelsAndParams[0], df_feat, features, targets)
            elif iType == 'deep':
                df_pred = dl.apply_classifier_models(lModelDict, lScaler, df_feat, features, targets)

        else:
            if iFromDate is not None:
                logging.warning(
                    'Data for stock ' + iStockSymbol + ' and date >= ' + iFromDate + ' was not usable. Please check')
            else:
                logging.warning(
                    'Data for stock ' + iStockSymbol + ' was not usable. Please check.')
            return None, None
        return df_pred, targets

    def get_predictions_on_stock(self, iStockSymbol, iModelDate=None, iNumDays=1, iType='classifier'):
        """

        :param iFromDate:
        :param iModelDate:
        :param iNumDays:
        :return:
        """
        results_per_stock = []
        targets_final = []

        iFromDate = (dt.date.today() - dt.timedelta(days=iNumDays)).strftime(format='%Y-%m-%d')
        current_stock_df, targets = self.apply_models_on_data(iStockSymbol, iFromDate=iFromDate, iModelDate=iModelDate,
                                                              iType=iType)

        if current_stock_df is not None:
            results_per_stock.append(current_stock_df.tail(iNumDays))

            if iType == 'classifier':
                targets_final = [target + '_p' for target in targets] + [target + '_ps' for target in targets]
            elif iType == 'deep':
                targets_final = [target + '_deep_p' for target in targets]
            elif iType == 'regressor':
                targets_final = [target + '_rf' for target in targets] + [target + '_gbr' for target in targets]

        logging.info('Prediction for stock ' + iStockSymbol + ' computed.')

        if len(results_per_stock) > 0:
            #
            # print len(results_per_stock)
            # print ['stock', 'date', 'close'] + targets_final
            oDf = pd.concat(results_per_stock).loc[:, ['stock', 'date', 'close'] + targets_final]
        else:
            return None

        if iType in ['classifier', 'deep']:
            # Filter out rows with all targets at 0
            oDf = oDf[oDf[targets_final].values.sum(axis=1) != 0]

        if oDf.shape[0] == 0:
            logging.warning('No move to do today.')
            return None
        else:
            return oDf.dropna()

    def get_predictions_on_stocklist(self, iFromDate=None, iModelDate=None, iNumDays=1, iType='classifier'):
        """

        :param iFromDate:
        :param iModelDate:
        :param iNumDays:
        :return:
        """
        results_per_stock = []
        targets_final = []
        for stock in self.Params['stocklist']:
            try:
                current_stock_df, targets = self.apply_models_on_data(stock, iFromDate=iFromDate, iModelDate=iModelDate,
                                                                      iType=iType)
                if current_stock_df is not None:
                    results_per_stock.append(current_stock_df.tail(iNumDays))

                    if iType == 'classifier':
                        targets_final = [target + '_p' for target in targets] + [target + '_ps' for target in
                                                                                 targets] + [target + '_final' for
                                                                                             target
                                                                                             in targets]
                    elif iType == 'deep':
                        targets_final = [target + '_deep_p' for target in targets] + [target + '_final' for target
                                                                                      in targets]
                    elif iType == 'regressor':
                        targets_final = [target + '_rf' for target in targets] + [target + '_gbr' for target in targets]

            except:
                print "Error on stock : " + stock

            logging.info('Prediction for stock ' + stock + ' computed.')

        if len(results_per_stock) > 0:
            #
            # print len(results_per_stock)
            # print ['stock', 'date', 'close'] + targets_final
            oDf = pd.concat(results_per_stock).loc[:, ['stock', 'date', 'close'] + targets_final]

            # Fetch data only for B moves (buy) before ABC advise
            moveStockList = oDf[oDf[[target for target in targets_final if
                                     ('B' in target) & ('final' not in target)]].values.sum(axis=1) != 0].loc[:,
                            'stock'].unique()
            abcDf = self.DataHand.get_stock_info_from_stocklist([stock.replace('.PA', '') for stock in moveStockList],
                                                                dt.date.today())


        else:
            return None

        if oDf.shape[0] == 0:
            logging.warning('No move to do today.')
            return None
        else:
            results = pd.merge(oDf[oDf[[target for target in targets_final if
                                        ('B' in target) & ('final' not in target)]].values.sum(axis=1) != 0], abcDf,
                               how='left', on=['stock'])
            # print results.shape

            if iType in ['classifier']:
                results.loc[:, 'B_TARGET_final'] = results.apply(
                    lambda x: 1 if ((x['B_TARGET_ps'] == 1) & (x['Conseil'] == 'Achat')) else 0, axis=1).values
                # Filter out rows with all targets at 0
                results = results[results[targets_final].values.sum(axis=1) != 0].sort_values(
                    ['B_TARGET_final', 'close'], ascending=[0, 1])
                for target in targets_final:
                    results.loc[:, target] = results.loc[:, target].fillna(0).astype(int)
            if iType in ['deep']:
                results.loc[:, 'B_TARGET_final'] = results.apply(
                    lambda x: 1 if ((x['B_TARGET_deep_p'] == 1) & (x['Conseil'] == 'Achat')) else 0, axis=1).values
                # Filter out rows with all targets at 0
                results = results[results[targets_final].values.sum(axis=1) != 0].sort_values(
                    ['B_TARGET_final', 'close'], ascending=[0, 1])
                for target in targets_final:
                    results.loc[:, target] = results.loc[:, target].fillna(0).astype(int)

            return results

    def save_predictions_on_stocklist(self, iFromDate=None, iModelDate=None, iNumDays=1, iType='classifier'):
        self.create_results_path(iType)
        pred_df = self.get_predictions_on_stocklist(iFromDate, iModelDate, iNumDays, iType)
        if pred_df is not None:
            pred_df.to_csv(
                self.Paths['ResultsPath'] + iType + '/' + 'results_' + dt.date.today().strftime(
                    format='%Y-%m-%d') + '.csv',
                sep=';', index=False, decimal=',', encoding='utf-8')
        else:
            print 'Pred df is empty !'
        return

    def get_last_predictions_on_stocklist(self, iType='classifier'):
        if os.path.exists(self.Paths['ResultsPath'] + iType):
            file_list = glob.glob(self.Paths['ResultsPath'] + iType + '/*.csv')
            if len(file_list) > 0:
                file_list = [file.replace(self.Paths['ResultsPath'] + iType + '/results_', '').replace('.csv', '')
                             for file in
                             file_list]
                data_list = sorted([pd.to_datetime(file, format='%Y-%m-%d').date() for file in file_list], reverse=True)
                if len(data_list) > 0:
                    last_date_folder = data_list[0]
                else:
                    return None
                return last_date_folder
            else:
                return None
        else:
            return None

    def clean_stock_data(self, iStockSymbol):
        self.DataHand.clean_stock_data(iStockSymbol)
        return

    def clean_all_stocks_data(self):
        for stock in self.Params['stocklist']:
            self.clean_stock_data(stock)
        return

    def daily_run(self, iSleepRange=(5, 10), iTrainingFromDate=None, iModelDate=None, iRetrain=False,
                  iType='classifier'):
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
        self.DataHand.save_all_stocks(iSleepRange)

        # Train models if needed
        self.train_and_save_models(iTrainingFromDate, iRetrain, iType)

        # Check last predictions
        last_pred = self.get_last_predictions_on_stocklist(iType)
        if last_pred is not None:
            num_days = (current_day - last_pred).days
            if num_days == 0:
                num_days = 1
        else:
            num_days = 1
        # print num_days

        iFromDate = (current_day - dt.timedelta(days=num_days)).strftime(format='%Y-%m-%d')
        # Compute predictions for days not since last prediction
        self.save_predictions_on_stocklist(iFromDate, iModelDate, iNumDays=num_days, iType=iType)

        return
