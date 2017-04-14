import logging
import os

from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from DataManagement import PathHandler as ph
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

import glob
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')


def create_regressor_models(iTargets, iScoring='neg_mean_absolute_error', iNumFolds=3, iPolyDeg=[1],
                            iPcaVarianceToKeep=None):
    """

    :param iTargets:
    :param iScoring: ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']
    :param iPolyDeg:
    :param iNumFolds: number of TimeSeriesSplit folds for GridSearchCV
    :param iPcaVarianceToKeep:
    :return:
    """

    gbr = GradientBoostingRegressor(random_state=123)
    rf = RandomForestRegressor(random_state=123)
    poly_ext = PolynomialFeatures(interaction_only=True, include_bias=False)

    if iPcaVarianceToKeep is not None:  # Add Standard Scaler + ACP (use if too long to run)
        scaler = StandardScaler()
        pca = PCA(n_components=iPcaVarianceToKeep)
        pipe_gbr = Pipeline([('poly', poly_ext),
                             ('scaler', scaler),
                             ('pca', pca),
                             ('gbr', gbr)])

        pipe_rf = Pipeline([('poly', poly_ext),
                            ('scaler', scaler),
                            ('pca', pca),
                            ('rf', rf)])

    else:
        params_gbr = {'poly__degree': iPolyDeg,
                      'gbr__n_estimators': [50, 100, 300],
                      'gbr__max_depth': [3, 6, 10],
                      'gbr__learning_rate': [0.1, 0.05],
                      'gbr__subsample': [1.0, 0.8]
                      }

        params_rf = {'poly__degree': iPolyDeg,
                     'rf__n_estimators': [50, 100, 300],
                     'rf__min_samples_leaf': [5, 10, 25],
                     'rf__max_depth': [3, 6, 10],
                     'rf__max_features': [1.0, 0.8]
                     }

        pipe_gbr = Pipeline([('poly', poly_ext), ('gbr', gbr)])

        pipe_rf = Pipeline([('poly', poly_ext), ('rf', rf)])

    model_dict = {}
    for target in iTargets:
        grid_gbr = GridSearchCV(pipe_gbr,
                                param_grid=params_gbr,
                                cv=iNumFolds,
                                scoring=iScoring,
                                iid=False,
                                n_jobs=-1,
                                )

        model_dict[target + '_gbr'] = grid_gbr

        grid_rf = GridSearchCV(pipe_rf,
                               param_grid=params_rf,
                               cv=iNumFolds,
                               scoring=iScoring,
                               iid=False,
                               n_jobs=-1,
                               )

        model_dict[target + '_rf'] = grid_rf

    return model_dict


def create_classifier_models(iTargets, iScoring='f1_weighted', iNumFolds=3, iPolyDeg=[1], iPcaVarianceToKeep=None):
    """

    :param iTargets:
    :param iScoring: ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']
    :param iPolyDeg:
    :param iNumFolds: number of TimeSeriesSplit folds for GridSearchCV
    :param iPcaVarianceToKeep:
    :return:
    """

    gbr = GradientBoostingClassifier(random_state=123)
    rf = RandomForestClassifier(random_state=123)
    poly_ext = PolynomialFeatures(interaction_only=True, include_bias=False)

    if iPcaVarianceToKeep is not None:  # Add Standard Scaler + ACP (use if too long to run)
        scaler = StandardScaler()
        pca = PCA(n_components=iPcaVarianceToKeep)
        pipe_gbr = Pipeline([('poly', poly_ext),
                             ('scaler', scaler),
                             ('pca', pca),
                             ('gbr', gbr)])

        pipe_rf = Pipeline([('poly', poly_ext),
                            ('scaler', scaler),
                            ('pca', pca),
                            ('rf', rf)])

    else:
        params_gbr = {'poly__degree': iPolyDeg,
                      'gbr__n_estimators': [50, 100, 300],
                      'gbr__max_depth': [3, 6, 10],
                      'gbr__learning_rate': [0.1, 0.05],
                      'gbr__subsample': [1.0, 0.8]
                      }

        params_rf = {'poly__degree': iPolyDeg,
                     'rf__n_estimators': [50, 100, 300],
                     'rf__min_samples_leaf': [5, 10, 25],
                     'rf__max_depth': [3, 6, 10],
                     'rf__max_features': [1.0, 0.8]
                     }

        pipe_gbr = Pipeline([('poly', poly_ext), ('gbr', gbr)])

        pipe_rf = Pipeline([('poly', poly_ext), ('rf', rf)])

    model_dict = {}
    for target in iTargets:
        grid_gbr = GridSearchCV(pipe_gbr,
                                param_grid=params_gbr,
                                cv=iNumFolds,
                                scoring=iScoring,
                                iid=False,
                                n_jobs=-1,
                                )

        model_dict[target + '_gbr'] = grid_gbr

        grid_rf = GridSearchCV(pipe_rf,
                               param_grid=params_rf,
                               cv=iNumFolds,
                               scoring=iScoring,
                               iid=False,
                               n_jobs=-1,
                               )

        model_dict[target + '_rf'] = grid_rf

    return model_dict


def fit_models(iModelDict, iDf, features, targets):
    """

    :param iModelDict:
    :param iDf:
    :param features:
    :param targets:
    :return:
    """
    oFittedModelsDict = {}
    # print targets
    for target in targets:

        # print target
        # print iDf.loc[:, target].value_counts().shape[0]

        if iDf.loc[:, target].value_counts().shape[0] > 1:
            oFittedModelsDict[target + '_gbr'] = iModelDict[target + '_gbr'].fit(iDf.loc[:, features].values,
                                                                                 iDf.loc[:, target].values)
            oFittedModelsDict[target + '_rf'] = iModelDict[target + '_rf'].fit(iDf.loc[:, features].values,
                                                                               iDf.loc[:, target].values)
    return oFittedModelsDict


def apply_classifier_models(iModelDict, iDf, features, targets):
    """

    :param iModelDict:
    :param iDf:
    :param features:
    :param targets:
    :return:
    """
    for target in targets:
        iDf.loc[:, target + '_p'] = iModelDict[target + '_gbr'].predict(iDf.loc[:, features].values) + \
                                    iModelDict[target + '_rf'].predict(iDf.loc[:, features].values)
        iDf.loc[:, target + '_ps'] = iDf.loc[:, target + '_p'].map(lambda x: 1 if x == 2 else 0)
        iDf.loc[:, target + '_p'] = iDf.loc[:, target + '_p'].map(lambda x: 1 if x > 0 else 0)
    return iDf


def apply_regressor_models(iModelDict, iDf, features, targets):
    """

    :param iModelDict:
    :param iDf:
    :param features:
    :param targets:
    :return:
    """
    for target in targets:
        iDf.loc[:, target + '_rf'] = iModelDict[target + '_rf'].predict(iDf.loc[:, features].values)
        iDf.loc[:, target + '_gbr'] = iModelDict[target + '_gbr'].predict(iDf.loc[:, features].values)

        iDf.loc[:, target + '_pmean'] = (iModelDict[target + '_gbr'].predict(iDf.loc[:, features].values) + \
                                         iModelDict[target + '_rf'].predict(iDf.loc[:, features].values)) / 2

    return iDf


def save_models(iModelDict, iModelPath, iDate, iType='classifier', iParams=None):
    """

    :param iModelDict:
    :param iModelPath:
    :param iDate:
    :param iParams:
    :return:
    """
    if os.path.exists(iModelPath):
        if not os.path.exists(iModelPath + '/' + iType + '/' + iDate + '/'):
            ph.create_path(iModelPath + '/' + iType + '/' + iDate + '/')
        for model_name in iModelDict.keys():
            model = iModelDict[model_name]
            joblib.dump(model, iModelPath + '/' + iType + '/' + iDate + '/' + model_name + '.pkl')
            if iParams is not None:
                joblib.dump(iParams, iModelPath + '/' + iType + '/' + iDate + '/' + 'pipeline_params.pkl')
            else:
                if os.path.exists(iModelPath + '/' + iType + '/' + iDate + '/' + 'pipeline_params.pkl'):
                    os.remove(iModelPath + '/' + iType + '/' + iDate + '/' + 'pipeline_params.pkl')

    return


def load_models(iModelPath, iDate=None, iType='classifier'):
    """

    :param iModelPath:
    :param iDate:
    :return:
    """
    if os.path.exists(iModelPath + '/' + iType):
        if iDate is None:
            file_list = glob.glob(iModelPath + '/' + iType + '/*')
            file_list = [file.replace(iModelPath + '/' + iType + '/', '') for file in file_list]
            data_list = sorted([pd.to_datetime(file, format='%Y-%m-%d').date() for file in file_list], reverse=True)
            # print data_list
            last_date_folder = data_list[0].strftime(format='%Y-%m-%d')
        else:
            last_date_folder = iDate

        file_list = glob.glob(iModelPath + '/' + iType + '/' + last_date_folder + '/*.pkl')
        file_list = [file.replace(iModelPath + '/' + iType + '/' + last_date_folder + '/', '') for file in file_list]

        if 'pipeline_params.pkl' in file_list:
            pipeline_params = joblib.load(iModelPath + '/' + iType + '/' + last_date_folder + '/pipeline_params.pkl')
            file_list = [file for file in file_list if file != 'pipeline_params.pkl']
        else:
            pipeline_params = None

        oModelDict = {}

        for file in file_list:
            oModelDict[file.replace('.pkl', '')] = joblib.load(
                iModelPath + '/' + iType + '/' + last_date_folder + '/' + file)
    else:
        return None, None
    return oModelDict, pipeline_params


def calibrate_proba_fitted_models(iDf, iFeatures, iModelsDict):
    iCalibratedModelsDict = {}

    for model_name in iModelsDict.keys():
        target = model_name.replace('_gbr', '').replace('_rf', '')
        proba_cal_sig = CalibratedClassifierCV(iModelsDict[model_name], method='sigmoid', cv='prefit')
        proba_cal_iso = CalibratedClassifierCV(iModelsDict[model_name], method='isotonic', cv='prefit')
        proba_cal_sig.fit(iDf.loc[:, iFeatures.values], iDf.loc[:, target].values)
        proba_cal_iso.fit(iDf.loc[:, iFeatures.values], iDf.loc[:, target].values)
        brier_sig = brier_score_loss(iDf.loc[:, target].value,
                                     proba_cal_sig.predict_proba(iDf.loc[:, iFeatures.values])[:, 1])
        brier_iso = brier_score_loss(iDf.loc[:, target].value,
                                     proba_cal_iso.predict_proba(iDf.loc[:, iFeatures.values])[:, 1])

        if brier_sig <= brier_iso:
            iCalibratedModelsDict[model_name] = proba_cal_sig.calibrated_classifiers_
        else:
            iCalibratedModelsDict[model_name] = proba_cal_iso.calibrated_classifiers_
    return iCalibratedModelsDict
