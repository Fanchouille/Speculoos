import logging
import os

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from DataManagement import PathHandler as ph
from sklearn.preprocessing import StandardScaler
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model

import glob

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')

batch_size = [10]  # [10, 50]
epochs = [50]
optimizer = ['Adam']  # possibilities ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
activation = ['relu',
              'linear']  # ['relu','linear']  # possibilities ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

dropout_rate = [0.5]
neurons = [5, 10, 20]

DEFAULTPARAMGRID = dict(iOptimizer=optimizer,
                        iActivation=activation,
                        iDropoutRate=dropout_rate,
                        iNumNeurons=neurons,
                        batch_size=batch_size, epochs=epochs)


def construct_deep_model(iNumNeurons=5, iActivation='linear', iDropoutRate=0.0,
                         iOptimizer='adam'):
    """

    :param iNumFeatures: number of features
    :param iNumNeurons: number of neurons in hidden layer
    :param iActivation: activation functions
    :param iDropoutRate: dropout rate to reduce overfitting
    :param iInitMode: initialization of weights method
    :param iOptimizer: optimizer used
    :return:
    """

    model = Sequential()

    model.add(Dense(iNumNeurons, input_shape=(96,), kernel_initializer='uniform', activation=iActivation))
    model.add(Dropout(iDropoutRate))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=iOptimizer, metrics=['accuracy'])

    return model


def construct_deep_gridsearch(iParamGrid=DEFAULTPARAMGRID, iNumFolds=3, iScoring='f1_weighted'):
    model = KerasClassifier(build_fn=construct_deep_model, verbose=0)
    oGrid = GridSearchCV(estimator=model, cv=iNumFolds, param_grid=iParamGrid, scoring=iScoring, n_jobs=1)
    return oGrid


def fit_models(iDf, features, targets):
    """
    :param iDf:
    :param features:
    :param targets:
    :return:
    """
    oFittedModelsDict = {}
    scaler = StandardScaler()
    X = scaler.fit_transform(iDf.loc[:, features].astype(float).values)

    for target in targets:
        # print target
        # print iDf.loc[:, target].value_counts().shape[0]
        lGrid = construct_deep_gridsearch()

        if iDf.loc[:, target].value_counts().shape[0] > 1:
            lGrid.fit(X, iDf.loc[:, target].astype(int).values)
            print lGrid.best_params_
            print lGrid.best_score_
            oFittedModelsDict[target + '_deep'] = lGrid.best_estimator_.model

    return scaler, oFittedModelsDict


def apply_classifier_models(iModelDict, iScaler, iDf, features, targets):
    """

    :param iModelDict:
    :param iDf:
    :param features:
    :param targets:
    :return:
    """
    X = iScaler.transform(iDf.loc[:, features].astype(float).values)
    for target in targets:
        iDf.loc[:, target + '_deep_p'] = iModelDict[target + '_deep'].predict(X)
    return iDf


def save_models(iModelDict, iScaler, iModelPath, iDate, iType='deep'):
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
            joblib.dump(iScaler, iModelPath + '/' + iType + '/' + iDate + '/scaler.pkl')
            for model_name in iModelDict.keys():
                model = iModelDict[model_name]
                model.save(iModelPath + '/' + iType + '/' + iDate + '/' + model_name + '.h5')

    return


def load_models(iModelPath, iDate=None, iType='deep'):
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

        file_list = glob.glob(iModelPath + '/' + iType + '/' + last_date_folder + '/*.h5')
        file_list = [file.replace(iModelPath + '/' + iType + '/' + last_date_folder + '/', '') for file in file_list]

        oModelDict = {}

        for file in file_list:
            # Deep model file
            if 'h5' in file:
                oModelDict[file.replace('.h5', '')] = load_model(file)
            # Standard scaler file
            elif 'pkl' in file:
                scaler = joblib.load(iModelPath + '/' + iType + '/' + last_date_folder + '/scaler.pkl')

        return oModelDict, scaler
    else:
        return None, None
