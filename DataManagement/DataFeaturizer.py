from talib.abstract import *
import numpy as np
import pandas as pd

import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')

def sma_cross(iDf, iPeriod1, iPeriod2):
    iPeriodmin = min(iPeriod1, iPeriod2)
    iPeriodmax = max(iPeriod1, iPeriod2)
    iDf.loc[:, 'SMA' + str(iPeriodmin)] = SMA(iDf, timeperiod=iPeriodmin)
    iDf.loc[:, 'SMA' + str(iPeriodmax)] = SMA(iDf, timeperiod=iPeriodmax)
    iDf.loc[:, 'DIFF_SMAS' + str(iPeriodmin) + str(iPeriodmax)] = iDf.loc[:, 'SMA' + str(iPeriodmax)] - iDf.loc[:,
                                                                                                        'SMA' + str(
                                                                                                            iPeriodmin)]
    iDf.loc[:, 'DIFF_SMAS' + str(iPeriodmin) + str(iPeriodmax) + '_D_BEF'] = iDf.loc[:,
                                                                             'DIFF_SMAS' + str(iPeriodmin) + str(
                                                                                 iPeriodmax)].shift(1)
    iDf.loc[:, 'SMAS' + str(iPeriodmin) + str(iPeriodmax) + '_B_IND'] = iDf.apply(
        lambda x: 1 if (x['DIFF_SMAS' + str(iPeriodmin) + str(iPeriodmax) + '_D_BEF'] < 0) & (
            x['DIFF_SMAS' + str(iPeriodmin) + str(iPeriodmax)] >= 0) else np.nan, axis=1).fillna(0)
    iDf.loc[:, 'SMAS' + str(iPeriodmin) + str(iPeriodmax) + '_S_IND'] = iDf.apply(
        lambda x: 1 if (x['DIFF_SMAS' + str(iPeriodmin) + str(iPeriodmax) + '_D_BEF'] > 0) & (
            x['DIFF_SMAS' + str(iPeriodmin) + str(iPeriodmax)] <= 0) else np.nan, axis=1).fillna(0)
    # iDf.loc[:,'SMAS'+str(iPeriodmin)+str(iPeriodmax)+'_B_IND']= iDf.apply(lambda x: x['close'] if (x['SMAS'+str(iPeriodmin)+str(iPeriodmax)+'_B_IND']==1) else np.nan,axis=1)
    # iDf.loc[:,'SMAS'+str(iPeriodmin)+str(iPeriodmax)+'_S_IND']= iDf.apply(lambda x: x['close'] if (x['SMAS'+str(iPeriodmin)+str(iPeriodmax)+'_S_IND']==1) else np.nan,axis=1)


    iDf.drop(['SMA' + str(iPeriodmin), 'SMA' + str(iPeriodmax), 'DIFF_SMAS' + str(iPeriodmin) + str(iPeriodmax),
              'DIFF_SMAS' + str(iPeriodmin) + str(iPeriodmax) + '_D_BEF'], axis=1, inplace=True)
    return iDf


def mom_macd(iDf, iFastPeriod, iSlowPeriod, iSignalPeriod, iThresPeriod):
    # Adds MACD ind : difference between 2 EMA of slow and fast period
    # Adds Signal curve : EMA of signal period
    # Adds MACD hist : difference between macd and signal curve
    if 'macd' in iDf.columns.values:
        iDf.drop(['macd', 'macdsignal', 'macdhist', 'PPO'], axis=1, inplace=True)
    iDf = pd.concat([iDf, MACD(iDf, iFastPeriod, iSlowPeriod, iSignalPeriod)], axis=1)
    iDf['macdhist_sign'] = iDf['macdhist'].map(lambda x: 1 if x >= 0 else -1)
    # Check change of sign
    grouper = (iDf['macdhist_sign'] != iDf['macdhist_sign'].shift(1)).cumsum()
    iDf.loc[:, 'macdhist_counts'] = np.abs(iDf.groupby(grouper)['macdhist_sign'].cumsum())
    iDf.loc[:, 'MACD_B_IND'] = ((iDf.loc[:, 'macdhist_counts'].shift() >= iThresPeriod - 1) & (
        iDf.loc[:, 'macdhist'] >= 0) & (iDf.loc[:, 'macdhist'].shift() < 0)) * 1
    # iDf.loc[:,'MACD_B_IND']= iDf.apply(lambda x: x['close'] if (x['MACD_B_IND']==1) else np.nan,axis=1)
    iDf.loc[:, 'MACD_S_IND'] = ((iDf.loc[:, 'macdhist_counts'].shift() >= iThresPeriod - 1) & (
        iDf.loc[:, 'macdhist'] < 0) & (iDf.loc[:, 'macdhist'].shift() >= 0)) * 1
    # iDf.loc[:,'MACD_S_IND']= iDf.apply(lambda x: x['close'] if (x['MACD_S_IND']==1) else np.nan,axis=1)
    iDf.drop(['macdhist_counts', 'macdhist_sign', 'macd', 'macdsignal', 'macdhist'], axis=1, inplace=True)
    iDf.loc[:, 'PPO'] = PPO(iDf, fastperiod=iFastPeriod, slowperiod=iSlowPeriod, matype=0)
    return iDf


def mom_bbands(iDf, iTimePeriod, iStdUp, iStdDwn):
    if 'upperband' in iDf.columns.values:
        iDf.drop(['upperband', 'middleband', 'lowerband'], axis=1, inplace=True)
    iDf = pd.concat([iDf, BBANDS(iDf, timeperiod=iTimePeriod, nbdevup=iStdUp, nbdevdn=iStdDwn, matype=0)], axis=1)
    iDf.loc[:, 'BBAND_B_IND'] = (iDf.loc[:, 'close'] < iDf.loc[:, 'lowerband']) * 1
    # iDf.loc[:,'BBAND_B_IND']= iDf.apply(lambda x: x['close'] if (x['BBAND_B_IND']==1) else np.nan,axis=1)
    iDf.loc[:, 'BBAND_S_IND'] = (iDf.loc[:, 'close'] > iDf.loc[:, 'upperband']) * 1
    # iDf.loc[:,'BBAND_S_IND']= iDf.apply(lambda x: x['close'] if (x['BBAND_S_IND']==1) else np.nan,axis=1)
    iDf.drop(['upperband', 'middleband', 'lowerband'], axis=1, inplace=True)
    return iDf


def chandeliers(iDf):
    iDf['CDL2CROWS'] = CDL2CROWS(iDf)
    iDf['CDL3BLACKCROWS'] = CDL3BLACKCROWS(iDf)
    iDf['CDL3INSIDE'] = CDL3INSIDE(iDf)
    iDf['CDL3LINESTRIKE'] = CDL3LINESTRIKE(iDf)
    iDf['CDL3OUTSIDE'] = CDL3OUTSIDE(iDf)
    iDf['CDL3STARSINSOUTH'] = CDL3STARSINSOUTH(iDf)
    iDf['CDL3WHITESOLDIERS'] = CDL3WHITESOLDIERS(iDf)
    iDf['CDLABANDONEDBABY'] = CDLABANDONEDBABY(iDf)
    iDf['CDLADVANCEBLOCK'] = CDLADVANCEBLOCK(iDf)
    iDf['CDLBELTHOLD'] = CDLBELTHOLD(iDf)
    iDf['CDLBREAKAWAY'] = CDLBREAKAWAY(iDf)
    iDf['CDLCLOSINGMARUBOZU'] = CDLCLOSINGMARUBOZU(iDf)
    iDf['CDLCONCEALBABYSWALL'] = CDLCONCEALBABYSWALL(iDf)
    iDf['CDLCOUNTERATTACK'] = CDLCOUNTERATTACK(iDf)
    iDf['CDLDARKCLOUDCOVER'] = CDLDARKCLOUDCOVER(iDf)
    iDf['CDLDOJI'] = CDLDOJI(iDf)
    iDf['CDLDOJISTAR'] = CDLDOJISTAR(iDf)
    iDf['CDLDRAGONFLYDOJI'] = CDLDRAGONFLYDOJI(iDf)
    iDf['CDLENGULFING'] = CDLENGULFING(iDf)
    iDf['CDLEVENINGDOJISTAR'] = CDLEVENINGDOJISTAR(iDf)
    iDf['CDLEVENINGSTAR'] = CDLEVENINGSTAR(iDf)
    iDf['CDLGAPSIDESIDEWHITE'] = CDLGAPSIDESIDEWHITE(iDf)
    iDf['CDLGRAVESTONEDOJI'] = CDLGRAVESTONEDOJI(iDf)
    iDf['CDLHAMMER'] = CDLHAMMER(iDf)
    iDf['CDLHANGINGMAN'] = CDLHANGINGMAN(iDf)
    iDf['CDLHARAMI'] = CDLHARAMI(iDf)
    iDf['CDLHARAMICROSS'] = CDLHARAMICROSS(iDf)
    iDf['CDLHIGHWAVE'] = CDLHIGHWAVE(iDf)
    iDf['CDLHIKKAKE'] = CDLHIKKAKE(iDf)
    iDf['CDLHIKKAKEMOD'] = CDLHIKKAKEMOD(iDf)
    iDf['CDLHOMINGPIGEON'] = CDLHOMINGPIGEON(iDf)
    iDf['CDLIDENTICAL3CROWS'] = CDLIDENTICAL3CROWS(iDf)
    iDf['CDLINNECK'] = CDLINNECK(iDf)
    iDf['CDLINVERTEDHAMMER'] = CDLINVERTEDHAMMER(iDf)
    iDf['CDLKICKING'] = CDLKICKING(iDf)
    iDf['CDLKICKINGBYLENGTH'] = CDLKICKINGBYLENGTH(iDf)
    iDf['CDLLADDERBOTTOM'] = CDLLADDERBOTTOM(iDf)
    iDf['CDLLONGLEGGEDDOJI'] = CDLLONGLEGGEDDOJI(iDf)
    iDf['CDLLONGLINE'] = CDLLONGLINE(iDf)
    iDf['CDLMARUBOZU'] = CDLMARUBOZU(iDf)
    iDf['CDLMATCHINGLOW'] = CDLMATCHINGLOW(iDf)
    iDf['CDLMATHOLD'] = CDLMATHOLD(iDf)
    iDf['CDLMORNINGDOJISTAR'] = CDLMORNINGDOJISTAR(iDf)
    iDf['CDLMORNINGSTAR'] = CDLMORNINGSTAR(iDf)
    iDf['CDLONNECK'] = CDLONNECK(iDf)
    iDf['CDLPIERCING'] = CDLPIERCING(iDf)
    iDf['CDLRICKSHAWMAN'] = CDLRICKSHAWMAN(iDf)
    iDf['CDLRISEFALL3METHODS'] = CDLRISEFALL3METHODS(iDf)
    iDf['CDLSEPARATINGLINES'] = CDLSEPARATINGLINES(iDf)
    iDf['CDLSHOOTINGSTAR'] = CDLSHOOTINGSTAR(iDf)
    iDf['CDLSHORTLINE'] = CDLSHORTLINE(iDf)
    iDf['CDLSPINNINGTOP'] = CDLSPINNINGTOP(iDf)
    iDf['CDLSTALLEDPATTERN'] = CDLSTALLEDPATTERN(iDf)
    iDf['CDLSTICKSANDWICH'] = CDLSTICKSANDWICH(iDf)
    iDf['CDLTAKURI'] = CDLTAKURI(iDf)
    iDf['CDLTASUKIGAP'] = CDLTASUKIGAP(iDf)
    iDf['CDLTHRUSTING'] = CDLTHRUSTING(iDf)
    iDf['CDLTRISTAR'] = CDLTRISTAR(iDf)
    iDf['CDLUNIQUE3RIVER'] = CDLUNIQUE3RIVER(iDf)
    iDf['CDLUPSIDEGAP2CROWS'] = CDLUPSIDEGAP2CROWS(iDf)
    iDf['CDLXSIDEGAP3METHODS'] = CDLXSIDEGAP3METHODS(iDf)
    return iDf
