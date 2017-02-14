from talib.abstract import *
import talib
import numpy as np
import pandas as pd

import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
from talib.abstract import *


def overlap_sma_cross(iDf, iPeriod1=20, iPeriod2=50):
    iPeriodmin = min(iPeriod1, iPeriod2)
    iPeriodmax = max(iPeriod1, iPeriod2)
    iDf.loc[:, 'SMA' + str(iPeriodmin)] = SMA(iDf, timeperiod=iPeriodmin, price='close')
    iDf.loc[:, 'SMA' + str(iPeriodmax)] = SMA(iDf, timeperiod=iPeriodmax, price='close')
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
    return iDf


def momentum_macd(iDf, iFastPeriod=12, iSlowPeriod=26, iSignalPeriod=9, iThresPeriod=14):
    # Adds MACD ind : difference between 2 EMA of slow and fast period
    # Adds Signal curve : EMA of signal period
    # Adds MACD hist : difference between macd and signal curve
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
    return iDf


def momentum_ppo(iDf, iFastPeriod=12, iSlowPeriod=26):
    iDf.loc[:, 'PPO'] = PPO(iDf, fastperiod=iFastPeriod, slowperiod=iSlowPeriod, matype=0)
    return iDf


def overlap_bbands(iDf, iTimePeriod=10, iStdUp=2, iStdDwn=2):
    iDf = pd.concat([iDf, BBANDS(iDf, timeperiod=iTimePeriod, nbdevup=iStdUp, nbdevdn=iStdDwn)], axis=1)
    iDf.loc[:, 'bband_width'] = iDf.loc[:, 'upperband'] - iDf.loc[:, 'lowerband']
    iDf.loc[:, 'bband_width_change'] = iDf.loc[:, 'bband_width'] / iDf.loc[:, 'bband_width'].rolling(iTimePeriod).mean()
    iDf.loc[:, 'BBAND_B_IND'] = (iDf.loc[:, 'close'] < iDf.loc[:, 'lowerband']) * 1
    iDf.loc[:, 'BBAND_S_IND'] = (iDf.loc[:, 'close'] > iDf.loc[:, 'upperband']) * 1
    return iDf


def pattern_chandeliers(iDf, iPatternList=talib.get_function_groups()['Pattern Recognition']):
    for pattern in iPatternList:
        func = getattr(talib.abstract, pattern)
        iDf.loc[:, pattern] = func(iDf)
    return iDf


def volume_obv(iDf, iAddSMA=True, iPeriod=20):
    iDf.loc[:, 'OBV'] = OBV(iDf)
    if iAddSMA:
        iDf.loc[:, 'OBV_SMA' + str(iPeriod)] = SMA(iDf, timeperiod=iPeriod, price='OBV')
    return iDf


def volume_ad(iDf):
    iDf.loc[:, 'AD'] = AD(iDf)
    return iDf


def volume_adosc(iDf, iFastPeriod=3, iSlowPeriod=10):
    iDf.loc[:, 'ADOSC'] = ADOSC(iDf, fastperiod=iFastPeriod, slowperiod=iSlowPeriod)
    return iDf


def volatility_ATR(iDf, iPeriod=14):
    iDf.loc[:, 'ATR'] = ATR(iDf, timeperiod=iPeriod)
    return iDf


def measure_ahead(iDf, iDaysNum):
    iDf.loc[:, 'rmax'] = iDf.loc[:, 'close'].rolling(iDaysNum).max().shift(-iDaysNum - 1)
    iDf.loc[:, 'rmin'] = iDf.loc[:, 'close'].rolling(iDaysNum).min().shift(-iDaysNum - 1)
    iDf.loc[:, 'rmean'] = iDf.loc[:, 'close'].rolling(iDaysNum).mean().shift(-iDaysNum - 1)
    iDf.loc[:, 'rstd'] = iDf.loc[:, 'close'].rolling(iDaysNum).std().shift(-iDaysNum - 1)
    iDf.loc[:, 'pct_rmax'] = (iDf.loc[:, 'rmax'] / iDf.loc[:, 'close'] - 1) * 100
    iDf.loc[:, 'pct_rmin'] = (iDf.loc[:, 'rmin'] / iDf.loc[:, 'close'] - 1) * 100
    iDf.loc[:, 'pct_rmean'] = (iDf.loc[:, 'rmean'] / iDf.loc[:, 'close'] - 1) * 100
    iDf.loc[:, 'norm_rstd'] = (iDf.loc[:, 'rstd'] / iDf.loc[:, 'rmean']) * 100
    return iDf


def compute_target(iDf, iDaysNum, iRollingWindow, iTopPct):
    iDf = measure_ahead(iDf, iDaysNum)
    iDf.loc[:, 'pct_mean_upper'] = iDf.loc[:, 'pct_rmean'].rolling(iRollingWindow).quantile(1 - iTopPct)
    iDf.loc[:, 'pct_mean_lower'] = iDf.loc[:, 'pct_rmean'].rolling(iRollingWindow).quantile(iTopPct)

    iDf.loc[:, 'B_TARGET'] = iDf.apply(
        lambda x: 1 if (x['pct_rmean'] >= x['pct_mean_upper']) & (x['pct_mean_upper'] > 0) else 0, axis=1)
    iDf.loc[:, 'S_TARGET'] = iDf.apply(
        lambda x: 1 if (x['pct_rmean'] <= x['pct_mean_lower']) & (x['pct_mean_lower'] < 0) else 0, axis=1)
    iDf.drop(['pct_mean_upper', 'pct_mean_lower'], axis=1, inplace=True)
    iDf.drop(['rmin', 'rmean', 'rmax', 'rstd', 'pct_rmin', 'pct_rmean', 'pct_rmax', 'norm_rstd'], axis=1, inplace=True)
    targets = ['B_TARGET', 'S_TARGET']
    return iDf, targets


def compute_signal(iDf, results):
    iDf = iDf.join(
        results.set_index('date').loc[:, ['B_TARGET_p', 'B_TARGET_p_strength', 'S_TARGET_p', 'S_TARGET_p_strength']],
        how='left')
    iDf.loc[:, 'B_SIGNAL_p'] = iDf.apply(lambda x: x['close'] if x['B_TARGET_p'] == 1 else np.nan, axis=1)
    iDf.loc[:, 'B_SIGNAL_ps'] = iDf.apply(lambda x: x['close'] if x['B_TARGET_p_strength'] == 2 else np.nan, axis=1)
    iDf.loc[:, 'S_SIGNAL_p'] = iDf.apply(lambda x: x['close'] if x['S_TARGET_p'] == 1 else np.nan, axis=1)
    iDf.loc[:, 'S_SIGNAL_ps'] = iDf.apply(lambda x: x['close'] if x['S_TARGET_p_strength'] == 2 else np.nan, axis=1)
    return iDf


# Not used
def compute_top_b_signal(iDf, results, top_pct):
    nb_rows = np.int(results.shape[0] * float(top_pct) / 100)
    buy_dates = results.sort_values(['pct_rmax_p', 'pct_rmean_p', 'norm_rstd_p'], ascending=[0, 0, 1]).iloc[0:nb_rows][
        'date']

    if results.sort_values(['pct_rmax_p', 'pct_rmean_p', 'norm_rstd_p'], ascending=[0, 0, 1]).iloc[0][
        'pct_rmean_p'] > 0:
        iDf.loc[buy_dates, 'B_SIGNAL'] = 1
    else:
        iDf.loc[:, 'B_SIGNAL'] = np.nan
    iDf.loc[:, 'B_SIGNAL'] = iDf.apply(lambda x: x['close'] if x['B_SIGNAL'] == 1 else np.nan, axis=1)
    return iDf


# Not used
def compute_top_s_signal(iDf, results, top_pct):
    nb_rows = np.int(results.shape[0] * float(top_pct) / 100)
    sell_dates = results.sort_values(['pct_rmin_p', 'pct_rmean_p', 'norm_rstd_p'], ascending=[1, 1, 1]).iloc[0:nb_rows][
        'date']
    if results.sort_values(['pct_rmin_p', 'pct_rmean_p', 'norm_rstd_p'], ascending=[1, 1, 1]).iloc[0][
        'pct_rmean_p'] < 0:
        iDf.loc[sell_dates, 'S_SIGNAL'] = 1
    else:
        iDf.loc[:, 'S_SIGNAL'] = np.nan
    iDf.loc[:, 'S_SIGNAL'] = iDf.apply(lambda x: x['close'] if x['S_SIGNAL'] == 1 else np.nan, axis=1)
    return iDf
