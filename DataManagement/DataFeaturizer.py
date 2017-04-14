import talib
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
from talib.abstract import *


def overlap_sma_cross(iDf, iPeriod1=20, iPeriod2=50):
    """

    :param iDf:
    :param iPeriod1:
    :param iPeriod2:
    :return:
    """
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
    """

    :param iDf:
    :param iFastPeriod:
    :param iSlowPeriod:
    :param iSignalPeriod:
    :param iThresPeriod:
    :return:
    """
    # Adds MACD ind : difference between 2 EMA of slow and fast period
    # Adds Signal curve : EMA of signal period
    # Adds MACD hist : difference between macd and signal curve
    iDf = pd.concat([iDf, MACD(iDf, iFastPeriod, iSlowPeriod, iSignalPeriod)], axis=1)
    iDf['macdhist_sign'] = iDf['macdhist'].map(lambda x: 1 if x >= 0 else -1)
    # Check change of sign
    grouper = (iDf['macdhist_sign'] != iDf['macdhist_sign'].shift(1)).cumsum()
    iDf.loc[:, 'macdhist_counts'] = np.abs(iDf.groupby(grouper)['macdhist_sign'].cumsum())
    # If change in MACD sign and sufficient number of days with same sign macd hist
    iDf.loc[:, 'MACD_B_IND'] = ((iDf.loc[:, 'macdhist_counts'].shift() >= iThresPeriod - 1) & (
        iDf.loc[:, 'macdhist'] >= 0) & (iDf.loc[:, 'macdhist'].shift() < 0)) * 1
    iDf.loc[:, 'MACD_S_IND'] = ((iDf.loc[:, 'macdhist_counts'].shift() >= iThresPeriod - 1) & (
        iDf.loc[:, 'macdhist'] < 0) & (iDf.loc[:, 'macdhist'].shift() >= 0)) * 1
    return iDf


def momentum_ppo(iDf, iFastPeriod=12, iSlowPeriod=26):
    """

    :param iDf:
    :param iFastPeriod:
    :param iSlowPeriod:
    :return:
    """
    iDf.loc[:, 'PPO'] = PPO(iDf, fastperiod=iFastPeriod, slowperiod=iSlowPeriod, matype=0)
    return iDf


def overlap_bbands(iDf, iTimePeriod=10, iStdUp=2, iStdDwn=2):
    """

    :param iDf:
    :param iTimePeriod:
    :param iStdUp:
    :param iStdDwn:
    :return:
    """
    iDf = pd.concat([iDf, BBANDS(iDf, timeperiod=iTimePeriod, nbdevup=iStdUp, nbdevdn=iStdDwn)], axis=1)
    iDf.loc[:, 'bband_width'] = iDf.loc[:, 'upperband'] - iDf.loc[:, 'lowerband']
    iDf.loc[:, 'bband_width_change'] = iDf.loc[:, 'bband_width'] / iDf.loc[:, 'bband_width'].rolling(iTimePeriod).mean()
    iDf.loc[:, 'BBAND_B_IND'] = (iDf.loc[:, 'close'] < iDf.loc[:, 'lowerband']) * 1
    iDf.loc[:, 'BBAND_S_IND'] = (iDf.loc[:, 'close'] > iDf.loc[:, 'upperband']) * 1
    return iDf


def pattern_chandeliers(iDf, iPatternList=talib.get_function_groups()['Pattern Recognition']):
    """

    :param iDf:
    :param iPatternList:
    :return:
    """
    for pattern in iPatternList:
        func = getattr(talib.abstract, pattern)
        iDf.loc[:, pattern] = func(iDf)
    return iDf


def volume_obv(iDf, iAddSMA=True, iPeriod=20):
    """

    :param iDf:
    :param iAddSMA:
    :param iPeriod:
    :return:
    """
    iDf.loc[:, 'OBV'] = OBV(iDf)
    if iAddSMA:
        iDf.loc[:, 'OBV_SMA' + str(iPeriod)] = SMA(iDf, timeperiod=iPeriod, price='OBV')
    return iDf


def volume_ad(iDf):
    """

    :param iDf:
    :return:
    """
    iDf.loc[:, 'AD'] = AD(iDf)
    return iDf


def volume_adosc(iDf, iFastPeriod=3, iSlowPeriod=10):
    """

    :param iDf:
    :param iFastPeriod:
    :param iSlowPeriod:
    :return:
    """
    iDf.loc[:, 'ADOSC'] = ADOSC(iDf, fastperiod=iFastPeriod, slowperiod=iSlowPeriod)
    return iDf


def volatility_ATR(iDf, iPeriod=14):
    """

    :param iDf:
    :param iPeriod:
    :return:
    """
    iDf.loc[:, 'ATR'] = ATR(iDf, timeperiod=iPeriod)
    return iDf


def cycle_ht_dcperiod(iDf):
    """

    :param iDf:
    :return:
    """
    iDf.loc[:, 'HT_DCPERIOD'] = HT_DCPERIOD(iDf)
    return iDf


def cycle_ht_dcphase(iDf):
    """

    :param iDf:
    :return:
    """
    iDf.loc[:, 'HT_DCPHASE'] = HT_DCPHASE(iDf)
    return iDf


def cycle_ht_trendmode(iDf):
    """

    :param iDf:
    :return:
    """
    iDf.loc[:, 'HT_TRENDMODE'] = HT_TRENDMODE(iDf)
    return iDf


def featurize_stock_data(df, iFromDate=None, **kwargs):
    """

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

        # get 1st features (before featuring)
        bef_feat = [col for col in df.columns if col not in ['stock', 'date']]
        cur_feat = bef_feat[:]

        # IF WE SPECIFY PATTERN INDICATORS
        if kwargs.has_key('pattern'):
            # IF A LIST OF CHANDELIERS IS PROVIDED
            if kwargs['pattern'] is not None:
                df = pattern_chandeliers(df, kwargs['pattern'])
        # ELSE USE PATTERN INDICATORS
        else:
            df = pattern_chandeliers(df)

        # IF WE SPECIFY VOLUME INDICATORS
        if kwargs.has_key('volume'):
            # IF A LIST OF VOLUME INDICATOR (AND POSSIBLY PARAMETERS IS PROVIDED)
            # AD
            if kwargs['volume'].has_key('AD'):
                if kwargs['volume']['AD'] is not None:
                    df = volume_ad(df)  # No parameter for AD
            # ADOSC
            if kwargs['volume'].has_key('ADOSC'):
                if kwargs['volume'][
                    'ADOSC'] is not None:  # ADOSC default parameters : iFastPeriod=3, iSlowPeriod=10
                    df = volume_adosc(df, kwargs['volume']['ADOSC'])
                else:
                    df = volume_adosc(df)
            # OBV
            if kwargs['volume'].has_key('OBV'):
                if kwargs['volume']['OBV'] is not None:
                    df = volume_obv(df, kwargs['volume'][
                        'OBV'])  # OBV default parameters : iAddSMA=True, iPeriod=20
                else:
                    df = volume_obv(df)
        # ELSE USE ALL VOLUME INDICATORS
        else:
            df = volume_ad(df)
            df = volume_adosc(df)
            df = volume_obv(df)

        # IF WE SPECIFY VOLATILITY INDICATORS
        if kwargs.has_key('volatility'):
            # IF A LIST OF VOLATILITY INDICATOR (AND POSSIBLY PARAMETERS IS PROVIDED)
            # ATR
            if kwargs['volatility'].has_key('ATR'):
                if kwargs['volatility']['ATR'] is not None:
                    df = volatility_ATR(df,
                                        kwargs['volatility']['ATR'])  # ATR default parameters : iPeriod=14
                else:
                    df = volatility_ATR(df)
        # ELSE USE ALL VOLATILITY INDICATORS
        else:
            df = volatility_ATR(df)

        # IF WE SPECIFY OVERLAP INDICATORS
        if kwargs.has_key('overlap'):
            # IF A LIST OF OVERLAP INDICATOR (AND POSSIBLY PARAMETERS IS PROVIDED)
            # SMA CROSS
            if kwargs['overlap'].has_key('SMA_CROSS'):
                if kwargs['overlap']['SMA_CROSS'] is not None:
                    df = overlap_sma_cross(df, kwargs['overlap'][
                        'SMA_CROSS'])  # SMA_CROSS default parameters : iPeriod1=20, iPeriod2=50
                else:
                    df = overlap_sma_cross(df)
            # BOLLINGER Bands
            if kwargs['overlap'].has_key('BBANDS'):
                if kwargs['overlap']['BBANDS'] is not None:
                    df = overlap_bbands(df, kwargs['overlap'][
                        'BBANDS'])  # BBANDS default parameters : iTimePeriod=10, iStdUp=2, iStdDwn=2
                else:
                    df = overlap_bbands(df)
        # ELSE USE ALL OVERLAP INDICATORS
        else:
            df = overlap_sma_cross(df)
            df = overlap_bbands(df)

        # IF WE SPECIFY MOMENTUM INDICATORS
        if kwargs.has_key('momentum'):
            # IF A LIST OF MOMENTUM INDICATOR (AND POSSIBLY PARAMETERS IS PROVIDED)
            # MACD
            if kwargs['momentum'].has_key('MACD'):
                if kwargs['momentum']['MACD'] is not None:
                    df = momentum_macd(df, kwargs['momentum'][
                        'MACD'])  # MACD default parameters : iFastPeriod=12, iSlowPeriod=26, iSignalPeriod=9, iThresPeriod=14
                else:
                    df = momentum_macd(df)
            # PPO
            if kwargs['momentum'].has_key('PPO'):
                if kwargs['momentum']['PPO'] is not None:
                    df = momentum_ppo(df, kwargs['momentum'][
                        'PPO'])  # PPO default parameters : iFastPeriod=12, iSlowPeriod=26
                else:
                    df = momentum_ppo(df)
        # ELSE USE ALL MOMENTUM INDICATORS
        else:
            df = momentum_macd(df)
            df = momentum_ppo(df)

        # IF WE SPECIFY CYCLE INDICATORS
        if kwargs.has_key('cycle'):
            # IF A LIST OF CYCLE INDICATOR (AND POSSIBLY PARAMETERS IS PROVIDED)
            # HT_DCPERIOD
            if kwargs['cycle'].has_key('HT_DCPERIOD'):
                if kwargs['cycle']['HT_DCPERIOD'] is not None:
                    df = cycle_ht_dcperiod(df)  # No parameter for HT_DCPERIOD
            # HT_DCPHASE
            if kwargs['cycle'].has_key('HT_DCPHASE'):
                if kwargs['cycle']['HT_DCPHASE'] is not None:
                    df = cycle_ht_dcphase(df)  # No parameter for HT_DCPHASE

            # HT_TRENDMODE
            if kwargs['cycle'].has_key('HT_TRENDMODE'):
                if kwargs['cycle']['HT_TRENDMODE'] is not None:
                    df = cycle_ht_trendmode(df)  # No parameter for HT_TRENDMODE

        # ELSE USE ALL CYCLE INDICATORS
        else:
            df = cycle_ht_dcperiod(df)
            df = cycle_ht_dcphase(df)
            df = cycle_ht_trendmode(df)

        new_feat = [col for col in df.columns if col not in cur_feat + ['stock', 'date']]
        cur_feat = cur_feat + new_feat

    else:
        return None, None

    df.loc[:, cur_feat] = df.loc[:, cur_feat].shift(1)
    if iFromDate is not None:
        df = df[df['date'] >= pd.to_datetime(iFromDate, format='%Y-%m-%d')]
    return df, cur_feat


def measure_ahead(iDf, iDaysNum):
    """

    :param iDf:
    :param iDaysNum:
    :return:
    """
    iDf.loc[:, 'rmax'] = iDf.loc[:, 'close'].rolling(iDaysNum).max().shift(-iDaysNum - 1)
    iDf.loc[:, 'rmin'] = iDf.loc[:, 'close'].rolling(iDaysNum).min().shift(-iDaysNum - 1)
    iDf.loc[:, 'rmean'] = iDf.loc[:, 'close'].rolling(iDaysNum).mean().shift(-iDaysNum - 1)
    iDf.loc[:, 'rstd'] = iDf.loc[:, 'close'].rolling(iDaysNum).std().shift(-iDaysNum - 1)
    iDf.loc[:, 'pct_rmax'] = (iDf.loc[:, 'rmax'] / iDf.loc[:, 'close'] - 1) * 100
    iDf.loc[:, 'pct_rmin'] = (iDf.loc[:, 'rmin'] / iDf.loc[:, 'close'] - 1) * 100
    iDf.loc[:, 'pct_rmean'] = (iDf.loc[:, 'rmean'] / iDf.loc[:, 'close'] - 1) * 100
    iDf.loc[:, 'norm_rstd'] = (iDf.loc[:, 'rstd'] / iDf.loc[:, 'rmean']) * 100
    return iDf


def compute_regressor_target(iDf, iFromDate=None, iDaysNum=10):
    """

    :param iDf:
    :param iDaysNum:
    :param iRollingWindow:
    :param iTopPct:
    :return:
    """

    iDf = measure_ahead(iDf, iDaysNum)

    targets = ['pct_rmin', 'pct_rmax', 'pct_rmean', 'norm_rstd']
    drop_list = ['rmin', 'rmean', 'rmax', 'rstd', 'pct_rmin', 'pct_rmean', 'pct_rmax', 'norm_rstd']

    iDf.drop([feat for feat in drop_list if feat not in targets], axis=1, inplace=True)

    if iFromDate is not None:
        iDf = iDf[iDf['date'] >= pd.to_datetime(iFromDate, format='%Y-%m-%d')].copy()
    return iDf, targets


def compute_classifier_target(iDf, iFromDate=None, iDaysNum=10, iRollingWindow=20, iTopPct=0.1):
    """

    :param iDf:
    :param iDaysNum:
    :param iRollingWindow:
    :param iTopPct:
    :return:
    """

    iDf = measure_ahead(iDf, iDaysNum)

    iDf.loc[:, 'pct_rmean_upper'] = iDf.loc[:, 'pct_rmean'].rolling(iRollingWindow).quantile(1 - iTopPct)
    iDf.loc[:, 'pct_rmean_lower'] = iDf.loc[:, 'pct_rmean'].rolling(iRollingWindow).quantile(iTopPct)

    iDf.loc[:, 'B_TARGET'] = iDf[['pct_rmean', 'pct_rmean_upper']].apply(
        lambda x: 1 if (x['pct_rmean'] >= x['pct_rmean_upper']) & (x['pct_rmean_upper'] > 0) else 0, axis=1)
    iDf.loc[:, 'S_TARGET'] = iDf[['pct_rmean', 'pct_rmean_lower']].apply(
        lambda x: 1 if (x['pct_rmean'] <= x['pct_rmean_lower']) & (x['pct_rmean_lower'] < 0) else 0, axis=1)

    iDf.drop(
        ['pct_rmean_upper', 'pct_rmean_lower', 'rmin', 'rmean', 'rmax', 'rstd', 'pct_rmin', 'pct_rmean', 'pct_rmax',
         'norm_rstd'], axis=1, inplace=True)

    targets = ['B_TARGET', 'S_TARGET']

    if iFromDate is not None:
        iDf = iDf[iDf['date'] >= pd.to_datetime(iFromDate, format='%Y-%m-%d')].copy()
    return iDf, targets


def compute_duration_between_targets(iDf, targets):
    return iDf.loc[:, ['date'] + targets]


def compute_signal(iDf, results):
    """

    :param iDf:
    :param results:
    :return:
    """
    iDf = iDf.join(
        results.set_index('date').loc[:,
        ['B_TARGET_p', 'B_TARGET_p_strength', 'S_TARGET_p', 'S_TARGET_p_strength']],
        how='left')
    iDf.loc[:, 'B_SIGNAL_p'] = iDf.apply(lambda x: x['close'] if x['B_TARGET_p'] == 1 else np.nan, axis=1)
    iDf.loc[:, 'B_SIGNAL_ps'] = iDf.apply(lambda x: x['close'] if x['B_TARGET_p_strength'] == 2 else np.nan, axis=1)
    iDf.loc[:, 'S_SIGNAL_p'] = iDf.apply(lambda x: x['close'] if x['S_TARGET_p'] == 1 else np.nan, axis=1)
    iDf.loc[:, 'S_SIGNAL_ps'] = iDf.apply(lambda x: x['close'] if x['S_TARGET_p_strength'] == 2 else np.nan, axis=1)
    return iDf


# Not used
def compute_top_b_signal(iDf, results, top_pct):
    """

    :param iDf:
    :param results:
    :param top_pct:
    :return:
    """
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
    """

    :param iDf:
    :param results:
    :param top_pct:
    :return:
    """
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
