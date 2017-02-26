from DataManagement import PathHandler as ph
from DataManagement import DataHandler as dhand

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


class PFManager:
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
        file_handler = RotatingFileHandler(self.Paths['LogsPath'] + 'pf_logfile.log', 'a', 1000000, 1)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        self.file_handler = file_handler
        return

    def load_movements(self):
        return self.DataHand.get_moves_histo()

    def load_portfolio(self):
        return self.DataHand.get_moves_histo()

        # TODO :
        # create a new PF if None
        # get PF date
        # Rename PF if needed
        # add movements to PF
        # Compute new PF with moves
        # Compute PF value
        # Add costs from De Giro
        # Add Sale Alert with threshold of gain
        # Add it in Run Manager
