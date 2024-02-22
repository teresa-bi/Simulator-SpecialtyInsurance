"""
Contains all the capabilities of Shareholder
"""

import numpy as np

class Shareholder:
    def __init__(self, shareholder_id, shareholder_args, num_risk_models, risk_model_configs):
        self.shareholder_id = shareholder_id

    def data(self):
        """
        Create a dictionary with key/value pairs representing the Shareholder data.

        Returns
        ----------
        dict
        """

        return {
            "shareholder_id": self.shareholder_id
        }

    def ask_dividends_from_insurer():
        """
        Receive dividends from insurance and reinsurance firms
        """
