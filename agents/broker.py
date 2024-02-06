"""
Contains all the capabilities of Brokers
"""

import numpy as np

class Broker:
    def __init__(self, broker_id, broker_args):
        self.broker_id = broker_id
        self.broker_lambda_risks = broker_args['lambda_risks_daily']
        self.bring_risk = {}
        self.contract = {}
        self.bring_claim = {}

    def data(self):
        """
        Create a dictionary with key/value pairs representing the Broker data.

        Returns
        ----------
        dict
        """

        return {
            "broker_id": self.broker_id,
            "broker_lambda_risks": self.broker_lambda_risks,
            "bring_risk": self.bring_risk,
            "contract": self.contract,
            "bring_claim": self.bring_claim
        }

    def pay_premium(self, ):
        """
        Pay premium to the syndicates
        """

