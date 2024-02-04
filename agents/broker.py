"""
Contains all the capabilities of Brokers
"""

import numpy as np

class Broker:
    def __init__(self, broker_id, broker_args):
        self.broker_id = broker_id
        self.broker_lambda_risks = broker_args['lambda_risks_daily']

    def bring_risk(self, ):
        """
        Bring insurable risks to the market
        """
        self.broker_id
        self.broker_lambda_risks

    def pay_premium(self, ):
        """
        Pay premium to the syndicates
        """

