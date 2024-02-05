"""
Environment with broker, syndicate, shareholder in the market, and all the syndicates use one risk modle
"""

from .agents import Broker, Syndicate, ReinsuranceFirm, Shareholder
from risk.catastrophe_generator import RiskModel

class NoReinsurance_RiskOne():
    def __init__(self):
        # With or without reinsurance firms in the scenario
        self.reinsurance = False
        # Use one risk models
        self.one_risk = True
        # Use four risk models
        self.four_risk = False
        # List of brokers
        self.brokers = []
        # List of syndicates
        self.syndicates = []
        # List of shareholders
        self.shareholders = []
        # List of risk models used by synidacates
        self.risks = []

    def generate_scenario(self, broker_args, syndicate_args, reinsurancefirm_args, shareholder_args, risk_args):
        # Create broker, syndicate, shareholder
        self.brokers = [Broker(i,broker_args) for i in range(broker_args["num_brokers"])]
        self.syndicates = [Syndicate(i,syndicate_args) for i in range(syndicate_args["num_syndicates"])]
        self.shareholders = [Shareholder(i,shareholder_args) for i in range(shareholder_args["num_shareholders"])]
        self.risks = [RiskModel(0, risk_args) for i in range(risk_args["num_riskmodels"])]

        return brokers, syndicates, shareholders, risks