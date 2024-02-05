"""
Environment with broker, syndicate, reinsurance firm and shareholder in the market, and syndicates use four risk modles
"""

from .agents import Broker, Syndicate, ReinsuranceFirm, Shareholder
from risk.catastrophe_generator import RiskModel

class Reinsurance_RiskFour(NoReinsurance_RiskOne):
    def __init__(self):
        super(Reinsurance_RiskFour, self).init()
        self.reinsurance = True
        self.one_risk = False
        self.four_risk = True

    def generate_scenario(self, sim_args, manager_args, broker_args, syndicate_args, reinsurancefirm_args, shareholder_args, risk_args):
        # Create broker, syndicate, shareholder
        brokers = []
        synidates = []

