"""
Environment with broker, syndicate, shareholder in the market, and syndicates use four risk modles
"""

from .agents import Broker, Syndicate, ReinsuranceFirm, Shareholder
from risk.catastrophe_generator import RiskModel

class NoReinsurance_RiskFour(NoReinsurance_RiskOne):
    def __init__(self):
        super(NoReinsurance_RiskFour, self).init()
        self.reinsurance = False
        self.one_risk = False
        self.four_risk = True
        # List of reinsurance firms
        self.reinsurancefirms = []

    def generate_scenario(self, sim_args, manager_args, broker_args, syndicate_args, reinsurancefirm_args, shareholder_args, risk_args):
        

