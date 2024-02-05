"""
Environment with broker, syndicate, reinsurance fimr, and shareholder in the market, and all the syndicates use one risk modle
"""

from .agents import Broker, Syndicate, ReinsuranceFirm, Shareholder
from risk.catastrophe_generator import RiskModel

class Reinsurance_RiskOne(NoReinsurance_RiskOne):
    def __init__(self):
        super(Reinsurance_RiskOne, self).init()
        self.reinsurance = True
        self.one_risk = True
        self.four_risk = False

    def generate_scenario(self, sim_args, manager_args, broker_args, syndicate_args, reinsurancefirm_args, shareholder_args, risk_args):
        # Create broker, syndicate, shareholder
        brokers = []
        synidates = []

