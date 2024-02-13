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

    def create(self, broker_args, syndicate_args, reinsurancefirm_args, shareholder_args, risk_args):
        """
        Create agents including brokers, syndicates, reinsurancefirms, shareholders, risk_models.

        Parameters
        ----------
        broker_args, syndicate_args, reinsurancefirm_args, shareholder_args, risk_args

        Returns
        ----------
        dict
        """
        for i in range(broker_args["num_brokers"]):
            self.brokers[str(i)] = Broker(i,broker_args)
        for i in range(syndicate_args["num_syndicates"])]:
            self.syndicates[str(i)] = Syndicate(i,syndicate_args)
        if self.reinsurance == True:
            for i in range(reinsurancefirm_args["num_reinsurancefirms"])]:
                self.reinsurancefirms[str(i)] = ReinsuranceFirm(i,reinsurancefirm_args)
        for i in range(shareholder_args["num_shareholders"]):
            self.shareholders[str(i)] = Shareholder(i,shareholder_args)
        for i in range(risk_args["num_risks"]):
            if self.one_risk:
                self.risk_models[str(i)] = RiskModel.one_risk_model(risk_args) 
            elif self.four_risk:
                self.risk_models[str(i)] = RiskModel.four_risk_model(risk_args)
            else:
                self.risk_models[str(i)] = RiskModel.other_risk_model(risk_args)

        return self.brokers, self.syndicates, self.reinsurancefirms, self.shareholders, self.risk_models

