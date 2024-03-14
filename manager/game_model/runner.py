"""
Environment including the initial, resume, step
"""

import numpy as np

class GameRunner:
    """
    Game model 
    """
    def __init__(self, sim_args, manager_args, brokers, syndicates, reinsurancefirms, shareholders, risks, risk_model_configs, with_reinsurance, num_risk_models):
        self.sim_args = sim_args
        self.manager_args = manager_args
        self.brokers = brokers
        self.syndicates = syndicates
        self.reinsurancefirms = reinsurancefirms
        self.shareholders = shareholders
        self.risks = risks
        self.risk_model_configs = risk_model_configs
        self.with_reinsurance = with_reinsurance
        self.num_risk_models = num_risk_models
        self.trainer = None
        if self.with_reinsurance == False:
            self.scenario = str("noreinsurance")
        else:
            self.scenario = str("reinsurance")
