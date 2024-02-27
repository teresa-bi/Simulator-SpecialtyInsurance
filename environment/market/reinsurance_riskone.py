import numpy as np
from environment.market import NoReinsurance_RiskOne

class Reinsurance_RiskOne(NoReinsurance_RiskOne):
    """
    Environment including brokers, syndicates, reinsurancefirms, sharholders, and one risk model
    """
    def __init__(self, time, sim_maxstep, manager_args, brokers, syndicates, reinsurancefirms, shareholders, risks, risk_model_configs, catastrophe_event, attritional_loss_event, broker_risk_event, broker_premium_event, broker_claim_event):
        super(Reinsurance_RiskOne, self).__init__()
