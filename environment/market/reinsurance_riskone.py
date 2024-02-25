import numpy as np
from environment.market import NoReinsurance_RiskOne

class Reinsurance_RiskOne(NoReinsurance_RiskOne):
    """
    Environment including brokers, syndicates, reinsurancefirms, sharholders, and one risk model
    """
    def __init__(self, time, sim_maxstep, manager_args, brokers, syndicates, reinsurancefirms, shareholders, risks, risk_model_configs):
        super(Reinsurance_RiskOne, self).__init__()
