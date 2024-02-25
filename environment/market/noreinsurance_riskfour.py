import numpy as np
from environment.market import NoReinsurance_RiskOne

class NoReinsurance_RiskFour(NoReinsurance_RiskOne):
    """
    Environment including brokers, syndicates, sharholders, and four risk models
    """
    def __init__(self, time, sim_maxstep, manager_args, brokers, syndicates, reinsurancefirms, shareholders, risks, risk_model_configs):
        super(NoReinsurance_RiskFour, self).__init__()


