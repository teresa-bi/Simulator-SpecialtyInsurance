"""
Start the simulation
"""

import os
import numpy as np
import logger.arguments import get_args
from environment.env import SpecialtyInsuranceMarketEnv
from manager.game_mpdel.runner import Runner
#from manager.ai_model.runner import Runner

if __name__ == '__main__':

    # Get the simulation parameters
    sim_args, manager_args, broker_args, syndicate_args, reinsurancefirm_args, shareholder_args, risk_args = get_args()

    # Create enviornment
    env = SpecialtyInsuranceMarketEnv(sim_args, manager_args, broker_args, syndicate_args, reinsurancefirm_args, shareholder_args, risk_args)

    # Run the simulation
    runner = Runner(env, sim_args, manager_args, broker_args, syndicate_args, reinsurancefirm_args, shareholder_args, risk_args)
    runner.run(sim_args)

