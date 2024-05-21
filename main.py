"""
Simulation main function
"""

import os
from logger.arguments import get_arguments
from logger import logger
import numpy as np

from environment.market_generator import MarketGenerator
from environment.risk_generator import RiskGenerator
from manager.ai_model.runner import AIRunner
from manager.game_model.runner import GameRunner


def seeds(num_replications,seed):
    # Generate random seed
    np.random.seed(seed)
    np_seeds = []
    for i in range(num_replications):
        np_seed = np.random.randint(0,2**31-1)
        np_seeds.append(np_seed)
    return np_seeds

if __name__ == '__main__':

    # Ensure the logging directory exists
    if not os.path.isdir("data"):
        assert not os.path.exists("data")
        os.makedirs("data")

    # Set the number of simulation replication
    num_replication = 1
    # Get the simulation parameters
    sim_args, manager_args, broker_args, syndicate_args, reinsurancefirm_args, shareholder_args, risk_args, seed = get_arguments()
    # Get random seeds for the simulation 
    np_seed = seeds(num_replication, seed)

    # Run the replication
    for i in range(num_replication):
        # Make numpy random generation predictable
        np.random.seed(np_seed[i])
        seed = np.random.randint(0,2**31-1)

        # Create scenario
        with_reinsurance = False
        num_risk_models = risk_args["num_riskmodels"]
        catastrophes, catastrophe_damage, broker_risks, fair_market_premium, risk_model_configs = RiskGenerator(num_risk_models, sim_args, broker_args, risk_args, seed).generate_risks()
        brokers, syndicates, reinsurancefirms, shareholders = MarketGenerator(with_reinsurance, num_risk_models, sim_args, broker_args, syndicate_args, reinsurancefirm_args, shareholder_args, risk_model_configs).generate_agents()
        log = logger.Logger(risk_args["num_riskmodels"], catastrophes, catastrophe_damage, brokers, syndicates)

        # Run the simulation
        model = 1
        if model == 0: 
            runner = AIRunner(sim_args, manager_args, broker_args, syndicate_args, reinsurancefirm_args, shareholder_args, risk_args, seed, brokers, syndicates, reinsurancefirms, shareholders, catastrophes, broker_risks, fair_market_premium, risk_model_configs, with_reinsurance, num_risk_models, log)
        elif model == 1:
            runner = GameRunner(sim_args, manager_args, broker_args, syndicate_args, reinsurancefirm_args, shareholder_args, risk_args, seed, brokers, syndicates, reinsurancefirms, shareholders, catastrophes, broker_risks, fair_market_premium, risk_model_configs, with_reinsurance, num_risk_models, log)
        logs = runner.run()

        # Restore the log
        log.restore_logger_object(list(logs))
        log.save_log()

