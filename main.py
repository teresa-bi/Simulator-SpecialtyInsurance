"""
Simulation main function
"""

import os
from logger.arguments import get_arguments
from logger import logger
import numpy as np

from environment.market_generator import MarketGenerator
from environment.risk_generator import RiskGenerator
#from manager.ai_model.runner import AIRunner
from manager.game_model.runner import GameRunner

if __name__ == '__main__':

    # Ensure the logging directory exists
    if not os.path.isdir("data"):
        assert not os.path.exists("data")
        os.makedirs("data")

    # Set the number of simulation replication
    num_replication = 9
    # Get the simulation parameters
    sim_args, manager_args, broker_args, syndicate_args, reinsurancefirm_args, shareholder_args, risk_args, seed = get_arguments()

    # Run the replication
    for i in range(num_replication):
        # Create scenario
        with_reinsurance = False
        num_risk_models = risk_args["num_riskmodels"]
        if i <= 2:
            syndicate_args["lead_line_size"] = 1
            syndicate_args["follow_line_size"] = 0
            if i == 0:
                syndicate_args["ambiguity_level"] = 0
            elif i == 1:
                syndicate_args["ambiguity_level"] = 0.5
            else:
                syndicate_args["ambiguity_level"] = 1
        elif 2 < i <= 5:
            syndicate_args["lead_line_size"] = 0.8
            syndicate_args["follow_line_size"] = 0.2
            if i == 3:
                syndicate_args["ambiguity_level"] = 0
            elif i == 4:
                syndicate_args["ambiguity_level"] = 0.5
            else:
                syndicate_args["ambiguity_level"] = 1
        else:
            syndicate_args["lead_line_size"] = 0.5
            syndicate_args["follow_line_size"] = 0.25
            if i == 6:
                syndicate_args["ambiguity_level"] = 0
            elif i == 7:
                syndicate_args["ambiguity_level"] = 0.5
            else:
                syndicate_args["ambiguity_level"] = 1
        catastrophes, broker_risks, fair_market_premium, risk_model_configs = RiskGenerator(num_risk_models, sim_args, broker_args, risk_args, seed).generate_risks()
        brokers, syndicates, reinsurancefirms, shareholders = MarketGenerator(with_reinsurance, num_risk_models, sim_args, broker_args, syndicate_args, reinsurancefirm_args, shareholder_args, risk_model_configs).generate_agents()
        log = logger.Logger(risk_args["num_riskmodels"], brokers, syndicates)

        # Run the simulation
        model = 1
        if model == 0: 
            runner = AIRunner(sim_args, manager_args, broker_args, syndicate_args, reinsurancefirm_args, shareholder_args, risk_args, seed, brokers, syndicates, reinsurancefirms, shareholders, catastrophes, broker_risks, fair_market_premium, risk_model_configs, with_reinsurance, num_risk_models, log)
        elif model == 1:
            runner = GameRunner(sim_args, manager_args, broker_args, syndicate_args, reinsurancefirm_args, shareholder_args, risk_args, seed, brokers, syndicates, reinsurancefirms, shareholders, catastrophes, broker_risks, fair_market_premium, risk_model_configs, with_reinsurance, num_risk_models, log)
        logs = runner.run()

        # Restore the log
        log.restore_logger_object(list(logs))
        log.save_log(syndicate_args["lead_line_size"], syndicate_args["ambiguity_level"], seed)

