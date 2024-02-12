"""
Simulation main function
"""

import os
import logger.arguments import get_args
from manager.ai_model.runner import AIRunner
from manager.game_model.runner import GameRunner
from environment.scenario_generator import NoReinsurance_RiskOne, Reinsurance_RiskOne, NoReinsurance_RiskFour, Reinsurance_RiskFour

if __name__ == '__main__':

    # Choose the model, 0 means ai model, 1 means game model
    model = 0
    models = {"0": AIRunner,
              "1": GameRunner
              }
    # Choose the scenario, 0 means noreinsurance_riskone, 1 means reinsurance_riskone, 2 means noreinsurance_riskfour, 3 means reinsurance_riskfour
    scenario = 0
    scenatios = {"0": NoReinsurance_RiskOne,
                 "1": Reinsurance_RiskOne,
                 "2": NoReinsurance_RiskFour,
                 "3": Reinsurance_RiskFour
                }

    # Get the simulation parameters
    sim_args, manager_args, broker_args, syndicate_args, reinsurancefirm_args, shareholder_args, risk_args = get_args()

    # Create scenario
    if str(scenario) == scenarios["0"]:
        brokers, syndicates, reinsurancefirms, shareholders, risk_models = NoReinsurance_RiskOne(sim_args, manager_args, broker_args, syndicate_args, reinsurancefirm_args, shareholder_args, risk_args)
    elif str(scenario) == scenarios["1"]:
        brokers, syndicates, reinsurancefirms, shareholders, risk_models = Reinsurance_RiskOne(sim_args, manager_args, broker_args, syndicate_args, reinsurancefirm_args, shareholder_args, risk_args)
    elif str(scenario) == scenarios["2"]:
        brokers, syndicates, reinsurancefirms, shareholders, risk_models = NoReinsurance_RiskFour(sim_args, manager_args, broker_args, syndicate_args, reinsurancefirm_args, shareholder_args, risk_args)
    else:
        brokers, syndicates, reinsurancefirms, shareholders, risk_models = Reinsurance_RiskFour(sim_args, manager_args, broker_args, syndicate_args, reinsurancefirm_args, shareholder_args, risk_args)

    # Run the simulation
    if str(model) == models["0"]: 
        runner = AIRunner(brokers, syndicates, reinsurancefirms, shareholders, risk_models)
    elif str(model) == models["1"]:
        runner = GameRunner(brokers, syndicates, reinsurancefirms, shareholders, risk_models)
    runner.run()

