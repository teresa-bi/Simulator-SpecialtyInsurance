"""
Simulation main function
"""

import os
import logger.arguments import get_args
from environment.scenario_generator import ScenarioGenerator
from environment.risk_generator import RiskGenerator
from manager.ai_model.runner import AIRunner
from manager.game_model.runner import GameRunner

if __name__ == '__main__':

    # Get the simulation parameters
    sim_args, manager_args, broker_args, syndicate_args, reinsurancefirm_args, shareholder_args, risk_args = get_args()

    # Create scenario
    with_reinsurance = Flase
    num_risk_models = 1
    brokers, syndicates, reinsurancefirms, shareholders = ScenarioGenerator(with_reinsurance, num_risk_models, broker_args, syndicate_args, reinsurancefirm_args, shareholder_args).generate_agents()
    risks = RiskGenerator(risk_args).generate_risks()

    # Run the simulation
    model = 0
    if model == 0: 
        runner = AIRunner(sim_args, manager_args, brokers, syndicates, reinsurancefirms, shareholders, risks, with_reinsurance, num_risk_models)
    elif model == 1:
        runner = GameRunner(sim_args, manager_args, brokers, syndicates, reinsurancefirms, shareholders, risks, with_reinsurance, num_risk_models)
    runner.run()

