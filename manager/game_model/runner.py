"""
Environment including the initial, resume, step
"""

import os
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO
from environment.multi_agent_env import MultiAgentBasedModel
from logger import logger
from ray.rllib.policy.policy import PolicySpec
import numpy as np
import gymnasium as gym
from ray import air, tune
from ray.rllib.examples.policy.random_policy import RandomPolicy

class GameRunner:
    """
    Game model 
    """
    def __init__(self, sim_args, manager_args, broker_args, syndicate_args, reinsurancefirm_args, shareholder_args, risk_args, seed, brokers, syndicates, reinsurancefirms, shareholders, risks, risk_model_configs, with_reinsurance, num_risk_models):
        self.sim_args = sim_args
        self.manager_args = manager_args
        self.broker_args = broker_args
        self.syndicate_args =  syndicate_args
        self.reinsurancefirm_args = reinsurancefirm_args
        self.shareholder_args = shareholder_args
        self.risk_args = risk_args
        self.seed = seed
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
        self.logger = logger.Logger(self.num_risk_models, self.risks, self.brokers, self.syndicates)

    def env_creator(self, env_config):
        """
        Register specialty insurance market environment for The rllib
        """

        return MultiAgentBasedModel(**env_config)
    
    def policy_mapping_fn(self, agent_id, episode, worker, **kwargs):
        # agent0 -> main0
        # agent1 -> main1
        return f"main{agent_id[-1]}"

    def ppo_trainer_creator(self, insurance_args):
        gym.logger.set_level(40)
        config = {
            "env": "SpecialtyInsuranceMarket-validation",
            "framework": "tf",
            "multi_agent": {"policies":{
                # The Policy we are actually learning.
                "main0": PolicySpec(
                    observation_space=gym.spaces.Box(low=np.array([-1000000,-1000000,-1000000,-1000000,-1000000,-1000000]), 
                                                     high=np.array([1000000,1000000,3000000,3000000,3000000,3000000]), dtype = np.float32),
                    action_space=gym.spaces.Box(0.5, 0.9, dtype = np.float32)
                ),
                "main1": PolicySpec(
                    observation_space=gym.spaces.Box(low=np.array([-1000000,-1000000,-1000000,-1000000,-1000000,-1000000]), 
                                                     high=np.array([1000000,1000000,3000000,3000000,3000000,3000000]), dtype = np.float32),
                    action_space=gym.spaces.Box(0.5, 0.9, dtype = np.float32)
                ),
                "random": PolicySpec(policy_class=RandomPolicy),
                }, 
                        "policy_mapping_fn": self.policy_mapping_fn,
                        "policies_to_train":["main0"],
            },
            "observation_space": gym.spaces.Box(low=np.array([-1000000,-1000000,-1000000,-1000000,-1000000,-1000000]), 
                                            high=np.array([1000000,1000000,3000000,3000000,3000000,3000000]), dtype = np.float32),
            "action_space": gym.spaces.Box(0.5, 0.9, dtype = np.float32),
            "env_config": insurance_args,
            "evaluation_interval": 2,
            "evaluation_duration": 20,
        }
        self.trainer = PPO(config=config)
    
    def save_data(self, brokers, syndicates, reinsurance_firms, shareholders):
        """Method to collect statistics about the current state of the simulation. Will pass these to the 
           Logger object (self.logger) to be recorded."""
        # Collect data
        total_cash_no = sum([insurancefirm.cash for insurancefirm in self.insurancefirms])
        total_excess_capital = sum([insurancefirm.get_excess_capital() for insurancefirm in self.insurancefirms])
        total_profitslosses =  sum([insurancefirm.get_profitslosses() for insurancefirm in self.insurancefirms])
        total_contracts_no = sum([len(insurancefirm.underwritten_contracts) for insurancefirm in self.insurancefirms])
        total_reincash_no = sum([reinsurancefirm.cash for reinsurancefirm in self.reinsurancefirms])
        total_reinexcess_capital = sum([reinsurancefirm.get_excess_capital() for reinsurancefirm in self.reinsurancefirms])
        total_reinprofitslosses =  sum([reinsurancefirm.get_profitslosses() for reinsurancefirm in self.reinsurancefirms])
        total_reincontracts_no = sum([len(reinsurancefirm.underwritten_contracts) for reinsurancefirm in self.reinsurancefirms])
        operational_no = sum([insurancefirm.operational for insurancefirm in self.insurancefirms])
        reinoperational_no = sum([reinsurancefirm.operational for reinsurancefirm in self.reinsurancefirms])
        catbondsoperational_no = sum([catbond.operational for catbond in self.catbonds])
        
        # Collect agent-level data
        insurance_firms = [(insurancefirm.cash,insurancefirm.id,insurancefirm.operational) for insurancefirm in self.insurancefirms]
        reinsurance_firms = [(reinsurancefirm.cash,reinsurancefirm.id,reinsurancefirm.operational) for reinsurancefirm in self.reinsurancefirms]
        
        # Prepare dict
        current_log = {}
        current_log['total_cash'] = total_cash_no
        current_log['total_excess_capital'] = total_excess_capital
        current_log['total_profitslosses'] = total_profitslosses
        current_log['total_contracts'] = total_contracts_no
        current_log['total_operational'] = operational_no
        current_log['total_reincash'] = total_reincash_no
        current_log['total_reinexcess_capital'] = total_reinexcess_capital
        current_log['total_reinprofitslosses'] = total_reinprofitslosses
        current_log['total_reincontracts'] = total_reincontracts_no
        current_log['total_reinoperational'] = reinoperational_no
        current_log['total_catbondsoperational'] = catbondsoperational_no
        current_log['market_premium'] = self.market_premium
        current_log['market_reinpremium'] = self.reinsurance_market_premium
        current_log['cumulative_bankruptcies'] = self.cumulative_bankruptcies
        current_log['cumulative_market_exits'] = self.cumulative_market_exits
        current_log['cumulative_unrecovered_claims'] = self.cumulative_unrecovered_claims
        current_log['cumulative_claims'] = self.cumulative_claims    #Log the cumulative claims received so far.
        
        # Add agent-level data to dict
        current_log['insurance_firms_cash'] = insurance_firms
        current_log['reinsurance_firms_cash'] = reinsurance_firms
        current_log['market_diffvar'] = self.compute_market_diffvar()
        
        current_log['individual_contracts'] = []
        individual_contracts_no = [len(insurancefirm.underwritten_contracts) for insurancefirm in self.insurancefirms]
        for i in range(len(individual_contracts_no)):
            current_log['individual_contracts'].append(individual_contracts_no[i])

        # Call to Logger object
        self.logger.record_data(current_log)

    def run(self):
        # Folder for recording
        top_dir = "insurance_scenario_" + self.scenario + "_model_" + str(self.num_risk_models)

        # Register environment
        register_env("SpecialtyInsuranceMarket-validation", self.env_creator)

        # Insurance arguments
        insurance_args = {"sim_args": self.sim_args,
                "manager_args": self.manager_args,
                "broker_args": self.broker_args,
                "syndicate_args": self.syndicate_args,
                "reinsurancefirm_args": self.reinsurancefirm_args,
                "shareholder_args": self.shareholder_args,
                "risk_args": self.risk_args,
                "brokers": self.brokers,
                "syndicates": self.syndicates,
                "reinsurancefirms": self.reinsurancefirms,
                "shareholders": self.shareholders,
                "catastrophes": self.risks,
                "risk_model_configs": self.risk_model_configs,
                "with_reinsurance": self.with_reinsurance,
                "num_risk_models": self.num_risk_models}
        self.ppo_trainer_creator(insurance_args)
        env = MultiAgentBasedModel(**insurance_args)
    
        total_steps = 0
        terminated_dict = {"__all__": False}
    
        obs_dict, info_dict = env.reset()

        while not terminated_dict["__all__"]:
            if total_steps % 20 == 0: print(".", end="")
        
            action_dict = self.trainer.compute_actions(obs_dict)  
            print(action_dict)
            total_steps += 1
        
            obs_dict, reward_dict, terminated_dict, flag_dict, info_dict = env.step(action_dict)
        
        self.save_data(env.mm.market.brokers, env.mm.market.syndicates, env.mm.market.reinsurancefirms, env.mm.market.shareholders)
        #return self.logger.obtain_log(logs)
        return 0