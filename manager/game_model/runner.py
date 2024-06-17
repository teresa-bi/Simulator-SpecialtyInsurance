"""
Environment including the initial, resume, step
"""

import os
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO
from environment.multi_agent_env import MultiAgentBasedModel
from agents import Syndicate
from ray.rllib.policy.policy import PolicySpec
import numpy as np
import gymnasium as gym
from ray import air, tune

class GameRunner:
    """
    Game model 
    """
    def __init__(self, sim_args, manager_args, broker_args, syndicate_args, reinsurancefirm_args, shareholder_args, risk_args, 
                 seed, brokers, syndicates, reinsurancefirms, shareholders, catastrophes, broker_risks, fair_market_premium, 
                 risk_model_configs, with_reinsurance, num_risk_models, logger):
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
        self.catastrophes = catastrophes
        self.broker_risks = broker_risks
        self.fair_market_premium = fair_market_premium
        self.risk_model_configs = risk_model_configs
        self.with_reinsurance = with_reinsurance
        self.num_risk_models = num_risk_models
        self.trainer = None
        if self.with_reinsurance == False:
            self.scenario = str("noreinsurance")
        else:
            self.scenario = str("reinsurance")
        self.logger = logger
        
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
                    observation_space=gym.spaces.Box(low=np.array([-1000000,-1000000,-1000000,-1000000]), 
                                                     high=np.array([3000000,3000000,3000000,3000000]), dtype = np.float32),
                    action_space=gym.spaces.Box(0.5, 0.9, dtype = np.float32)
                ),
                "main1": PolicySpec(
                    observation_space=gym.spaces.Box(low=np.array([-1000000,-1000000,-1000000,-1000000]), 
                                                     high=np.array([3000000,3000000,3000000,3000000]), dtype = np.float32),
                    action_space=gym.spaces.Box(0.5, 0.9, dtype = np.float32)
                ),
                }, 
                        "policy_mapping_fn": self.policy_mapping_fn,
                        "policies_to_train":["main0"],
            },
            "observation_space": gym.spaces.Box(low=np.array([-1000000,-1000000,-1000000,-1000000]), 
                                            high=np.array([3000000,3000000,3000000,3000000]), dtype = np.float32),
            "action_space": gym.spaces.Box(0.5, 0.9, dtype = np.float32),
            "env_config": insurance_args,
            "evaluation_interval": 2,
            "evaluation_duration": 20,
        }
        self.trainer = PPO(config=config)

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
                "catastrophes": self.catastrophes,
                "broker_risks": self.broker_risks,
                "fair_market_premium": self.fair_market_premium,
                "risk_model_configs": self.risk_model_configs,
                "with_reinsurance": self.with_reinsurance,
                "num_risk_models": self.num_risk_models,
                "logger": self.logger}
        self.ppo_trainer_creator(insurance_args)
        env = MultiAgentBasedModel(**insurance_args)
    
        total_steps = 0
        terminated_dict = {"__all__": False}
    
        obs_dict, info_dict = env.reset()

        while not terminated_dict["__all__"]:
        
            action_dict = env.get_actions(total_steps)  
            total_steps += 1
            print(total_steps)

            # Add new syndicates with market entry probability every year
            new_syndicate = None
            if total_steps % 12 == 0:
                num_syndicates = len(self.syndicates)
                prob = self.syndicate_args["market_entry_probability"]
                np.random.seed(self.seed+total_steps)
                if np.random.random() < prob:
                    new_syndicate = Syndicate(str(num_syndicates), self.syndicate_args, self.num_risk_models, self.sim_args, self.risk_model_configs)
                    num_syndicates += 1

            obs_dict, reward_dict, terminated_dict, flag_dict, info_dict = env.step(action_dict, new_syndicate)

            # Save data for every step
            env.save_data()
        log = env.obtain_log()
        return log