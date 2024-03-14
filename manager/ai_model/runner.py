import os
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO
from ipywidgets import IntProgress
from environment.environment import SpecialtyInsuranceMarketEnv
from ray.rllib.policy.policy import PolicySpec
import numpy as np
import gymnasium as gym
from ray import air, tune
from ray.rllib.examples.policy.random_policy import RandomPolicy

class AIRunner:
    """
    AI model training and testing steps
    """
    def __init__(self, sim_args, manager_args, brokers, syndicates, reinsurancefirms, shareholders, risks, risk_model_configs, with_reinsurance, num_risk_models):
        self.sim_args = sim_args
        self.manager_args = manager_args
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

    def env_creator(self, env_config):
        """
        Register specialty insurance market environment for The rllib
        """

        return SpecialtyInsuranceMarketEnv(**env_config)

    def policy_mapping_fn(self, agent_id, episode, worker, **kwargs):
        # agent0 -> main0
        # agent1 -> main1
        return f"main{agent_id[-1]}"

    def ppo_trainer_creator(self, insurance_args):
    
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

    def training(self, top_dir, n):
        """
        Initial Training Iteration for the PPO Trainer
        """
        # Create a path to store the trained agent for each iteration
        model_filepath = f"{top_dir}/{str(n)}/saved_models"
        
        num_episode = 10

        # A training iteration includes parallel sample collection by the environment workers 
        # as well as loss calculation on the collected batch and a model update.

        bar = IntProgress(min=0, max=num_episode)
        display(bar)
        list_mean_rewards = []
        list_min_rewards = []
        list_max_rewards = []
        list_train_step = []

        for i in range(num_episode):
            self.trainer.train()     
            print("Progress:", i+1, "/", num_episode, end="\r")
            bar.value += 1
            if (i+1) % 2 == 0:
                list_mean_rewards.append(trainer.evaluation_metrics["evaluation"]["episode_reward_mean"])
                list_min_rewards.append(trainer.evaluation_metrics["evaluation"]["episode_reward_min"])
                list_max_rewards.append(trainer.evaluation_metrics["evaluation"]["episode_reward_max"])
                list_train_step.append(i+1)
            if i % 10 == 0:
                self.trainer.save(model_filepath)

    def trainer_restore(self, top_dir, n):
  
        if n <= 9:
            path0 = top_dir
            path1 = str(n-1)
            path2 = "saved_models"
            path3 = "checkpoint_"+str(0)+str(0)+str(0)+str(0)+str(0)+str(n)
            path4 = "rllib_checkpoint.json"
        elif 9 < n <= 99:
            path0 = top_dir
            path1 = str(n-1)
            path2 = "saved_models"
            path3 = "checkpoint_"+str(0)+str(0)+str(0)+str(0)+str(n)
            path4 = "rllib_checkpoint.json"
        elif 99 < n <= 999:
            path0 = top_dir
            path1 = str(n-1)
            path2 = "saved_models"
            path3 = "checkpoint_"+str(0)+str(0)+str(0)+str(n)
            path4 = "rllib_checkpoint.json"

        # Join various path components

        self.trainer.restore(os.path.join(path0, path1, path2, path3, path4))

    def testing(self):
        """
        Test the training performance
        """

        insurance_args = {"sim_args": self.sim_args,
                "manager_args": self.manager_args,
                "brokers": self.brokers,
                "syndicates": self.syndicates,
                "reinsurancefirms": self.reinsurancefirms,
                "shareholders": self.shareholders,
                "risks": self.risks,
                "risk_model_configs": self.risk_model_configs,
                "with_reinsurance": self.with_reinsurance,
                "num_risk_models": self.num_risk_models}

        validation_episodes = 1
        
        for epi in range(validation_episodes):
            env = SpecialtyInsuranceMarket(**insurance_args)
    
            print(f"\nepisode: {epi} | ")
            total_steps = 0
            done = {"__all__": False}
            all_rewards[epi] = {}
    
            obs = env.reset()
    
            while not done["__all__"]:
                if total_steps % 20 == 0: print(".", end="")
        
                action_dict = trainer.compute_actions(obs)  
                total_steps += 1
        
                obs, reward, done, info = env.step(action_dict, 
                                          draw_to_file=True)
                for k, v in reward.items():
                    if k not in all_rewards[epi]:
                        all_rewards[epi][k] = [v]
                    else:
                        all_rewards[epi][k].append(v)

        return all_rewards

    def run(self):
        # Folder for recording
        top_dir = "insurance_scenario_" + self.scenario + "_model_" + str(self.num_risk_models)

        # Register environment
        register_env("SpecialtyInsuranceMarket-validation", self.env_creator)

        # The number of training iteration for the RL agent
        num_training = 10

        insurance_args = {"sim_args": self.sim_args,
                        "manager_args": self.manager_args,
                        "brokers": self.brokers,
                        "syndicates": self.syndicates,
                        "reinsurancefirms": self.reinsurancefirms,
                        "shareholders": self.shareholders,
                        "risks": self.risks,
                        "risk_model_configs": self.risk_model_configs,
                        "with_reinsurance": self.with_reinsurance,
                        "num_risk_models": self.num_risk_models}
        self.ppo_trainer_creator(insurance_args)

        for n in range(num_training):
            if n == 0:
        
                # Return the trained trainer
                self.training(top_dir, n)
        
                # Test the performance of the trained agent
                #rewards = self.testing()
            else:
        
                # Then train
                self.trainer_restore(top_dir, n)
        
                self.training(top_dir, n)
        
                # Test the performance of the trained agent
                #rewards = self.testing(self.trainer)
                
