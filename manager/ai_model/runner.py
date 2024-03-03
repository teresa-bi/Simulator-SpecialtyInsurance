import os
import ray
from ray.tune.registry import register_env
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ipywidgets import IntProgress
from environment.environment import SpecialtyInsuranceMarketEnv
from ray.rllib.policy.policy import PolicySpec
from gymnasium.spaces import Box
import numpy as np

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

    def ppo_trainer_creator(self, insurance_args):
        """
        Choose PPO Algorithm for Training
        """
        low, high = [], []
        n = len(self.syndicates)
        low.extend([0.0, 0.0])
        high.extend([10.0, 10000000.0]) # Number of risk category, risk limit, current capital
        for num in range(self.risk_model_configs[0]["num_categories"]):
            low.append(-10000000.0)
            high.append(30000000.0)
        config={"log_level": "ERROR",
            "env": "SpecialtyInsuranceMarket-validation",
            "num_workers": 1,
            "framework": "tf",
            "model": {
                "fcnet_hiddens": [32, 16],
                "fcnet_activation": "relu",
                },
            "evaluation_interval": 2, #num of training iter between evaluation
            "evaluation_duration": 20,
            "num_gpus": 0,
            "multiagent": {
            "policies": {
                self.syndicates[i].syndicate_id: PolicySpec(observation_space=Box(np.array(low, dtype=np.float32), np.array(high, dtype=np.float32)), action_space=Box(0.5, 0.9, dtype = np.float32)) for i in range(n)
                
            },
            "policies_to_train": ["0"]
            },
            "env_config": insurance_args}
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
            if (i+1) % config["evaluation_interval"] == 0:
                list_mean_rewards.append(trainer.evaluation_metrics["evaluation"]["episode_reward_mean"])
                list_min_rewards.append(trainer.evaluation_metrics["evaluation"]["episode_reward_min"])
                list_max_rewards.append(trainer.evaluation_metrics["evaluation"]["episode_reward_max"])
                list_train_step.append(i+1)

    # Plot mean reward
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]
    def plot():
        x = list_train_step
        y = list_mean_rewards
        plot_utils.plot_eval(y, x)

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
                
