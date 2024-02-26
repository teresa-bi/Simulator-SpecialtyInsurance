import os
import ray
from ray.tune.registry import register_env
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ipywidgets import IntProgress
from environment.environment import SpecialtyInsuranceMarketEnv

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

    def env_creator(self, env_config):
        """
        Register specialty insurance market environment for The rllib
        """

        return SpecialtyInsuranceMarketEnv(**env_config)

    def ppo_trainer_creator(self, insurance_args):
        """
        Choose PPO Algorithm for Training
        """
    
        config={"log_level": "ERROR",
            "env": "SpecialtyInsuranceMarket-validation",
            "num_workers": 2,
            "framework": "tf",
            "evaluation_interval": 2, #num of training iter between evaluation
            "evaluation_duration": 20,
            "num_gpus": 0,
            "env_config": insurance_args}
        self.trainer = PPO(config=config)

    def initial_training(self, top_dir, n):
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
        convergence_track = []

        for i in range(num_episode):
            result = self.trainer.train()        
            print("Progress:", i+1, "/", num_episode, end="\r")
            bar.value += 1
            convergence_track.append(result["episode_reward_mean"])
            if i % 10 == 0:
                self.trainer.save(model_filepath)
            if i % 10 == 0:
                plt.plot(convergence_track)
                plt.show()

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

    def compute_rewards(self, x):
        """
        Compute total rewards
        """

        args = {
            "": x[0],
            "": x[1],
            "": x[2],
            "": x[3],
        }      #####TODO: auguments to be analysed in specialty insruance market simlation, forexample, incentive

        validation_episodes = 1
        env = SpecialtyInsuranceMarket(**args)
        total_steps = 0
        for epi in range(validation_episodes):
            obs = env.reset()
            done = False
            total_reward = 0
    
            print(f"\nepisode: {epi} | ")
            while not done:
                if total_steps % 20 == 0: print(".", end="")   
                action = self.trainer.compute_single_action(obs)   
                total_steps += 1    
                obs, reward, done, info = env.step(action)
                #env.render()
                total_reward += reward
            print("Done")
            print("Reward: ", total_reward)
            #env.close()

        return total_reward

    def testing(self):
        """
        Test the training performance
        """

        insurance_args = {"simulation_args": self.sim_args,
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
        env = SpecialtyInsuranceMarket(**insurance_args)
        total_steps = 0
        for epi in range(validation_episodes):
            obs = env.reset()
            done = False
            total_reward = 0
    
            print(f"\nepisode: {epi} | ")
            while not done:
                if total_steps % 20 == 0: 
                    print(".", end="")   
                action = self.trainer.compute_single_action(obs)   
                total_steps += 1    
                obs, reward, done, info = env.step(action)
                #env.render()
                total_reward += reward
            print("Done")
            print("Reward: ", total_reward)
            #env.close()

        return total_reward

    def robust_training(self, scenario_training, top_dir, n):
        """
        Train agent with collected worse scenario
        """
        # Create a path to store the trained agent for each iteration
        model_filepath = f"{top_dir}/{str(n)}/saved_models"
        num_episode = 10
        bar = IntProgress(min=0, max=num_episode)
        display(bar)
        for i in range(len(scenario_training)):
            self.trainer.config["env_config"] = scenario_training[i]
            for i in range(num_episode):
                result = self.trainer.train() 
                print("Progress:", i+1, "/", num_episode, end="\r")
                bar.value += 1
                if i % 10 == 0:
                    self.trainer.save(model_filepath)

    def run(self):
        # Folder for recording
        top_dir = "insurance_scenario_" + self.scenario + "_model_" + self.model

        # Register environment
        register_env("SpecialtyInsuranceMarket-validation", self.env_creator)

        # The number of training iteration for the RL agent
        num_training = 100

        is_initial = True
        for n in range(num_training):
            # Initialize the training policy pi_0
            if is_initial:
                # Initial arguments to define Specialty Insurance Market scenario 
                insurance_args = {"simulation_args": self.sim_args,
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
        
                # Return the trained trainer
                self.initial_training(top_dir, n)
        
                # Test the performance of the trained agent
                rewards = self.testing()
                is_initial = False
            else:
                # For each training agent, initialize a training set
                scenario_training = []
                rewards_training = []
        
                # Then train
                self.trainer_restore(top_dir, n)
        
                self.robust_training(scenario_training[:-10], top_dir, n)
        
                # Test the performance of the trained agent
                rewards = self.testing(self.trainer)
                
