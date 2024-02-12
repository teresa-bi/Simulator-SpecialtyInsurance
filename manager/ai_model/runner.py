import ray
from ray.tune.registry import register_env
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ipywidgets import IntProgress
from environment.env import SpecialtyInsuranceMarketEnv

class AIRunner:
    """
    AI model training and testing steps
    """
    def __init__(self, sim_args, manager_args, brokers, syndicates, reinsurancefirms, shareholders, risk_models, scenario, model):
        self.sim_args = sim_args
        self.manager_args = manager_args
        self.brokers = brokers
        self.syndicates = syndicates
        self.reinsurancefirms = reinsurancefirms
        self.shareholders = shareholders
        self.risk_models = risk_models
        self.env = None
        self.scenario = scenario
        self.model = model

    def env_creator(self, env_config):
        """
        Register specialty insurance market environment for The rllib
        """

        return SpecialtyInsuranceMarketEnv(**env_config)

    def ppo_trainer_creator(insurance_args):
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
        trainer = PPOTrainer(config=config)

        return trainer

    def initial_training(trainer, top_dir, n):
        """
        Initial Training Iteration for the PPO Trainer
        """
        # Create a path to store the trained agent for each iteration
        model_filepath = f"{top_dir}/{str(n)}/saved_models"
        
        num_episode = 10

        # Run it for n training iterations. A training iteration includes
        # parallel sample collection by the environment workers as well as
        # loss calculation on the collected batch and a model update.

        bar = IntProgress(min=0, max=num_episode)
        display(bar)
        convergence_track = []

        for i in range(num_episode):
            result = trainer.train()        
            print("Progress:", i+1, "/", num_episode, end="\r")
            bar.value += 1
            convergence_track.append(result["episode_reward_mean"])
            if i % 10 == 0:
                trainer.save(model_filepath)
            if i % 10 == 0:
                plt.plot(convergence_track)
                plt.show()
        
        return trainer

    def trainer_restore(trainer, top_dir, n):
        """
        Restore the trainer from the last iteration
        """

        trainer.restore(f"{top_dir}/{str(n-1)}/saved_models/checkpoint_000001/rllib_checkpoint.json")

        return trainer

    def compute_rewards(trainer, x):
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
                action = trainer.compute_single_action(obs)   
                total_steps += 1    
                obs, reward, done, info = env.step(action)
                #env.render()
                total_reward += reward
            print("Done")
            print("Reward: ", total_reward)
            #env.close()

        return total_reward

    def testing(trainer):
        """
        Test the training performance
        """

        np.random.seed(234)
        args = {
            "": np.float32(np.random.uniform(-4.8, 4.8)), 
            "": np.float32(np.random.uniform(-2e10, 2e10)),
            "": np.float32(np.random.uniform(-0.418, 0.418)),
            "": np.float32(np.random.uniform(-2e10, 2e10)),
        }

        validation_episodes = 1
        env = SpecialtyInsuranceMarket(**args)
        total_steps = 0
        for epi in range(validation_episodes):
            obs = env.reset()
            done = False
            total_reward = 0
    
            print(f"\nepisode: {epi} | ")
            while not done:
                if total_steps % 20 == 0: 
                    print(".", end="")   
                action = trainer.compute_single_action(obs)   
                total_steps += 1    
                obs, reward, done, info = env.step(action)
                #env.render()
                total_reward += reward
            print("Done")
            print("Reward: ", total_reward)
            #env.close()

        return total_reward

    def robust_training(trainer, scenario_training, top_dir, n):
        """
        Train agent with collected worse scenario
        """
        # Create a path to store the trained agent for each iteration
        model_filepath = f"{top_dir}/{str(n)}/saved_models"
        num_episode = 10
        bar = IntProgress(min=0, max=num_episode)
        display(bar)
        for i in range(len(scenario_training)):
            trainer.config["env_config"] = scenario_training[i]
            for i in range(num_episode):
                result = trainer.train() 
                print("Progress:", i+1, "/", num_episode, end="\r")
                bar.value += 1
                if i % 10 == 0:
                    trainer.save(model_filepath)
        trainer.restore(f"{top_dir}/{str(n)}/saved_models/checkpoint_000001/rllib_checkpoint.json")
        
        return trainer

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
                        "management_args": self.manager_args,
                        "brokers": self.brokers,
                        "syndicates": self.syndicates,
                        "reinsurancefirms": self.reinsurancefirms,
                        "shareholders": self.shareholders,
                        "risk_models": self.risk_models}
                trainer = self.ppo_trainer_creator(insurance_args)
        
                # Return the trained trainer
                trainer = self.initial_training(trainer, top_dir, n)
        
                # Test the performance of the trained agent
                rewards = self.testing(trainer)
                is_initial = False
            else:
                # For each training agent, initialize a training set
                scenario_training = []
                rewards_training = []
        
                # Then train
                trainer = self.trainer_restore(trainer, top_dir, n)
        
                trainer = robust_training(trainer, scenario_training[:-10], top_dir, n)
        
                # Test the performance of the trained agent
                rewards = self.testing(trainer)
                
