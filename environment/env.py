import os
import random
import numpy as np
import cairosvg

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.envs.registration import EnvSpec
from gymnasium.error import DependencyNotInstalled

from environment.risk import CatastropheEvent, AttritionalLossEvent, AddRiskEvent, AddClaimEvent, RiskModel
from manager import EventHandler, EnvironmentManager

class SpecialtyInsuranceMarketEnv(gym.Env):
    """
    This environment corresponds to Syndicates Lloyd's of London 

    ### Action Space
    For each syndiacte, the offering line size for each contract

    ### Observation Space
    For each insurable risks, the accept or refuse status
    For each claim, the payment status, fullied paid or bankruptcy
    For each syndicate, the current capital, the capital in each risk region

    ### Rewards
    The accepted line size, the profit, and the bankruptcy

    ### Starting State
    All the activated agents holding their default settings

    ### Episode End
    The episode ends if any one of the following occurs:
    1. Termination: Simulation total time step reached 
    2. Termination: All the syndicates bankrupt

    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self,
                 sim_args,
                 manager_args,
                 brokers,
                 syndicates,
                 reinsurancefirms,
                 shareholders,
                 risk_models,
                 dt = 1,
                 maxstep = 30000,
                 reinsurance = False
                ):
        super(SpecialtyInsuranceMarketEnv, self).__init__()

        self.sim_args = sim_args
        self.manager_args = manager_args
        self.brokers = brokers
        self.initial_brokers = brokers
        self.syndicates = syndicates
        self.initial_syndicates = syndicates
        self.reinsurancefirms = reinsurancefirms
        self.initial_reinsurancefirms = reinsurancefirms
        self.shareholders = shareholders
        self.initial_shareholders = shareholders
        self.risk_models = risk_models
        self.initial_risk_models = risk_models
        self.dt = dt
        self.maxstep = maxstep
        self.em = None
        self.reinsurance = reinsurance
        self.catastrophe_events = {}
        self.attritional_loss_events = {}
        self.broker_risk_events = {}
        self.broker_claim_events = {}
        self.syndicate_active = {}

        # Reset the environmnet
        self.reset()

    def reset(self):
        
        # Reset the environment to an initial state
        self.brokers = self.initial_brokers
        self.syndicates = self.initial_syndicates
        self.reinsurancefirms = self.initial_reinsurancefirms
        self.shareholders = self.initial_shareholders
        self.risk_models = self.initial_risk_models

        # Initiate event handler
        catastrophe_events = CatastropheEvent(risk_models=self.risk_models).generate_risk_events()
        attritional_loss_events = AttritionalLossEvent(risk_models=self.risk_models).generate_risk_events()
        broker_risk_events = AddRiskEvent(risk_models=self.risk_models).generate_risk_events()
        broker_claim_events = AddClaimEvent(risk_models=self.risk_models).generate_risk_events()
        event_handler = EventHandler(catastrophe_events, attritional_loss_events, broker_risk_events, broker_claim_events, self.brokers, self.syndicates, self.reinsurancefirms, self.shareholders, self.risk_models)

        # Initiate environment manager
        self.em = EnvironmentManager(self.brokers, self.syndicates, self.reinsurancefirms, self.shareholders, self.risks, catastrophe_events, attritional_loss_events, broker_risk_events, broker_claim_events, event_handler)
        self.em.evolve(self.dt)
        
        # Set per syndicate active status and build status list
        self.syndicate_list = []
        for sy in self.em.environment.syndicate:
            self.syndicate_active[sy] = True
            self.syndicate_list.append(sy)

        # Set per insurable risks
        for risk_id in range(self.risk_args["num_risks"]):
            self.broker_risk_event[risk_id] = None

        # Set per claim events
        for claim_id in range(self.risk_args["num_risks"]):
            self.broker_claim_event[claim_id] = None
        
        # Initiate action list for syndicates, no synidactes joined the market in the beginning
        self.action_list = []
        for syndicate in self.em.environment.syndicate:
            self.action_list.append(0)

        # Initiate time step
        self.timestep = -1
        self.step_track = 0
        self.log = []

        return self.state_encoder()
        
    def step(self, action):
        
        log = {}  

        # Update environemnt after actions
        self.send_action2env(action)
        environment = self.em.evolve(self.dt)

        # Time
        self.timestep += 1

        # Get next observation
        obs = self.state_encoder()

        # Compute rewards
        reward = self.compute_reward()

        # Check termination status
        done = self.check_termination()

        # Update Plot 
        self.draw2file(environment)

        return obs, reward, done, log

    def draw2file(self, environment):

        # For visualisation  
        # Show syndaites catastrophe region (one dot represents £1000000), syndicates capital (£1000000 represents 1%) and time step

        self.step_track += 1

    def check_termination(self):

        # Update per syndicate status, True-Not bankrupt, False-Bankrupt
        for syndicate in self.em.environment.syndicate:
            if syndicate.update_capital() < 0:
                self.syndicate_active[syndicate] = False

        # The simulation is done when all syndicates bankrupt or reach the maximum time step
        run_complete = True
        for syndicate in self.em.environment.syndicate:
            if self.syndicate_active[syndicate] == True:
                run_complete = False
                break
        if run_complete or (self.timestep >= self.maxstep):
            done = True
        else:
            done = False

        return done

    def compute_reward(self, action):

        environment = self.em.environment
        # calculate reward function
        r = [0.0] * 3

        # Offering line size is accepted
        if(self.timestep <= self.maxstep):
            for sy in environment.syndicate:
                if self.syndicate_active[sy]:
                    syndicate = environment.syndicate[sy]
                    if syndicate.leader:
                        r[0] += syndicate.lead_line_size *
                    elif sydicate.follower:
                        r[0] += syndicate.follow_line_size *
                    else:
                        r[0] -= 100

        # Profit        
        if(self.timestep <= self.maxstep):
            for sy in environment.syndicate:
                if self.syndicate_status[sy]:
                    syndicate = environment.syndicate[sy]
                    initial_capital = syndicate.initial_capital
                    current_capital = syndicate.update_capital()
                    r[1] += current_capital - initial_capital
        
        # Bankrupt
        if(self.timestep >= self.maxstep):
            for sy in environment.syndicate:
                if not self.syndicate_active[sy]:
                    r[2] -= 100000
        

        # Sum reward
        reward = 0.0
        reward += np.sum(r)

        return reward

    def send_action2env(self, action):               
            
        # Apply action
        # Note that em.receive_actions caches actions until em.evolve is called in step
        self.em.receive_actions(actions=action)        
    
    def state_encoder(self):
                        
        obv = []
        environment = self.em.environment
        # Insurable risks: the accept or refuse status
        for risk_id in range(self.risk_args["num_risks"]):
            risk = environment.broker_risk_event[risk_id]
            obv.append(risk[1])  # risk[id, status], status is 0 pr 1

        # Claims: the payment status, fullied paid or bankruptcy
        for claim_id in range(self.risk_args["num_risks"]):
            claim = environment.broker_claim_event[claim_id]
            obv.append(claim[1])  # claim[id, status], status is 0 or 1

        for sy in self.syndicate_list:
            syndicate = environment.syndicate[sy]
            # Syndicates: the current capital, the capital in each risk region
            current_capital = syndicate.update_capital()
            obv.append(current_capital)
            regions = syndicate.update_underwrite_risk_regions()
            for r in range(len(regions))
                obv.append(regions[r])
            
        return np.array(obv, dtype = np.float32)
   