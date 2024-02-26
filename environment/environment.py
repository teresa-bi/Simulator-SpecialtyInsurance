import os
import random
import numpy as np
import cairosvg

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.envs.registration import EnvSpec
from gymnasium.error import DependencyNotInstalled
from agents import Broker, Syndicate, ReinsuranceFirm, Shareholder
from environment.event_generator import EventGenerator
from manager import *

class SpecialtyInsuranceMarketEnv(gym.Env):
    """
    This environment corresponds to Syndicates Lloyd's of London 

    ### Action Space
    For each syndiacte, the offering line size for each contract

    ### Observation Space
    For each syndicate, the active or exit status, the remaining capital amount in each risk category
    For each catastrophe, the risk category and the risk value

    ### Rewards
    For each insurable risk being accepted or refused
    For each claim being paied or refused bacaused of bankruptcy
    For each syndicate, the profit +int or loss -int,
    For each syndicate, bankruptcy

    ### Starting State
    All the activated agents holding their default settings, and all the risks 

    ### Episode End
    The episode ends if any one of the following occurs:
    1. Termination: Simulation total time step reached 
    2. Termination: All the syndicates bankrupt

    """

    def __init__(self, sim_args, manager_args, brokers, syndicates, reinsurancefirms, shareholders, risks, risk_model_configs, with_reinsurance, num_risk_models, dt = 1):

        super(SpecialtyInsuranceMarketEnv, self).__init__()

        self.sim_args = sim_args
        self.maxstep = self.sim_args["max_time"]
        self.manager_args = manager_args
        self.brokers = brokers
        self.initial_brokers = brokers
        self.syndicates = syndicates
        self.initial_syndicates = syndicates
        self.reinsurancefirms = reinsurancefirms
        self.initial_reinsurancefirms = reinsurancefirms
        self.shareholders = shareholders
        self.initial_shareholders = shareholders
        self.risks = risks
        self.initial_risks = risks
        self.risk_model_configs = risk_model_configs
        self.with_reinsurance = with_reinsurance
        self.num_risk_models = num_risk_models
        self.dt = dt
        self.mm = None
        # Active syndicate list
        self.syndicate_active_list = []
        # Action list
        self.action_list = []
        # Catastrophe event at time t, including risk category and risk value
        self.catastrophe = []

        # Reset the environmnet
        self.reset()

    def reset(self):
        
        # Reset the environment to an initial state
        self.brokers = self.initial_brokers
        self.syndicates = self.initial_syndicates
        self.reinsurancefirms = self.initial_reinsurancefirms
        self.shareholders = self.initial_shareholders
        self.risks = self.initial_risks

        # Catastrophe event 
        catastrophe_events = EventGenerator(self.risk_model_configs).generate_catastrophe_events(self.risks)
        # Attritioal loss event daily
        attritional_loss_events = EventGenerator(self.risk_model_configs).generate_attritional_loss_events(self.sim_args, self.risks)
        # Broker risk event daily: broker generate risk according to poisson distribution
        broker_risk_events = EventGenerator(self.risk_model_configs).generate_risk_events(self.brokers)
        # Broker premium event monthly: broker pay premium to the syndicate 
        broker_premium_events = EventGenerator(self.risk_model_configs).generate_premium_events(self.brokers)
        # Broker claim event: when catastrophe happens, croker brings corresponding claim to syndicate 
        broker_claim_events = EventGenerator(self.risk_model_configs).generate_claim_events(self.brokers)
        # Initiate event handler
        event_handler = EventHandler(self.maxstep, catastrophe_events, attritional_loss_events, broker_risk_events, broker_premium_events, broker_claim_events)

        # Initiate market manager
        self.mm = MarketManager(self.maxstep, self.manager_args, self.brokers, self.syndicates, self.reinsurancefirms, self.shareholders, self.risks, self.risk_model_configs, self.with_reinsurance, self.num_risk_models, catastrophe_events, attritional_loss_events, broker_risk_events, broker_premium_events, broker_claim_events, event_handler)
        self.mm.evolve(self.dt)
        
        # Set per syndicate active status and build status list
        self.syndicate_active_list = []   # Store syndicates currently in the market
        for sy in self.mm.market.syndicates:
            if self.mm.market.syndicates[sy].status == True:
                self.syndicate_active_list.append(sy)
        
        # Initiate action list for syndicates, no syndicate joined the market at the begining, but set action for all the syndicates becuase RL require the same size action size, for exit syndicate and for no action, the actions are 0
        self.action_list = []
        # Cannot know the number of risks brought by brokers each day, so set a max number and the action size is fixed
        max_number_risks_per_day = 100
        for sy in self.syndicate_active_list:
            for r in max_number_risks_per_day:
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
        market = self.mm.evolve(self.dt)

        # Time
        self.timestep += 1

        # Get next observation
        obs = self.state_encoder()

        # Compute rewards
        reward = self.compute_reward()

        # Check termination status
        done = self.check_termination()

        # Update Plot 
        self.draw2file(market)

        return obs, reward, done, log

    def draw2file(self, market):

        # For visualisation  
        # Show syndaites catastrophe category (one dot represents £1000000), syndicates capital (£1000000 represents 1%) and time step

        self.step_track += 1

    def check_termination(self):

        # Update per syndicate status, True-active in market, False-exit market becuase of no contract or bankruptcy
        market = self.mm.market
        for sy in market.syndicates:
            if market.syndicates[sy].status == False:
                self.syndicate_active[sy] = False
                del self.syndicate_active_list[sy]

        # The simulation is done when all syndicates exit or bankrupt or reach the maximum time step
        run_complete = True
        for sy in market.syndicates:
            if self.syndicate_active[sy] == True:
                run_complete = False
                break
        if run_complete or (self.timestep >= self.maxstep):
            done = True
        else:
            done = False

        return done

    def compute_reward(self, action):

        market = self.mm.market
        # calculate reward function
        r = [0.0] * 4

        # For each insurable risk being accepted +1 or refused -1
        if(self.timestep <= self.maxstep):
            for risk in range(len(market.brokers.risks)):
                for contract in range(len(market.brokers.underwritten_contracts)):
                    if market.brokers.risks[risk]["risk_id"] == market.brokers.underwritten_contracts[contract]["risk_id"]:
                        r[0] += 1
                    else:
                        r[0] -= 1

        # For each claim being paied +1 or refused -1
        if(self.timestep <= self.maxstep):
            for contract in range(len(market.brokers.underwritten_contracts)):
                if market.brokers.underwritten_contracts[contract]["claim"] == True:
                    r[1] += 1
                else:
                    r[1] -= 1

        # Profit and Bankruptcy       
        if(self.timestep <= self.maxstep):
            for sy in market.syndicates:
                if self.syndicate_status[sy]:
                    syndicate = market.syndicates[sy]
                    initial_capital = syndicate.initial_capital
                    current_capital = syndicate.update_capital()
                    r[2] += current_capital - initial_capital
                    if (current_capital - initial_capital) < 0:
                        r[3] -= 10000

        # Sum reward
        reward = 0.0
        reward += np.sum(r)

        return reward

    def send_action2env(self, action):               
            
        # Apply action
        # Note that mm.receive_actions caches actions until mm.evolve is called in step
        self.mm.receive_actions(actions=action)        
    
    def state_encoder(self):

        ### Observation Space: TODO: size is not fixed, may need language model for the coding               
        obv = []
        market = self.mm.market
        for risk in range(len(market.risks)):
            if market.risks[risk]["risk_start_time"] == self.timestep:
                self.catastrophe.append(market.risks[risk])
        # Catastrophe risk category and risk value
        for i in range(len(self.catastrophe)):
            obv.append(self.catastrophe[i]["risk_category"])
            obv.append(self.catastrophe[i]["risk_value"])

        # Syndicates status current capital in 
        for i in range(len(self.syndicate_active_list)):
            obv.append(self.syndicate_active_list[i]["current_capital"])
            for num in range(len(self.syndicate_active_list[i]["current_capital_category"])):
                obv.append(self.syndicate_active_list[i]["current_capital_category"][num])
            
        return np.array(obv, dtype = np.float32)
   