import os
import random
import numpy as np
import cairosvg

import gym
import ray
from ray.rllib.utils.typing import MultiAgentDict
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from agents import Broker, Syndicate, ReinsuranceFirm, Shareholder
from environment.event_generator import EventGenerator
from manager.ai_model.action import Action
from manager import *

class SpecialtyInsuranceMarketEnv(MultiAgentEnv):
    """
    This environment corresponds to Syndicates Lloyd's of London 

    ### Action Space
    For each syndiacte, the offering line size for each contract and corresponding price

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
        self.event_handler = None

        # Active syndicate list
        self.syndicate_active_list = []
        # Initialise events, actions, and states 
        self.catastrophe_events = []
        self.attritional_loss_events = []
        self.broker_risk_events = []
        self.broker_premium_events = []
        self.broker_claim_events = []
        self.action_map_dict = {}
        self.state_encoder_dict = {}
        self.step_track = 0

        # Define Action Space, Define Observation Space
        self.n = len(self.syndicates)
        self.agents = {self.syndicates[i].syndicate_id for i in range(self.n)}
        self._agent_ids = set(self.agents)
        self.dones = set()
        self._spaces_in_preferred_format = True
        self.observation_space = gym.spaces.Dict({self.syndicates[i].syndicate_id: self.obs_space_creator() for i in range(self.n)})
        self.action_space = gym.spaces.Dict({self.syndicates[i].syndicate_id: self.set_action_space() for i in range(self.n)})

        super(SpecialtyInsuranceMarketEnv, self).__init__()

        # Reset the environmnet
        self.reset()

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        
        # Reset the environment to an initial state
        self.brokers = self.initial_brokers
        self.syndicates = self.initial_syndicates
        self.reinsurancefirms = self.initial_reinsurancefirms
        self.shareholders = self.initial_shareholders
        self.risks = self.initial_risks

        # Catastrophe event 
        self.catastrophe_events = EventGenerator(self.risk_model_configs).generate_catastrophe_events(self.risks)
        # Attritioal loss event daily
        self.attritional_loss_events = EventGenerator(self.risk_model_configs).generate_attritional_loss_events(self.sim_args, self.risks)
        # Broker risk event daily: broker generate risk according to poisson distribution
        self.broker_risk_events = EventGenerator(self.risk_model_configs).generate_risk_events(self.brokers, self.risks)
        # Broker pay premium according to underwritten contracts
        self.broker_premium_events = []
        # Broker ask for claim if the contract affected by catastrophe
        self.broker_claim_events = []
        self.event_handler = EventHandler(self.maxstep, self.catastrophe_events, self.attritional_loss_events, self.broker_risk_events, self.broker_premium_events, self.broker_claim_events)

        # Initiate market manager
        self.mm = MarketManager(self.maxstep, self.manager_args, self.brokers, self.syndicates, self.reinsurancefirms, self.shareholders, self.risks, self.risk_model_configs, self.with_reinsurance, self.num_risk_models, self.catastrophe_events, self.attritional_loss_events, self.broker_risk_events, self.broker_premium_events, self.broker_claim_events, self.event_handler)
        self.mm.evolve(self.dt)
        
        # Set per syndicate active status and build status list
        self.syndicate_active_list = []   # Store syndicates currently in the market
        for syndicate_id in self.mm.market.syndicates:
            if self.mm.market.syndicates[syndicate_id].status == True:
                self.syndicate_active_list.append(syndicate_id)

        # Create action map and state list
        info_dict = {}
        for syndicate_id in self.syndicate_active_list:
            self.action_map_dict[syndicate_id] = self.action_map_creator(self.mm.market.syndicates[syndicate_id], 0)
            #self.state_encoder_dict[syndicate_id] = self.state_encoder(syndicate_id)
            self.state_encoder_dict = {i: self.observation_space[i].sample() for i in self.agents}
            info_dict[syndicate_id] = self._get_info()

        # Initiate time step
        self.timestep = -1
        self.step_track = 0
        self.log = []

        return self.state_encoder_dict, info_dict
        
    def step(self, action_dict):

        obs_dict, reward_dict, terminated_dict, info_dict = {}, {}, {}, {}
        flag_dict = {}

        # Update environemnt after actions
        parsed_actions = []        
        for syndicate_id, action in action_dict.items():
            # update action map
            self.action_map = self.action_map_creator(self.mm.market.syndicates[syndicate_id],action)
            parsed_ac2add = self.action_map
            parsed_actions.append(parsed_ac2add)
        
        self.send_action2env(parsed_actions)

        # Update broker_premium_events, broker_claim_events, event_handler, market manager
        self.broker_premium_events = EventGenerator(self.risk_model_configs).generate_premium_events(self.brokers, self.timestep)
        self.event_handler.add_premium_events(self.broker_premium_events)
        for i in range(len(self.catastrophe_events)):
            if self.catastrophe_events[i].risk_start_time == self.timestep:
                self.broker_claim_events = EventGenerator(self.risk_model_configs).generate_claim_events(self.brokers, self.timestep)
                self.event_handler.add_claim_events(self.broker_claim_events)
        self.mm.update_premium_events(self.broker_premium_events, self.event_handler)
        self.mm.update_claim_events(self.broker_claim_events, self.event_handler)
        
        market = self.mm.evolve(self.dt)
        self.timestep += 1

        # Compute rewards and get next observation
        for syndicate_id, action in action_dict.items():
            reward_dict[syndicate_id] = self.compute_reward(action, syndicate_id)
            obs_dict[syndicate_id]= self.state_encoder(syndicate_id)
            info_dict[syndicate_id] = self._get_info()
            flag_dict[syndicate_id] = False
        
        # Check termination
        for syndicate_id in self.syndicate_active_list:
            terminated_dict[syndicate_id] = self.check_termination(syndicate_id)

        # Update plot 
        self.draw2file(market)

        # All done termination check
        all_terminated = True
        for _, syndicate_terminated in terminated_dict.items():
            if syndicate_terminated is False:
                all_terminated = False
                break
        
        terminated_dict["__all__"] = all_terminated

        return obs_dict, reward_dict, terminated_dict, flag_dict, info_dict

    def draw2file(self, market):

        # For visualisation  
        # Show syndaites catastrophe category (one dot represents £1000000), syndicates capital (£1000000 represents 1%) and time step

        self.step_track += 1

    def check_termination(self, syndicate_id):

        # Update per syndicate status, True-active in market, False-exit market becuase of no contract or bankruptcy
        market = self.mm.market
        sy = market.syndicates[syndicate_id] 
        if sy.status == False:
            self.syndicate_active[syndicate_id] = False
            del self.syndicate_active_list[syndicate_id]

        # The simulation is done when syndicates exit or bankrupt or reach the maximum time step
        run_complete = True
        if self.syndicate_active[syndicate_id] == True:
            run_complete = False
        if run_complete or ((self.timestep+1) >= self.maxstep):
            terminated = True
        else:
            terminated = False

        return terminated

    def compute_reward(self, action, syndicate_id):

        market = self.mm.market
        # calculate reward function
        r = [0.0] * 4

        # For each insurable risk being accepted +1 or refused -1
        if(self.timestep <= self.maxstep):
            for broker_id in range(len(market.brokers)):
                for risk in range(len(market.brokers[broker_id].risks)):
                    for contract in range(len(market.brokers[broker_id].underwritten_contracts)):
                        if market.brokers[broker_id].risks[risk]["risk_id"] == market.brokers[broker_id].underwritten_contracts[contract]["risk_id"]:
                            r[0] += 1
                        else:
                            r[0] -= 1

        # For each claim being paied +1 or refused -1
        if(self.timestep <= self.maxstep):
            for claim in range(len(market.syndicate[syndicate_id].paid_claim)):
                if market.syndicate[syndicate_id].paid_claim[claim]["status"] == True:
                    r[1] += 1
                else:
                    r[1] -= 1

        # Profit and Bankruptcy       
        if(self.timestep <= self.maxstep):
            initial_capital = self.syndicate_status[syndicate_id].initial_capital
            current_capital = self.syndicate_status[syndicate_id].update_capital()
            r[2] += current_capital - initial_capital
            if (current_capital - initial_capital) < 0:
                r[3] -= 10000

        # Sum reward
        reward = 0.0
        reward += np.sum(r)

        return reward

    def send_action2env(self, parsed_actions):               
            
        # Apply action
        # Note that mm.receive_actions caches actions until mm.evolve is called in step
        if len(parsed_actions) > 0:
            self.mm.receive_actions(actions=parsed_actions)          
    
    def state_encoder(self, syndicate_id):

        ### Observation Space:             
        obs = []
        market = self.mm.market
        for risk in range(len(market.risks)):
            if market.risks[risk]["risk_start_time"] == self.timestep:
                # Catastrophe risk category and risk value
                obs.append(market.risks[risk]["risk_category"])
                obs.append(market.risks[risk]["risk_value"])

        # Syndicates status current capital in 
        obs.append(self.syndicates[syndicate_id]["current_capital"])
        for num in range(len(self.syndicates[syndicate_id].current_capital_category)):
            obs.append(self.syndicates[syndicate_id].current_capital_category[num])
            
        return obs

    def obs_space_creator(self):
        low, high = [], []
        # risk_category, risk_value, current capital for each category for each syndicate
        low.extend([0.0, 0.0])
        high.extend([10.0, 10000000.0]) # Number of risk category, risk limit, current capital
        for num in range(self.risk_model_configs[0]["num_categories"]):
            low.append(-10000000.0)
            high.append(30000000.0)
        observation_space = gym.spaces.Box(np.array(low, dtype=np.float32), np.array(high, dtype=np.float32)) 
        return observation_space

    def action_map_creator(self, syndicate, line_size):

        for r in range(len(syndicate.received_risk_list)):
            if syndicate.received_risk_list["start_time"] == self.timestep:
                action_map = Action(syndicate.syndicate_id, line_size, syndicate.received_risk[r]["risk_id"], syndicate.received_risk[r]["broker_id"])
       
        return action_map

    def set_action_space(self):
        return gym.spaces.Box(0.5, 0.9, dtype = np.float32)
   