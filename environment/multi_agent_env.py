import gym
import numpy as np
import scipy
from environment.event_generator import EventGenerator
from manager.ai_model.action import Action
from manager import EventHandler, MarketManager
from logger import logger
from environment.risk_model import RiskModel
from environment.environment import SpecialtyInsuranceMarketEnv

class MultiAgentBasedModel(SpecialtyInsuranceMarketEnv):

    def __init__(self, sim_args, manager_args, broker_args, syndicate_args, reinsurancefirm_args, shareholder_args, risk_args, 
                 brokers, syndicates, reinsurancefirms, shareholders, catastrophes, broker_risks, fair_market_premium,
                 risk_model_configs, with_reinsurance, num_risk_models, logger, dt = 1):
        self.sim_args = sim_args
        self.maxstep = self.sim_args["max_time"]
        self.manager_args = manager_args
        self.broker_args = broker_args
        self.syndicate_args = syndicate_args
        self.reinsurancefirm_args = reinsurancefirm_args
        self.shareholder_args = shareholder_args
        self.risk_args = risk_args
        self.brokers = brokers
        self.syndicates = syndicates
        self.reinsurancefirms = reinsurancefirms
        self.shareholders = shareholders
        self.catastrophes = catastrophes
        self.broker_risks = broker_risks
        self.fair_market_premium = fair_market_premium
        self.market_premium = fair_market_premium
        self.initial_catastrophes = catastrophes
        self.risk_model_configs = risk_model_configs
        self.with_reinsurance = with_reinsurance
        self.num_risk_models = num_risk_models
        self.logger = logger
        self.dt = dt
        self.mm = None
        self.event_handler = None

        # Active syndicate list
        self.syndicate_active_list = []
        # Initialise events, actions, and states 
        self.attritional_loss_events = []
        self.catastrophe_events = []
        self.broker_risk_events = []
        self.broker_premium_events = []
        self.broker_claim_events = []
        self.action_map_dict = {}
        self.state_encoder_dict = {}

        # Define Action Space, Define Observation Space
        self.n = len(self.syndicates)
        self.agents = {self.syndicates[i].syndicate_id for i in range(self.n)} 
        self._agent_ids = set(self.agents)
        self.dones = set()
        self._spaces_in_preferred_format = True
        self.observation_space = gym.spaces.Dict({
            self.syndicates[i].syndicate_id: gym.spaces.Box(low=np.array([-30000000,-30000000,-30000000,-30000000]), 
                                                     high=np.array([30000000,30000000,30000000,30000000]), dtype = np.float32) for i in range(self.n)
        })
        self.action_space = gym.spaces.Dict({
            self.syndicates[i].syndicate_id: gym.spaces.Box(0.5, 0.9, dtype = np.float32) for i in range(self.n)})

        super(MultiAgentBasedModel, self).__init__(sim_args = self.sim_args, 
                                                   manager_args = self.manager_args,
                                                   broker_args = self.broker_args, 
                                                   syndicate_args = self.syndicate_args, 
                                                   reinsurancefirm_args = self.reinsurancefirm_args, 
                                                   shareholder_args = self.shareholder_args, 
                                                   risk_args = self.risk_args, 
                                                   brokers = self.brokers, 
                                                   syndicates = self.syndicates, 
                                                   reinsurancefirms = self.reinsurancefirms, 
                                                   shareholders = self.shareholders, 
                                                   catastrophes = self.catastrophes, 
                                                   broker_risks = self.broker_risks,
                                                   fair_market_premium = self.fair_market_premium,
                                                   risk_model_configs = self.risk_model_configs, 
                                                   with_reinsurance = self.with_reinsurance, 
                                                   num_risk_models = self.num_risk_models,
                                                   logger = self.logger,
                                                   dt = 1)

        # Log data
        self.cumulative_bankruptcies = 0
        self.cumulative_market_exits = 0
        self.cumulative_unrecovered_claims = 0.0
        self.cumulative_claims = 0.0
        self.total_cash = 0.0
        self.total_excess_capital = 0.0
        self.total_profitslosses =  0.0
        self.total_contracts = 0.0
        self.uncovered_risks = 0.0
        self.operational_syndicates = 0.0
        self.insurance_models_counter = np.zeros(self.risk_args["num_categories"])
        self.inaccuracy = []

        # Reset the environmnet
        self.reset()

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        
        # Broker risk event daily: TODO: broker generate risk according to poisson distribution
        # Catastrophe event 
        self.catastrophe_events = EventGenerator(self.risk_model_configs).generate_catastrophe_events(self.catastrophes)
        # Attritioal loss event daily
        self.attritional_loss_events = EventGenerator(self.risk_model_configs).generate_attritional_loss_events(self.sim_args, self.broker_risks)
        # Broker risk event daily: TODO: broker generate risk according to poisson distribution
        self.broker_risk_events = EventGenerator(self.risk_model_configs).generate_risk_events(self.sim_args, self.broker_risks)
        # Broker pay premium according to underwritten contracts
        self.broker_premium_events = EventGenerator(self.risk_model_configs).generate_premium_events(self.sim_args)
        # Broker ask for claim if the contract reaches the end time
        self.broker_claim_events = EventGenerator(self.risk_model_configs).generate_claim_events(self.sim_args)
        # Initiate event handler
        self.event_handler = EventHandler(self.sim_args["max_time"], self.catastrophe_events, self.attritional_loss_events, self.broker_risk_events, self.broker_premium_events, self.broker_claim_events)
        # Initiate market manager
        self.mm = MarketManager(self.sim_args["max_time"], self.sim_args, self.manager_args, self.syndicate_args, self.brokers, self.syndicates, self.reinsurancefirms, self.shareholders, self.catastrophes, self.fair_market_premium,
                                self.risk_model_configs, self.with_reinsurance, self.num_risk_models, self.catastrophe_events, self.attritional_loss_events, 
                                self.broker_risk_events, self.broker_premium_events, self.broker_claim_events, self.event_handler)
        self.mm.evolve(self.dt)
        
        # Set per syndicate active status and build status list
        self.syndicate_active_list = []   # Store syndicates currently in the market
        for sy in range(len(self.mm.market.syndicates)):
            if self.mm.market.syndicates[sy].status == True:
                self.syndicate_active_list.append(self.mm.market.syndicates[sy].syndicate_id)

        # Create action map and state list
        info_dict = {}
        for sy in range(len(self.mm.market.syndicates)):
            self.action_map_dict[self.mm.market.syndicates[sy].syndicate_id] = self.action_map_creator(self.mm.market.syndicates[sy], 0, self.broker_risk_events[0])
            self.state_encoder_dict[self.mm.market.syndicates[sy].syndicate_id] = self.state_encoder(self.mm.market.syndicates[sy].syndicate_id)
            info_dict[self.mm.market.syndicates[sy].syndicate_id] = None

        for i in range(len(self.mm.market.brokers)):
            self.mm.market.brokers[i].underwritten_contracts = []
            self.mm.market.brokers[i].not_underwritten_risks = []
            self.mm.market.brokers[i].not_paid_claims = []
        for i in range(len(self.mm.market.syndicates)):
            self.mm.market.syndicates[i].current_hold_contracts = []
            self.mm.market.syndicates[i].current_capital = self.syndicate_args["initial_capital"]
            self.mm.market.syndicates[i].current_capital_category = [self.syndicate_args["initial_capital"]/self.risk_args["num_categories"] for i in range(self.risk_args["num_categories"])]
        # Reset broker and syndicates variables
        for i in range(len(self.mm.market.syndicates)):
            self.mm.market.syndicates[i].reset_pl()
        # Log data
        self.cumulative_bankruptcies = 0
        self.cumulative_market_exits = 0
        self.cumulative_unrecovered_claims = 0.0
        self.cumulative_claims = 0.0

        # Risk model settings
        self.inaccuracy = []
        self.insurance_models_counter = np.zeros(self.risk_args["num_categories"])
        for i in range(len(self.risk_model_configs)):
            self.inaccuracy.append(self.risk_model_configs[i]["inaccuracy_by_categ"])
        for i in range(len(self.mm.market.syndicates)):
            for j in range(len(self.inaccuracy)):
                if self.mm.market.syndicates[i].riskmodel.inaccuracy == self.inaccuracy[j]:
                        self.insurance_models_counter[j] += 1
        
        # Initiate time step
        self.timestep = -1
        self.step_track = 0

        return self.state_encoder_dict, info_dict
    
    def insurance_entry_index(self):
        return self.insurance_models_counter[0:self.risk_args["num_riskmodels"]].argmin()
    
    def adjust_market_premium(self, capital, market_premium):
        """Adjust_market_premium Method.
               Accepts arguments
                   capital: Type float. The total capital (cash) available in the insurance market (insurance only).
               No return value.
           This method adjusts the premium charged by insurance firms for the risks covered. The premium reduces linearly
           with the capital available in the insurance market and viceversa. The premium reduces until it reaches a minimum
           below which no insurer is willing to reduce further the price. """
        underwriting_premium = market_premium * self.syndicate_args["upper_premium_limit"] - (self.syndicate_args["premium_sensitivity"] 
                                                   * capital / (self.syndicate_args["initial_capital"] 
                                                   * self.risk_model_configs[0]["damage_distribution"].mean() * self.broker_args["num_brokers"] * self.broker_args["lambda_risks_daily"]))
        if underwriting_premium < market_premium * self.syndicate_args["lower_premium_limit"]:
            underwriting_premium = market_premium * self.syndicate_args["lower_premium_limit"]
        return underwriting_premium

    def get_mean(self,x):
        return sum(x) / len(x)
    
    def get_mean_std(self, x):
        m = self.get_mean(x)
        variance = sum((val - m) ** 2 for val in x)
        return m, np.sqrt(variance / len(x))

    def balanced_portfolio(self, syndicate_id, risk, cash_left_by_categ, var_per_risk): #This method decides whether the portfolio is balanced enough to accept a new risk or not. If it is balanced enough return True otherwise False.
                                                                          #This method also returns the cash available per category independently the risk is accepted or not.
        cash_reserved_by_categ = self.mm.market.syndicates[syndicate_id].current_capital - cash_left_by_categ     #Here it is computed the cash already reserved by category

        _, std_pre = self.get_mean_std(cash_reserved_by_categ)

        cash_reserved_by_categ_store = np.copy(cash_reserved_by_categ)

        cash_reserved_by_categ_store[risk.risk_category] += var_per_risk[risk.risk_category] #Here it is computed how the cash reserved by category would change if the new insurance risk was accepted

        mean, std_post = self.get_mean_std(cash_reserved_by_categ_store)     #Here it is computed the mean, std of the cash reserved by category after the new risk of reinrisk is accepted

        total_cash_reserved_by_categ_post = sum(cash_reserved_by_categ_store)

        if (std_post * total_cash_reserved_by_categ_post/self.mm.market.syndicates[syndicate_id].current_capital) <= (self.mm.market.syndicates[syndicate_id].balance_ratio * mean) or std_post < std_pre:      #The new risk is accepted is the standard deviation is reduced or the cash reserved by category is very well balanced. (std_post) <= (self.balance_ratio * mean)
            for i in range(len(cash_left_by_categ)):                                                                           #The balance condition is not taken into account if the cash reserve is far away from the limit. (total_cash_employed_by_categ_post/self.cash <<< 1)
                cash_left_by_categ[i] = self.mm.market.syndicates[syndicate_id].current_capital - cash_reserved_by_categ_store[i]

            return True, cash_left_by_categ
        else:
            for i in range(len(cash_left_by_categ)):
                cash_left_by_categ[i] = self.mm.market.syndicates[syndicate_id].current_capital - cash_reserved_by_categ[i]

            return False, cash_left_by_categ


    def process_newrisks_insurer(self, new_risks, syndicate_id, acceptable_by_category, var_per_risk_per_categ, cash_left_by_categ, time): #This method processes one by one the risks contained in risks_per_categ in order to decide whether they should be underwritten or not
        accept = []
        for i in range(len(new_risks)):
            for categ_id in range(len(acceptable_by_category)):    #Here we take only one risk per category at a time to achieve risk[C1], risk[C2], risk[C3], risk[C4], risk[C1], risk[C2], ... if possible.
                if (acceptable_by_category[categ_id] > 0) and (int(new_risks[i].risk_category) == categ_id):
                    risk_to_insure = new_risks[i]
                    [condition, cash_left_by_categ] = self.balanced_portfolio(int(syndicate_id), risk_to_insure, cash_left_by_categ, var_per_risk_per_categ)   
                    if condition:
                        accept.append(True)
                        acceptable_by_category[categ_id] -= 1  # TODO: allow different values per risk (i.e. sum over value (and reinsurance_share) or exposure instead of counting)
                    else:
                        accept.append(False)
                elif (acceptable_by_category[categ_id] <= 0) and (int(new_risks[i].risk_category) == categ_id):
                    accept.append(False)
        return accept # This list store the accept decision for each risk by this syndicate at this time, its size varied 
    
    def get_actions(self, time):
        new_risks = []
        for risk in range(len(self.broker_risk_events)):
            if self.broker_risk_events[risk].risk_start_time == time:
                new_risks.append(self.broker_risk_events[risk])
        action_dict = [{} for x in range(len(new_risks))]
        min_premium = [0 for x in range(len(new_risks))]
        for i in range(len(self.mm.market.syndicates)):
            expected_profits, acceptable_by_category, cash_left_by_categ, var_per_risk_per_categ, self.excess_capital  = self.mm.market.syndicates[i].riskmodel.evaluate(self.mm.market.syndicates[i].current_hold_contracts, self.mm.market.syndicates[i].current_capital)
            accept = self.process_newrisks_insurer(new_risks, i, acceptable_by_category, var_per_risk_per_categ, cash_left_by_categ, time)
            for num in range(len(new_risks)):
                market_premium = self.mm.market.syndicates[i].offer_premium(new_risks[num])
                sum_capital = sum([self.mm.market.syndicates[k].current_capital for k in range(len(self.mm.market.syndicates))]) 
                market_premium = self.adjust_market_premium(capital=sum_capital, market_premium=market_premium)
                if accept[num]:
                    action_dict[num].update({self.mm.market.syndicates[i].syndicate_id: market_premium})
                    if min_premium[num] == 0 or market_premium < min_premium[num]:
                        min_premium[num] = market_premium
                else:
                    action_dict[num].update({self.mm.market.syndicates[i].syndicate_id: 0})
                    if min_premium[num] == 0:
                        min_premium[num] = market_premium
        
        premium_sum = 0
        k = 0
        for i in range(len(min_premium)):
            if min_premium[i] != 0:
                premium_sum += min_premium[i]
                k += 1
        self.market_premium = premium_sum / k
                
        return action_dict
        
    def step(self, action_dict, new_syndicate):

        obs_dict, reward_dict, terminated_dict, info_dict = {}, {}, {}, {}
        flag_dict = {}

        # Update environemnt after actions
        new_risk = []
        for risk in range(len(self.broker_risk_events)):
            if self.broker_risk_events[risk].risk_start_time == (self.timestep+1):
                new_risk.append(self.broker_risk_events[risk])
        parsed_actions = [[] for x in range(len(new_risk))]  
        for l in range(len(new_risk)):
            for syndicate_id, action in action_dict[l].items():
                # update action map
                self.action_map = self.action_map_creator(self.mm.market.syndicates[int(syndicate_id)], action, new_risk[l]) 
                parsed_ac2add = self.action_map
                parsed_actions[l].append(parsed_ac2add)
        
        self.send_action2env(parsed_actions)

        # Evolve the market
        self.mm.evolve(self.dt)
        
        self.timestep += 1

        # Compute rewards and get next observation
        for l in range(len(new_risk)):
            for syndicate_id, action in action_dict[l].items():
                reward_dict[syndicate_id] = self.compute_reward(action, syndicate_id)
                obs_dict[syndicate_id]= self.state_encoder(syndicate_id)
                info_dict[syndicate_id] = {}
                flag_dict[syndicate_id] = False
                terminated_dict[syndicate_id] = self.check_termination(syndicate_id)
                if terminated_dict[syndicate_id]:
                    self.dones.add(syndicate_id)
        # Update plot 
        self.draw2file(self.mm.market)

        # All done termination check
        #all_terminated = True
        #for _, syndicate_terminated in terminated_dict.items():
            #if syndicate_terminated is False:
                #all_terminated = False
                #break

        if self.timestep >= self.sim_args["max_time"]-1:
            all_terminated = True
        else:
            all_terminated = False
        
        terminated_dict["__all__"] = all_terminated
        flag_dict["__all__"] = all_terminated

        if new_syndicate != None:
            self.mm.market.syndicates.append(new_syndicate)

        return obs_dict, reward_dict, terminated_dict, flag_dict, info_dict

    def check_termination(self, syndicate_id):

        # The simulation is done when syndicates exit or bankrupt or reach the maximum time step
        if self.timestep >= self.sim_args["max_time"]-1:
            terminated = True
        else:
            terminated = False

        return terminated

    def compute_reward(self, action, syndicate_id):

        market = self.mm.market
        # calculate reward function
        r = [0.0] * 4

        # For each insurable risk being accepted +1 or refused -1
        if(self.timestep <= self.sim_args["max_time"]):
            for broker_id in range(len(market.brokers)):
                for risk in range(len(market.brokers[broker_id].risks)):
                    for contract in range(len(market.brokers[broker_id].underwritten_contracts)):
                        if market.brokers[broker_id].risks[risk]["risk_id"] == market.brokers[broker_id].underwritten_contracts[contract]["risk_id"]:
                            r[0] += 1
                        else:
                            r[0] -= 1

        # For each claim being paied +1 or refused -1
        if(self.timestep <= self.sim_args["max_time"]):
            for claim in range(len(market.syndicates[int(syndicate_id)].paid_claim)):
                if market.syndicate[syndicate_id].paid_claim[claim]["status"] == True:
                    r[1] += 1
                else:
                    r[1] -= 1

        # Profit and Bankruptcy       
        if(self.timestep <= self.sim_args["max_time"]):
            initial_capital = market.syndicates[int(syndicate_id)].initial_capital
            current_capital = market.syndicates[int(syndicate_id)].current_capital
            r[2] += current_capital - initial_capital
            if (current_capital - initial_capital) < 0:
                r[3] -= 10000

        # Sum reward
        reward = r[0] + r[1] + r[2] + r[3]

        return reward     

    def send_action2env(self, parsed_actions):               
            
        # Apply action
        if len(parsed_actions) > 0:
            self.mm.receive_actions(actions=parsed_actions) 
    
    def state_encoder(self, syndicate_id):
        
        ### Observation Space:             
        obs = []
        #for risk in range(len(self.broker_risk_events)):
            #if self.broker_risk_events[risk].risk_start_time == self.timestep+1:
                # Catastrophe risk category and risk value
                #obs.append(self.broker_risk_events[risk].risk_category)
                #obs.append(self.broker_risk_events[risk].risk_value)
                #obs.append(self.broker_risk_events[risk].risk_factor)
                #break   # Just for the game version, if AI considered, it needs to fix the size of the obs
        
        # Syndicates status current capital in 
        market = self.mm.market
        for num in range(len(market.syndicates[int(syndicate_id)].current_capital_category)):
            obs.append(market.syndicates[int(syndicate_id)].current_capital_category[num])
            
        return obs

    def action_map_creator(self, syndicate, premium, new_risk):

        action_map = Action(syndicate.syndicate_id, premium, new_risk.risk_id, new_risk.broker_id)
       
        return action_map
    
    def save_data(self):
        """Method to collect statistics about the current state of the simulation. Will pass these to the 
           Logger object (self.logger) to be recorded."""
        # Collect data
        self.total_cash = sum([self.mm.market.syndicates[i].current_capital for i in range(len(self.mm.market.syndicates))])
        self.total_excess_capital = sum([self.mm.market.syndicates[i].excess_capital for i in range(len(self.mm.market.syndicates))])
        self.total_profitslosses =  sum([self.mm.market.syndicates[i].profits_losses for i in range(len(self.mm.market.syndicates))])
        self.total_contracts = sum([len(self.mm.market.brokers[i].underwritten_contracts) for i in range(len(self.mm.market.brokers))])
        self.uncovered_risks = sum([len(self.mm.market.brokers[i].not_underwritten_risks) for i in range(len(self.mm.market.brokers))])
        self.operational_syndicates = sum([self.mm.market.syndicates[i].status for i in range(len(self.mm.market.syndicates))])
        #operational_catbonds = sum([catbond.operational for catbond in self.catbonds])
        
        # Collect agent-level data
        syndicates_data = [(self.mm.market.syndicates[i].current_capital, 
                            self.mm.market.syndicates[i].syndicate_id, 
                            self.mm.market.syndicates[i].status) for i in range(len(self.mm.market.syndicates))]
        
        # Update cumulative information
        for i in range(len(self.mm.market.syndicates)):
            if self.mm.market.syndicates[i].status == False:
                self.cumulative_bankruptcies += 1
            if self.mm.market.syndicates[i].current_capital < self.mm.market.syndicates[i].capital_permanency_limit:         #If their level of cash is so low that they cannot underwrite anything they also leave the market.
                self.cumulative_market_exits += 1  # TODO: update the syndicates list becuase of market exit
            for j in range(len(self.mm.market.syndicates[i].current_hold_contracts)):
                if self.mm.market.syndicates[i].current_hold_contracts[j]["pay"] == False:
                    self.cumulative_unrecovered_claims += self.mm.market.syndicates[i].current_hold_contracts[j]["risk_value"]
                elif self.mm.market.syndicates[i].current_hold_contracts[j]["pay"] == True:
                    self.cumulative_claims += self.mm.market.syndicates[i].current_hold_contracts[j]["risk_value"]
        
        # Prepare dict
        current_log = {}
        current_log['total_cash'] = self.total_cash
        current_log['syndicateA_cash'] = self.mm.market.syndicates[0].current_capital
        current_log['syndicateB_cash'] = self.mm.market.syndicates[1].current_capital
        current_log['syndicateC_cash'] = self.mm.market.syndicates[2].current_capital
        current_log['syndicateD_cash'] = self.mm.market.syndicates[3].current_capital
        current_log['syndicateE_cash'] = self.mm.market.syndicates[4].current_capital
        current_log['syndicateF_cash'] = self.mm.market.syndicates[5].current_capital
        current_log['total_excess_capital'] = self.total_excess_capital
        current_log['total_profits_losses'] = self.total_profitslosses
        current_log['total_contracts'] = self.total_contracts
        current_log['syndicateA_contracts'] = len(self.mm.market.syndicates[0].current_hold_contracts)
        current_log['syndicateB_contracts'] = len(self.mm.market.syndicates[1].current_hold_contracts)
        current_log['syndicateC_contracts'] = len(self.mm.market.syndicates[2].current_hold_contracts)
        current_log['syndicateD_contracts'] = len(self.mm.market.syndicates[3].current_hold_contracts)
        current_log['syndicateE_contracts'] = len(self.mm.market.syndicates[4].current_hold_contracts)
        current_log['syndicateF_contracts'] = len(self.mm.market.syndicates[5].current_hold_contracts)
        current_log['uncovered_risks'] = self.uncovered_risks
        current_log['total_operational'] = self.operational_syndicates
        current_log['market_premium'] = self.market_premium  # Oxford Picing has a fair premium and adjust TODO:  
        current_log['cumulative_bankruptcies'] = self.cumulative_bankruptcies
        current_log['cumulative_market_exits'] = self.cumulative_market_exits
        current_log['cumulative_uncovered_claims'] = self.cumulative_unrecovered_claims
        current_log['cumulative_claims'] = self.cumulative_claims    #Log the cumulative claims received so far. 
        
        # Add agent-level data to dict
        current_log['insurance_firms_cash'] = syndicates_data
        
        current_log['individual_contracts'] = []
        individual_contracts_no = [len(self.mm.market.syndicates[i].current_hold_contracts) for i in range(len(self.mm.market.syndicates))]
        for i in range(len(individual_contracts_no)):
            current_log['individual_contracts'].append(individual_contracts_no[i])

        # Call to Logger object
        self.logger.record_data(current_log)

    def obtain_log(self, requested_logs=None):
        #This function allows to return in a list all the data generated by the model. There is no other way to transfer it back from the cloud.
        return self.logger.obtain_log(requested_logs)