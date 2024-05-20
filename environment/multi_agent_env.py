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
                 risk_model_configs, with_reinsurance, num_risk_models, dt = 1):
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
        self.fair_market_premium =fair_market_premium
        self.initial_catastrophes = catastrophes
        self.risk_model_configs = risk_model_configs
        self.with_reinsurance = with_reinsurance
        self.num_risk_models = num_risk_models
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
            self.syndicates[i].syndicate_id: gym.spaces.Box(low=np.array([-10000000,-10000000,-10000000,-30000000,-30000000,-30000000,-30000000]), 
                                                     high=np.array([10000000,10000000,10000000,30000000,30000000,30000000,30000000]), dtype = np.float32) for i in range(self.n)
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
                                                   dt = 1)
        
        self.riskmodel = RiskModel(self.risk_model_configs[0]["damage_distribution"], 
                                   self.risk_model_configs[0]["expire_immediately"],
                                   self.risk_model_configs[0]["catastrophe_separation_distribution"],
                                   self.risk_model_configs[0]["norm_premium"],
                                   self.risk_model_configs[0]["num_categories"],
                                   self.risk_model_configs[0]["risk_value_mean"],
                                   self.risk_model_configs[0]["risk_factor_mean"],
                                   self.risk_model_configs[0]["norm_profit_markup"],
                                   self.risk_model_configs[0]["margin_of_safety"],
                                   self.risk_model_configs[0]["var_tail_prob"],
                                   self.risk_model_configs[0]["inaccuracy_by_categ"])

        # Log data
        self.cumulative_bankruptcies = 0
        self.cumulative_market_exits = 0
        self.cumulative_unrecovered_claims = 0.0
        self.cumulative_claims = 0.0
        self.total_cash = 0.0
        self.total_excess_capital = 0.0
        self.total_profitslosses =  0.0
        self.total_contracts = 0.0
        self.operational_syndicates = 0.0
        self.logger = logger.Logger(self.num_risk_models, self.catastrophes, self.brokers, self.syndicates)

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
        self.broker_risk_events = EventGenerator(self.risk_model_configs).generate_risk_events(self.sim_args, self.brokers, self.broker_risks)
        # Broker pay premium according to underwritten contracts
        self.broker_premium_events = EventGenerator(self.risk_model_configs).generate_premium_events(self.sim_args)
        # Broker ask for claim if the contract reaches the end time
        self.broker_claim_events = EventGenerator(self.risk_model_configs).generate_claim_events(self.sim_args)
        # Initiate event handler
        self.event_handler = EventHandler(self.maxstep, self.catastrophe_events, self.attritional_loss_events, self.broker_risk_events, self.broker_premium_events, self.broker_claim_events)
        # Initiate market manager
        self.mm = MarketManager(self.maxstep, self.manager_args, self.brokers, self.syndicates, self.reinsurancefirms, self.shareholders, self.catastrophes, self.fair_market_premium,
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
            self.action_map_dict[self.mm.market.syndicates[sy].syndicate_id] = self.action_map_creator(self.mm.market.syndicates[sy], 0)
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
        self.logger = logger.Logger(self.num_risk_models, self.catastrophes, self.mm.market.brokers, self.mm.market.syndicates)

        # Initiate time step
        self.timestep = -1
        self.step_track = 0

        return self.state_encoder_dict, info_dict
    
    def adjust_market_premium(self, capital):
        """Adjust_market_premium Method.
               Accepts arguments
                   capital: Type float. The total capital (cash) available in the insurance market (insurance only).
               No return value.
           This method adjusts the premium charged by insurance firms for the risks covered. The premium reduces linearly
           with the capital available in the insurance market and viceversa. The premium reduces until it reaches a minimum
           below which no insurer is willing to reduce further the price. """
        self.market_premium = self.fair_market_premium * (self.syndicate_args["upper_premium_limit"] 
                                                   - self.syndicate_args["premium_sensitivity"] 
                                                   * capital / (self.syndicate_args["initial_capital"] 
                                                   * self.risk_model_configs[0]["damage_distribution"].mean() * self.risk_args["num_risks"]))
        if self.market_premium < self.fair_market_premium * self.syndicate_args["lower_premium_limit"]:
            self.market_premium = self.fair_market_premium * self.syndicate_args["lower_premium_limit"]

    def get_mean(self,x):
        return sum(x) / len(x)
    
    def get_mean_std(self, x):
        m = self.get_mean(x)
        variance = sum((val - m) ** 2 for val in x)
        return m, np.sqrt(variance / len(x))

    def balanced_portfolio(self, risk_categ, risk_value, risk_factor, syndicate_id, cash_left_by_categ, var_per_risk): #This method decides whether the portfolio is balanced enough to accept a new risk or not. If it is balanced enough return True otherwise False.
                                                                          #This method also returns the cash available per category independently the risk is accepted or not.
        cash_reserved_by_categ = self.mm.market.syndicates[syndicate_id].current_capital - cash_left_by_categ     #Here it is computed the cash already reserved by category

        _, std_pre = self.get_mean_std(cash_reserved_by_categ)

        cash_reserved_by_categ_store = np.copy(cash_reserved_by_categ)
        
        percentage_value_at_risk = self.riskmodel.getPPF(risk_categ, tailSize=self.riskmodel.var_tail_prob)
        expected_damage = percentage_value_at_risk * risk_value * risk_factor * self.riskmodel.inaccuracy[risk_categ]
        expected_claim = min(expected_damage, risk_value * 1.0) - risk_value * 0.3

        # record liquidity requirement and apply margin of safety for liquidity requirement

        cash_reserved_by_categ_store[risk_categ] += expected_claim * self.riskmodel.margin_of_safety  #Here it is computed how the cash reserved by category would change if the new reinsurance risk was accepted

        mean, std_post = self.get_mean_std(cash_reserved_by_categ_store)     #Here it is computed the mean, std of the cash reserved by category after the new risk of reinrisk is accepted

        total_cash_reserved_by_categ_post = sum(cash_reserved_by_categ_store)

        if (std_post * total_cash_reserved_by_categ_post/self.mm.market.syndicates[syndicate_id].current_capital) <= (self.syndicate_args["insurers_balance_ratio"] * mean) or std_post < std_pre:      
            for i in range(len(cash_left_by_categ)):                                                                           #The balance condition is not taken into account if the cash reserve is far away from the limit. (total_cash_employed_by_categ_post/self.cash <<< 1)
                cash_left_by_categ[i] = self.mm.market.syndicates[syndicate_id].current_capital - cash_reserved_by_categ_store[i]

            return True, cash_left_by_categ
        else:
            for i in range(len(cash_left_by_categ)):
                cash_left_by_categ[i] = self.mm.market.syndicates[syndicate_id].current_capital - cash_reserved_by_categ[i]

            return False, cash_left_by_categ

    def process_newrisks_insurer(self, risk_categ, risk_value, risk_factor, syndicate_id, acceptable_by_category, var_per_risk_per_categ, cash_left_by_categ, time): 
        contract_runtime_dist = scipy.stats.randint(self.sim_args["mean_contract_runtime"] - self.sim_args["contract_runtime_halfspread"], self.sim_args["mean_contract_runtime"] + self.sim_args["contract_runtime_halfspread"] + 1)
        _cached_rvs = contract_runtime_dist.rvs()
        if acceptable_by_category[risk_categ] > 0:
            [condition, cash_left_by_categ] = self.balanced_portfolio(risk_categ, risk_value, risk_factor, syndicate_id, cash_left_by_categ, None)   #Here it is check whether the portfolio is balanced or not if the reinrisk (risk_to_insure) is underwritten. Return True if it is balanced. False otherwise.
            if not condition:
                [condition, cash_left_by_categ] = self.balanced_portfolio(risk_categ, risk_value, risk_factor, syndicate_id, cash_left_by_categ, var_per_risk_per_categ) #Here it is check whether the portfolio is balanced or not if the risk (risk_to_insure) is underwritten. Return True if it is balanced. False otherwise.
                if condition:
                    accept = True
                    self.cash_left_by_categ = cash_left_by_categ
                    acceptable_by_category[risk_categ] -= 1  # TODO: allow different values per risk (i.e. sum over value (and reinsurance_share) or exposure instead of counting)
                    return accept
                else:
                    accept = False
                    return accept
    
    def get_actions(self, obs_dict, time):
        action_dict = {}
        for i in range(len(self.mm.market.syndicates)):
            if len(self.mm.market.syndicates[i].current_hold_contracts) == 0: 
                # Syndicates compete for the ledership, they will all cover 0.5
                sum_capital = sum([self.mm.market.syndicates[i].current_capital for i in range(len(self.mm.market.syndicates))]) 
                self.adjust_market_premium(capital=sum_capital)
                action_dict.update({self.mm.market.syndicates[i].syndicate_id: self.market_premium})
            else: 
                expected_profits, acceptable_by_category, cash_left_by_categ, var_per_risk_per_categ, self.excess_capital  = self.riskmodel.evaluate(self.mm.market.syndicates[i].current_hold_contracts, self.mm.market.syndicates[i].current_capital)
                risk_categ = obs_dict[self.mm.market.syndicates[i].syndicate_id][0]
                risk_value = obs_dict[self.mm.market.syndicates[i].syndicate_id][1]
                risk_factor = obs_dict[self.mm.market.syndicates[i].syndicate_id][2]
                accept = self.process_newrisks_insurer(risk_categ, risk_value, risk_factor, i, acceptable_by_category, var_per_risk_per_categ, cash_left_by_categ, time)
                if accept:
                    sum_capital = sum([self.mm.market.syndicates[i].current_capital for i in range(len(self.mm.market.syndicates))]) 
                    self.adjust_market_premium(capital=sum_capital)
                    action_dict.update({self.mm.market.syndicates[i].syndicate_id: self.market_premium})
                else:
                    action_dict.update({self.mm.market.syndicates[i].syndicate_id: 0})

        #{'0': array([0.9], dtype=float32), '1': array([0.67964387], dtype=float32), '2': array([0.77142656], dtype=float32)}
        return action_dict
        
    def step(self, action_dict):

        obs_dict, reward_dict, terminated_dict, info_dict = {}, {}, {}, {}
        flag_dict = {}

        # Update environemnt after actions
        parsed_actions = []        
        for syndicate_id, action in action_dict.items():
            # update action map
            self.action_map = self.action_map_creator(self.mm.market.syndicates[int(syndicate_id)],action)
            parsed_ac2add = self.action_map
            parsed_actions.append(parsed_ac2add)
        
        self.send_action2env(parsed_actions)

        # Evolve the market
        self.mm.evolve(self.dt)
        
        self.timestep += 1

        # Compute rewards and get next observation
        for syndicate_id, action in action_dict.items():
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
        all_terminated = True
        for _, syndicate_terminated in terminated_dict.items():
            if syndicate_terminated is False:
                all_terminated = False
                break
        
        terminated_dict["__all__"] = all_terminated
        flag_dict["__all__"] = all_terminated

        return obs_dict, reward_dict, terminated_dict, flag_dict, info_dict

    def check_termination(self, syndicate_id):

        # Update per syndicate status, True-active in market, False-exit market becuase of no contract or bankruptcy
        market = self.mm.market
        sy = market.syndicates[int(syndicate_id)] 

        # The simulation is done when syndicates exit or bankrupt or reach the maximum time step
        if self.timestep >= self.maxstep-1:
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
            for claim in range(len(market.syndicates[int(syndicate_id)].paid_claim)):
                if market.syndicate[syndicate_id].paid_claim[claim]["status"] == True:
                    r[1] += 1
                else:
                    r[1] -= 1

        # Profit and Bankruptcy       
        if(self.timestep <= self.maxstep):
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
        for risk in range(len(self.broker_risk_events)):
            if self.broker_risk_events[risk].risk_start_time == self.timestep+1:
                # Catastrophe risk category and risk value
                obs.append(self.broker_risk_events[risk].risk_category)
                obs.append(self.broker_risk_events[risk].risk_value)
                obs.append(self.broker_risk_events[risk].risk_factor)
        
        # Syndicates status current capital in 
        market = self.mm.market
        for num in range(len(market.syndicates[int(syndicate_id)].current_capital_category)):
            obs.append(market.syndicates[int(syndicate_id)].current_capital_category[num])
            
        return obs

    def action_map_creator(self, syndicate, premium):

        action_map = None
        for risk in range(len(self.broker_risk_events)):
            if self.broker_risk_events[risk].risk_start_time == self.timestep+1:
                action_map = Action(syndicate.syndicate_id, premium, self.broker_risk_events[risk].risk_id, self.broker_risk_events[risk].broker_id)
       
        return action_map
    
    def save_data(self):
        """Method to collect statistics about the current state of the simulation. Will pass these to the 
           Logger object (self.logger) to be recorded."""
        # Collect data
        self.total_cash = sum([self.mm.market.syndicates[i].current_capital for i in range(len(self.mm.market.syndicates))])
        self.total_excess_capital = sum([self.mm.market.syndicates[i].excess_capital for i in range(len(self.mm.market.syndicates))])
        self.total_profitslosses =  sum([self.mm.market.syndicates[i].profits_losses for i in range(len(self.mm.market.syndicates))])
        self.total_contracts = sum([len(self.mm.market.syndicates[i].current_hold_contracts) for i in range(len(self.mm.market.syndicates))])
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
        current_log['total_excess_capital'] = self.total_excess_capital
        current_log['total_profits_losses'] = self.total_profitslosses
        current_log['total_contracts'] = self.total_contracts
        current_log['total_operational'] = self.operational_syndicates
        #current_log['total_catbondsoperational'] = catbondsoperational_no
        current_log['market_premium'] = self.market_premium  # Oxford Picing has a fair premium and adjust TODO:  
        current_log['cumulative_bankruptcies'] = self.cumulative_bankruptcies
        current_log['cumulative_market_exits'] = self.cumulative_market_exits
        current_log['cumulative_unrecovered_claims'] = self.cumulative_unrecovered_claims
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

        