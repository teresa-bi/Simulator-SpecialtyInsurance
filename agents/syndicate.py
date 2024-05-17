"""
Contains all the capabilities of Syndicates
"""

import numpy as np
import scipy.stats
import copy
from environment.risk_model import RiskModel

class Syndicate:
    """
    Instance for syndicate
    """
    def __init__(self, syndicate_id, syndicate_args, num_risk_models, sim_args, risk_model_configs):
        self.syndicate_id = syndicate_id
        self.initial_capital = syndicate_args['initial_capital']
        self.current_capital = syndicate_args['initial_capital']
        self.current_capital_category = [syndicate_args['initial_capital'] / risk_model_configs[0]["num_categories"] for x in range(risk_model_configs[0]["num_categories"])]
        self.acceptance_threshold = syndicate_args['initial_acceptance_threshold']
        self.acceptance_threshold_friction = syndicate_args['initial_acceptance_threshold_friction']
        self.reinsurance_limit = syndicate_args["reinsurance_limit"]
        self.non_proportional_reinsurance_level = syndicate_args["default_non_proportional_reinsurance_deductible"]
        self.capacity_target_decrement_threshold = syndicate_args["capacity_target_decrement_threshold"]
        self.capacity_target_increment_threshold = syndicate_args["capacity_target_increment_threshold"]
        self.capital_permanency_limit = syndicate_args["cash_permanency_limit"]
        self.capacity_target_decrement_factor = syndicate_args["capacity_target_decrement_factor"]
        self.capacity_target_increment_factor = syndicate_args["capacity_target_increment_factor"]
        self.interest_rate = syndicate_args['interest_rate']
        self.dividend_share_of_profits = syndicate_args["dividend_share_of_profits"]

        self.contract_runtime_dist = scipy.stats.randint(sim_args["mean_contract_runtime"] - sim_args["contract_runtime_halfspread"], sim_args["mean_contract_runtime"] + sim_args["contract_runtime_halfspread"]+1)
        self.default_contract_payment_period = sim_args["default_contract_payment_period"]
        self.simulation_reinsurance_type = sim_args["simulation_reinsurance_type"]
        self.capacity_target = self.current_capital * 0.9
        self.capacity_target_decrement_threshold = self.capacity_target_decrement_threshold
        self.capacity_target_increment_threshold = self.capacity_target_increment_threshold
        self.capacity_target_decrement_factor = self.capacity_target_decrement_factor
        self.capacity_target_increment_factor = self.capacity_target_increment_factor

        self.riskmodel_config = risk_model_configs[int(syndicate_id) % len(risk_model_configs)]
        self.premium = self.riskmodel_config["norm_premium"]
        self.profit_target = self.riskmodel_config["norm_profit_markup"]
        self.excess_capital = self.current_capital
        self.num_risk_categories = self.riskmodel_config["num_categories"]
        
        self.per_period_dividend = 0
        self.capital_last_periods = list(np.zeros(4,dtype=int)*self.current_capital)

        self.premium_internal_weight = syndicate_args['actuarial_pricing_internal_weight']
        self.play_leader_in_contracts = []  # Include risk_id, broker_id, line_size
        self.lead_line_size = syndicate_args['lead_line_size']
        self.play_follower_in_contracts = []  # Include risk_id, broker_id, line_size
        self.follow_line_size = syndicate_args['follow_line_size']
        self.loss_experiency_weight = syndicate_args['loss_experiency_weight']
        self.volatility_weight = syndicate_args['volatility_weight']
        self.underwriter_markup_recency_weight = syndicate_args['underwriter_markup_recency_weight']
        self.upper_premium_limit = syndicate_args['upper_premium_limit']
        self.lower_premium_limit = syndicate_args['lower_premium_limit']
        self.premium_reserve_ratio = syndicate_args['premium_reserve_ratio']
        self.minimum_capital_reserve_ratio = syndicate_args['minimum_capital_reserve_ratio']
        self.maximum_scaling_factor = syndicate_args['maximum_scaling_factor']
        self.market_entry_probability = syndicate_args['market_entry_probability']
        self.exit_capital_threshold = syndicate_args['exit_capital_threshold']
        self.exit_time_limit = syndicate_args['exit_time_limit']
        self.premium_sensitivity = syndicate_args['premium_sensitivity']

        self.status = True   # status True means active, status False means exit (no contract or bankruptcy, at the begining the status is 0 because no syndicate joining the market

        self.current_hold_contracts = []
        # Include paid status, True or False
        self.paid_claim = []
        # Include unpaid claims and loss
        self.unpaid_claim = []
        margin_of_safety_correction = (self.riskmodel_config["margin_of_safety"] + (num_risk_models - 1) * sim_args["margin_increase"])

        self.riskmodel = RiskModel(damage_distribution = self.riskmodel_config["damage_distribution"],
                                expire_immediately = self.riskmodel_config["expire_immediately"],
                                catastrophe_separation_distribution = self.riskmodel_config["catastrophe_separation_distribution"],
                                norm_premium = self.riskmodel_config["norm_premium"],
                                category_number = self.riskmodel_config["num_categories"],
                                init_average_exposure = self.riskmodel_config["risk_value_mean"],
                                init_average_risk_factor = self.riskmodel_config["risk_factor_mean"],
                                init_profit_estimate = self.riskmodel_config["norm_profit_markup"],
                                margin_of_safety = margin_of_safety_correction,
                                var_tail_prob = self.riskmodel_config["var_tail_prob"],
                                inaccuracy = self.riskmodel_config["inaccuracy_by_categ"])

        self.category_reinsurance = [None for i in range(self.num_risk_categories)]

        if self.simulation_reinsurance_type == 'non-proportional':
            if self.non_proportional_reinsurance_level is not None:
                self.np_reinsurance_deductible_fraction = self.non_proportional_reinsurance_level
            else:
                self.np_reinsurance_deductible_fraction = syndicate_args["default_non-proportional_reinsurance_deductible"]
            self.np_reinsurance_excess_fraction = syndicate_args["default_non-proportional_reinsurance_excess"]
            self.np_reinsurance_premium_share = syndicate_args["default_non-proportional_reinsurance_premium_share"]

        self.obligations = []
        self.underwritten_contracts = []
        self.profits_losses = 0

        # Set up risk value estimate variables
        self.var_counter = 0           # Sum over risk model inaccuracies for all contracts
        self.var_counter_per_risk = 0    # Average risk model inaccuracy across contracts
        self.var_sum = 0           # Sum over initial vaR for all contracts
        self.counter_category = np.zeros(self.num_risk_categories)     # var_counter disaggregated by category
        self.var_category = np.zeros(self.num_risk_categories)
        self.naccep = []
        self.risk_kept = []
        self.reinrisks_kept = []
        self.balance_ratio = syndicate_args['insurers_balance_ratio']
        self.recursion_limit = syndicate_args["insurers_recursion_limit"]
        self.capital_left_by_categ = [self.current_capital for i in range(self.num_risk_categories)]
        self.market_permanency_counter = 0
        self.received_risk_list = []

    def bankrupt(self):

        pass

    def received_risk(self, risk_id, broker_id, start_time):
        """
        After broker send risk to the market, all the active syndicate receive risks and update their received_risk list
        """
        self.received_risk_list.append({"risk_id": risk_id,
                                  "broker_id": broker_id,
                                  "start_time": start_time})

    def add_leader(self, risks, lead_line_size, lead_syndicate_premium):
        """
        Add new contract to the play_leader_in_contracts list
        """
        self.play_leader_in_contracts.append({"risk_id": risks.get("risk_id"),
                                    "broker_id": risks.get("broker_id"),
                                    "line_size": lead_line_size,
                                    "premium": lead_syndicate_premium})

    def add_follower(self, risks, follow_line_sizes, follow_syndicates_premium):
        """
        Add new contract to the play_follower_in_contracts list
        """
        self.play_follower_in_contracts.append({"risk_id": risks.get("risk_id"),
                                    "broker_id": risks.get("broker_id"),
                                    "line_size": follow_line_sizes,
                                    "premium": follow_syndicates_premium})

    def add_contract(self, risks, broker_id, premium):
        """
        Add new contract to the current_hold_contracts list
        """
        self.current_hold_contracts.append({"risk_id": risks.get("risk_id"),
                                    "broker_id": broker_id,
                                    "risk_start_time": risks.get("risk_start_time"),
                                    "risk_factor": risks.get("risk_factor"),
                                    "risk_category": risks.get("risk_category"),
                                    "risk_value": risks.get("risk_value"),
                                    "syndicate_id": self.syndicate_id,
                                    "premium": premium,
                                    "risk_end_time": risks.get("risk_end_time"),
                                    "pay": None})
        for i in range(len(self.current_capital_category)):
            if i == int(risks.get("risk_category")):
                self.current_capital_category[i] -= risks.get("risk_value")   

    def rereceived_risk(self, risk_id, broker_id, risk_start_time):
        pass

    def pay_claim(self, broker_id, category_id, claim_value):
        """
        Pay claim for ended contract and for contracts affected by catastrophe
        """
        for i in range(len(self.current_hold_contracts)):
            if (int(self.current_hold_contracts[i]["broker_id"]) == int(broker_id)) and (self.current_hold_contracts[i]["risk_category"] == category_id): 
                if self.current_capital >= claim_value:  
                    self.current_hold_contracts[i]["pay"] = True  
                    self.current_capital
                else:
                    self.current_hold_contracts[i]["pay"] = False

    def receive_premium(self, premium, category_id):
        """
        Receive premium from brokers
        """
        self.current_capital_category[int(category_id)] += premium
        self.current_capital += premium

    def estimated_var(self):

        self.counter_category = np.zeros(self.num_risk_categories)
        self.var_category = np.zeros(self.num_risk_categories)

        self.var_counter = 0
        self.var_counter_per_risk = 0
        self.var_sum = 0
        
        if self.operational:

            for contract in self.underwritten_contracts:
                self.counter_category[contract.category] = self.counter_category[contract.category] + 1
                self.var_category[contract.category] = self.var_category[contract.category] + contract.initial_VaR

            for category in range(len(self.counter_category)):
                self.var_counter = self.var_counter + self.counter_category[category] * self.riskmodel.inaccuracy[category]
                self.var_sum = self.var_sum + self.var_category[category]

            if not sum(self.counter_category) == 0:
                self.var_counter_per_risk = self.var_counter / sum(self.counter_category)
            else:
                self.var_counter_per_risk = 0

    def balanced_portfolio(self, risk, capital_left_by_categ, var_per_risk): #This method decides whether the portfolio is balanced enough to accept a new risk or not. If it is balanced enough return True otherwise False.
                                                                          #This method also returns the cash available per category independently the risk is accepted or not.
        capital_reserved_by_categ = self.current_capital - capital_left_by_categ     #Here it is computed the cash already reserved by category

        _, std_pre = get_mean_std(capital_reserved_by_categ)

        capital_reserved_by_categ_store = np.copy(capital_reserved_by_categ)

        if risk.get("insurancetype")=='excess-of-loss':
            percentage_value_at_risk = self.riskmodel.getPPF(categ_id=risk["category"], tailSize=self.riskmodel.var_tail_prob)
            expected_damage = percentage_value_at_risk * risk["value"] * risk["risk_factor"] \
                              * self.riskmodel.inaccuracy[risk["category"]]
            expected_claim = min(expected_damage, risk["value"] * risk["excess_fraction"]) - risk["value"] * risk["deductible_fraction"]

            # record liquidity requirement and apply margin of safety for liquidity requirement

            capital_reserved_by_categ_store[risk["category"]] += expected_claim * self.riskmodel.margin_of_safety  #Here it is computed how the cash reserved by category would change if the new reinsurance risk was accepted

        else:
            capital_reserved_by_categ_store[risk["category"]] += var_per_risk[risk["category"]] #Here it is computed how the cash reserved by category would change if the new insurance risk was accepted

        mean, std_post = get_mean_std(capital_reserved_by_categ_store)     #Here it is computed the mean, std of the cash reserved by category after the new risk of reinrisk is accepted

        total_capital_reserved_by_categ_post = sum(capital_reserved_by_categ_store)

        if (std_post * total_capital_reserved_by_categ_post/self.current_capital) <= (self.balance_ratio * mean) or std_post < std_pre:      #The new risk is accepted is the standard deviation is reduced or the cash reserved by category is very well balanced. (std_post) <= (self.balance_ratio * mean)
            for i in range(len(capital_left_by_categ)):                                                                           #The balance condition is not taken into account if the cash reserve is far away from the limit. (total_capital_employed_by_categ_post/self.current_capital <<< 1)
                capital_left_by_categ[i] = self.current_capital - capital_reserved_by_categ_store[i]

            return True, capital_left_by_categ
        else:
            for i in range(len(capital_left_by_categ)):
                capital_left_by_categ[i] = self.current_capital - capital_reserved_by_categ[i]

            return False, capital_left_by_categ
        
    def process_newrisks_insurer(self, risks_per_categ, number_risks_categ, acceptable_by_category, var_per_risk_per_categ, capital_left_by_categ, time): #This method processes one by one the risks contained in risks_per_categ in order to decide whether they should be underwritten or not.
                                                                                             #It is done in this way to maintain the portfolio as balanced as possible. For that reason we process risk[C1], risk[C2], risk[C3], risk[C4], risk[C1], risk[C2], ... and so forth.
        _cached_rvs = self.contract_runtime_dist.rvs()
        for iter in range(max(number_risks_categ)):
            for categ_id in range(len(acceptable_by_category)):    #Here we take only one risk per category at a time to achieve risk[C1], risk[C2], risk[C3], risk[C4], risk[C1], risk[C2], ... if possible.
                if iter < number_risks_categ[categ_id] and acceptable_by_category[categ_id] > 0 and \
                                risks_per_categ[categ_id][iter] is not None:
                    risk_to_insure = risks_per_categ[categ_id][iter]
                    if risk_to_insure.get("contract") is not None and risk_to_insure[
                        "contract"].expiration > time:  # risk_to_insure["contract"]: # required to rule out contracts that have exploded in the meantime
                        [condition, capital_left_by_categ] = self.balanced_portfolio(risk_to_insure, capital_left_by_categ, None)   #Here it is check whether the portfolio is balanced or not if the reinrisk (risk_to_insure) is underwritten. Return True if it is balanced. False otherwise.
                        if condition:
                            contract = ReinsuranceContract(self, risk_to_insure, time, \
                                                           self.simulation.get_reinsurance_market_premium(),
                                                           risk_to_insure["expiration"] - time, \
                                                           self.default_contract_payment_period, \
                                                           expire_immediately=self.simulation_parameters[
                                                               "expire_immediately"], )
                            self.underwritten_contracts.append(contract)
                            self.capital_left_by_categ = capital_left_by_categ
                            risks_per_categ[categ_id][iter] = None
                            # TODO: move this to insurancecontract (ca. line 14) -> DONE
                            # TODO: do not write into other object's properties, use setter -> DONE
                    else:
                        [condition, capital_left_by_categ] = self.balanced_portfolio(risk_to_insure, capital_left_by_categ,
                                                                                  var_per_risk_per_categ) #Here it is check whether the portfolio is balanced or not if the risk (risk_to_insure) is underwritten. Return True if it is balanced. False otherwise.
                        if condition:
                            contract = InsuranceContract(self, risk_to_insure, time, self.simulation.get_market_premium(), \
                                                         _cached_rvs, \
                                                         self.default_contract_payment_period, \
                                                         expire_immediately=self.simulation_parameters[
                                                             "expire_immediately"], \
                                                         initial_VaR=var_per_risk_per_categ[categ_id])
                            self.underwritten_contracts.append(contract)
                            self.capital_left_by_categ = capital_left_by_categ
                            risks_per_categ[categ_id][iter] = None
                    acceptable_by_category[categ_id] -= 1  # TODO: allow different values per risk (i.e. sum over value (and reinsurance_share) or exposure instead of counting)

        not_accepted_risks = []
        for categ_id in range(len(acceptable_by_category)):
            for risk in risks_per_categ[categ_id]:
                if risk is not None:
                    not_accepted_risks.append(risk)

        return risks_per_categ, not_accepted_risks
    
    def market_permanency(self, time):     #This method determines whether an insurer or reinsurer stays in the market. If it has very few risks underwritten or too much cash left for TOO LONG it eventually leaves the market.
                                                      # If it has very few risks underwritten it cannot balance the portfolio so it makes sense to leave the market.
        if not self.simulation_parameters["market_permanency_off"]:

            capital_left_by_categ = np.asarray(self.capital_left_by_categ)

            avg_capital_left = get_mean(capital_left_by_categ)

            if self.current_capital < self.simulation_parameters["capital_permanency_limit"]:         #If their level of cash is so low that they cannot underwrite anything they also leave the market.
                self.market_exit(time)
            else:
                if self.is_insurer:

                    if len(self.underwritten_contracts) < self.simulation_parameters["insurance_permanency_contracts_limit"] or avg_capital_left / self.current_capital > self.simulation_parameters["insurance_permanency_ratio_limit"]:
                        #Insurers leave the market if they have contracts under the limit or an excess capital over the limit for too long.
                        self.market_permanency_counter += 1
                    else:
                        self.market_permanency_counter = 0                                    #All these limits maybe should be parameters in isleconfig.py

                    if self.market_permanency_counter >= self.simulation_parameters["insurance_permanency_time_constraint"]:    # Here we determine how much is too long.
                        self.market_exit(time)

                if self.is_reinsurer:

                    if len(self.underwritten_contracts) < self.simulation_parameters["reinsurance_permanency_contracts_limit"] or avg_capital_left / self.current_capital > self.simulation_parameters["reinsurance_permanency_ratio_limit"]:
                        #Reinsurers leave the market if they have contracts under the limit or an excess capital over the limit for too long.

                        self.market_permanency_counter += 1                                       #Insurers and reinsurers potentially have different reasons to leave the market. That's why the code is duplicated here.
                    else:
                        self.market_permanency_counter = 0

                    if self.market_permanency_counter >= self.simulation_parameters["reinsurance_permanency_time_constraint"]:  # Here we determine how much is too long.
                        self.market_exit(time)

    def reset_pl(self):
        """Reset_pl Method.
               Accepts no arguments:
               No return value.
           Reset the profits and losses variable of each firm at the beginning of every iteration. It has to be run in insurancesimulation.py at the beginning of the iterate method"""
        self.profits_losses = 0

    def roll_over(self,time):
        """Roll_over Method.
               Accepts arguments
                   time: Type integer. The current time.               No return value.
               No return value.
            This method tries to roll over the insurance and reinsurance contracts expiring in the next iteration. In
            the case of insurance contracts it assumes that it can only retain a fraction of contracts inferior to the
            retention rate. The contracts that cannot be retained are sent back to insurancesimulation.py. The rest are
            kept and evaluated the next iteration. For reinsurancecontracts is exactly the same with the difference that
            there is no need to return the contracts not rolled over to insurancesimulation.py, since reinsurance risks
            are created and destroyed every iteration. The main reason to implemented this method is to avoid a lack of
            coverage that appears, if contracts are allowed to mature and are evaluated again the next iteration."""

        maturing_next = [contract for contract in self.underwritten_contracts if contract.expiration == time + 1]

        if self.is_insurer is True:
            for contract in maturing_next:
                contract.roll_over_flag = 1
                if np.random.uniform(0,1,1) > self.simulation_parameters["insurance_retention"]:
                    self.simulation.return_risks([contract.risk_data])   # TODO: This is not a retention, so the roll_over_flag might be confusing in this case
                else:
                    self.risks_kept.append(contract.risk_data)

        if self.is_reinsurer is True:
            for reincontract in maturing_next:
                if reincontract.property_holder.operational:
                    reincontract.roll_over_flag = 1
                    reinrisk = reincontract.property_holder.create_reinrisk(time, reincontract.category)
                    if np.random.uniform(0,1,1) < self.simulation_parameters["reinsurance_retention"]:
                        if reinrisk is not None:
                            self.reinrisks_kept.append(reinrisk)

    def data(self):
        """
        Create a dictionary with key/value pairs representing the Syndicate data.

        Returns
        ----------
        dict
        """

        return {
            "syndicate_id": self.syndicate_id,
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "premium_internal_weight": self.premium_internal_weight,
            "interest_rate": self.interest_rate,
            "play_leader_in_contracts": self.play_leader_in_contracts,
            "play_follower_in_contracts": self.play_follower_in_contracts,
            "loss_experiency_weight": self.loss_experiency_weight,
            "volatility_weight": self.volatility_weight,
            "underwriter_markup_recency_weight": self.underwriter_markup_recency_weight,
            "upper_premium_limit": self.upper_premium_limit,
            "lower_premium_limit": self.lower_premium_limit,
            "premium_reserve_ratio": self.premium_reserve_ratio,
            "minimum_capital_reserve_ratio": self.minimum_capital_reserve_ratio,
            "maximum_scaling_factor": self.maximum_scaling_factor,
            "market_entry_probability": self.market_entry_probability,
            "exit_capital_threshold": self.exit_capital_threshold,
            "exit_time_limit": self.exit_time_limit,
            "premium_sensitivity": self.premium_sensitivity,
            "acceptance_threshold_friction": self.acceptance_threshold_friction
        }

    def to_json(self):
        """
        Serialise the instance to JSON.

        Returns
        ----------
        str
        """

        return json.dumps(self.data(), indent=4)

    def save(self, filename: str):
        """
        Write the instance to a file.

        Parameters
        ----------
        filename: str
            Path to file.
        """

        with open(filename, "w") as file:
            file.write(self.to_json())
