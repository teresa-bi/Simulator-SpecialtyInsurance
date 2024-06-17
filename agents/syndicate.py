"""
Contains all the capabilities of Syndicates
"""

import numpy as np
import scipy.stats
import copy
from environment.risk_model import RiskModel

def get_mean(x):
    return sum(x) / len(x)

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
        self.insurance_permanency_contracts_limit = syndicate_args["insurance_permanency_contracts_limit"]
        self.insurance_permanency_ratio_limit = syndicate_args["insurance_permanency_ratio_limit"]

        self.contract_runtime_dist = scipy.stats.randint(sim_args["mean_contract_runtime"] - sim_args["contract_runtime_halfspread"], sim_args["mean_contract_runtime"] + sim_args["contract_runtime_halfspread"]+1)
        self.default_contract_payment_period = sim_args["default_contract_payment_period"]
        self.simulation_reinsurance_type = sim_args["simulation_reinsurance_type"]
        self.market_permanency_off = sim_args["market_permanency_off"]
        self.mean_contract_runtime = sim_args["mean_contract_runtime"]
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

        self.ambiguity_level = syndicate_args['ambiguity_level']
        self.cost_of_capital = syndicate_args['cost_of_capital']
        self.min_cat_prob_distortion = self.riskmodel_config["min_cat_prob_distortion"]
        self.max_cat_prob_distortion = self.riskmodel_config["max_cat_prob_distortion"]

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
                                inaccuracy = self.riskmodel_config["inaccuracy_by_categ"],
                                ambiguity = self.ambiguity_level,
                                min_cat_prob_distortion = self.min_cat_prob_distortion,
                                max_cat_prob_distortion = self.max_cat_prob_distortion)

        self.category_reinsurance = [None for i in range(self.num_risk_categories)]

        if self.simulation_reinsurance_type == 'non-proportional':
            if self.non_proportional_reinsurance_level is not None:
                self.np_reinsurance_deductible_fraction = self.non_proportional_reinsurance_level
            else:
                self.np_reinsurance_deductible_fraction = syndicate_args["default_non-proportional_reinsurance_deductible"]
            self.np_reinsurance_excess_fraction = syndicate_args["default_non-proportional_reinsurance_excess"]
            self.np_reinsurance_premium_share = syndicate_args["default_non-proportional_reinsurance_premium_share"]

        self.obligations = []
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
        self.capital_left_by_categ = self.current_capital_category
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
                                    "initial_VaR": risks.get("risk_VaR"),
                                    "pay": None})
        for i in range(len(self.current_capital_category)):
            if i == int(risks.get("risk_category")):
                self.current_capital_category[i] -= premium

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
        
        if self.status:

            for contract in self.current_hold_contracts:
                self.counter_category[contract.risk_category] = self.counter_category[contract.risk_category] + 1
                self.var_category[contract.risk_category] = self.var_category[contract.risk_category] + contract.initial_VaR

            for category in range(len(self.counter_category)):
                self.var_counter = self.var_counter + self.counter_category[category] * self.riskmodel.inaccuracy[category]
                self.var_sum = self.var_sum + self.var_category[category]

            if not sum(self.counter_category) == 0:
                self.var_counter_per_risk = self.var_counter / sum(self.counter_category)
            else:
                self.var_counter_per_risk = 0
    
    def market_permanency(self):     #This method determines whether an insurer stays in the market. If it has very few risks underwritten or too much cash left for TOO LONG it eventually leaves the market.
                                                      # If it has very few risks underwritten it cannot balance the portfolio so it makes sense to leave the market.
        if not self.market_permanency_off:

            capital_left_by_categ = np.asarray(self.capital_left_by_categ)

            avg_capital_left = get_mean(capital_left_by_categ)

            if self.current_capital < self.capital_permanency_limit:         #If their level of cash is so low that they cannot underwrite anything they also leave the market.
                self.current_hold_contracts = []
                self.risks_kept = []
                self.reinrisks_kept = []
                self.excess_capital = 0                 
                self.profits_losses = 0                 
                self.status = False
            else:
                if len(self.current_hold_contracts) < self.insurance_permanency_contracts_limit or avg_capital_left / self.current_capital > self.insurance_permanency_ratio_limit:
                    #Insurers leave the market if they have contracts under the limit or an excess capital over the limit for too long.
                    self.market_permanency_counter += 1
                else:
                    self.market_permanency_counter = 0                                    #All these limits maybe should be parameters in isleconfig.py

                if self.market_permanency_counter >= self.exit_time_limit:    # Here we determine how much is too long.
                    self.current_hold_contracts = []
                    self.risks_kept = []
                    self.reinrisks_kept = []
                    self.excess_capital = 0                 
                    self.profits_losses = 0                 
                    self.status = True

    def adjust_market_premium(self, norm_premium):
        """Adjust_market_premium Method.
               Accepts arguments
                   capital: Type float. The total capital (cash) available in the insurance market (insurance only).
               No return value.
           This method adjusts the premium charged by insurance firms for the risks covered. The premium reduces linearly
           with the capital available in the insurance market and viceversa. The premium reduces until it reaches a minimum
           below which no insurer is willing to reduce further the price. """
        self.market_premium = norm_premium * (self.upper_premium_limit
                                                   - self.premium_sensitivity
                                                   * self.current_capital / (self.initial_capital
                                                   * self.riskmodel_config["damage_distribution"].mean() * 30 * 4 / 6))
        if self.market_premium < norm_premium * self.lower_premium_limit:
            self.market_premium = norm_premium * self.lower_premium_limit
        return self.market_premium 
    
    def reserve_capital(self, risk):
        var = self.riskmodel.get_var(risk)
        var = self.ambiguity_level * (1+self.max_cat_prob_distortion) * var + (1-self.ambiguity_level) * (1+self.min_cat_prob_distortion) * var
        return var

    def offer_premium(self, risk):
        """
        Offer premium based on the syndicate's loss probability model, current capital, ambiguity level
        """
        expected_damage_frequency = self.mean_contract_runtime / self.riskmodel_config["catastrophe_separation_distribution"].mean() 
        expected_damage_loss = expected_damage_frequency * self.riskmodel_config["risk_factor_mean"] * self.riskmodel_config["damage_distribution"].mean()
        #self.norm_premium = expected_damage_frequency * self.riskmodel_config["damage_distribution"].mean() * self.riskmodel_config["risk_factor_mean"] * (1 + self.riskmodel_config["norm_profit_markup"])
        norm_premium = (expected_damage_loss + self.cost_of_capital * self.reserve_capital(risk) / risk.risk_value) * (1 + self.riskmodel_config["norm_profit_markup"])
        norm_premium = self.adjust_market_premium(norm_premium)

        return norm_premium[0]

    def reset_pl(self):
        """Reset_pl Method.
               Accepts no arguments:
               No return value.
           Reset the profits and losses variable of each firm at the beginning of every iteration. It has to be run in insurancesimulation.py at the beginning of the iterate method"""
        self.profits_losses = 0

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
