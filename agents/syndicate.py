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

        self.status = False   # status True means active, status False means exit (no contract or bankruptcy, at the begining the status is 0 because no syndicate joining the market

        self.current_hold_contracts = []
        # Include paid status, True or False
        self.paid_claim = []

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

    def received_risk(self, risk_id, broker_id, start_time):
        """
        After broker send risk to the market, all the active syndicate receive risks and update their received_risk list
        """
        self.received_risk_list.append({"risk_id": risk_id,
                                  "broker_id": broker_id,
                                  "start_time": start_time})

    def add_leader(self, risks, line_size, premium):
        """
        Add new contract to the play_leader_in_contracts list
        """
        self.play_leader_in_contracts.append({"risk_id": risks.get("risk_id"),
                                    "broker_id": risks.get("broker_id"),
                                    "line_size": line_size,
                                    "premium": premium})

    def add_follower(self, risks, line_size, premium):
        """
        Add new contract to the play_follower_in_contracts list
        """
        self.play_follower_in_contracts.append({"risk_id": risks.get("risk_id"),
                                    "broker_id": risks.get("broker_id"),
                                    "line_size": line_size,
                                    "premium": premium})

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
                                    "risk_end_time": risks.get("risk_start_time")+365,
                                    "pay": False})
        for i in range(len(self.current_capital_category)):
            if i == int(risks.get("risk_category")):
                self.current_capital_category[i] -= risks.get("risk_value")

    def pay_claim(self, broker_id, category_id, claim_value, pay_value):
        """
        Pay claim for ended contract and for contracts affected by catastrophe
        """
        for i in range(len(self.current_hold_contracts)):
            if (int(self.current_hold_contracts[i]["broker_id"]) == int(broker_id)) and (int(self.current_hold_contracts[i]["risk_category"]) == int(category_id)): 
                if self.current_capital >= claim_value:  
                    self.current_hold_contracts[i]["pay"] = True      
                    self.current_capital -= claim_value
                else:
                    self.current_capital -= claim_value

    def receive_premium(self, premium, category_id):
        """
        Receive premium from brokers
        """
        self.current_capital_category[int(category_id)] += premium
        self.current_capital += premium

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
