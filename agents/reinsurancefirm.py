"""
Contains all the capabilities of Reinsurance Firms
"""

import numpy as np

class ReinsuranceFirm:
    def __init__(self, reinsurancefirm_id, reinsurancefirm_args, num_risk_models, risk_model_configs):
        self.reinsurancefirm_id = reinsurancefirm_id
        self.initial_capital = reinsurancefirm_args['initial_capital']
        self.deductible = reinsurancefirm_args['deductible']
        self.market_entry_probability = reinsurancefirm_args['market_entry_probability']
        self.exit_capital_threshold = reinsurancefirm_args['exit_capital_threshold']
        self.exit_time_limit = reinsurancefirm_args['exit_time_limit']
        self.sensitivity_premium = reinsurancefirm_args['sensitivity_premium']
        self.reinsurance_list = {}  # Include syndicate_id, risk_region, risk_value

        

        self.reinsurer_id_counter = 0
        for i in range(reinsurancefirm_args["num_reinsurancefirms"]):
            reinsurance_reinsurance_level = syndicate_args["default_non_proportional_reinsurance_deductible"]
            riskmodel_config = risk_model_configurations[i % len(risk_model_configurations)]
            reinsurancefirms[i].append({"id": str(i),
                                        "initial_cash":  reinsurancefirm_args["initial_capital"],
                                        "riskmodel_config": riskmodel_config,
                                        "norm_premium": self.norm_premium,
                                        "profit_target": risk_args["norm_profit_markup"],
                                        "initial_acceptance_threshold": reinsurancefirm_args["initial_acceptance_threshold"],
                                        "acceptance_threshold_friction": reinsurancefirm_args["acceptance_threshold_friction"],
                                        "reinsurance_limit": reinsurancefirm_args["reinsurance_limit"],
                                        "non_proportional_reinsurance_level": reinsurance_reinsurance_level,
                                        "capacity_target_decrement_threshold": reinsurancefirm_args["capacity_target_decrement_threshold"],
                                "capacity_target_increment_threshold": reinsurancefirm_args["capacity_target_increment_threshold"],
                                "capacity_target_decrement_factor": reinsurancefirm_args["capacity_target_decrement_factor"],
                                "capacity_target_increment_factor": reinsurancefirm_args["capacity_target_increment_factor"],
                                "interest_rate": reinsurancefirm_args["interest_rate"]})

    def data(self):
        """
        Create a dictionary with key/value pairs representing the ReinsuranceFirms data.

        Returns
        ----------
        dict
        """

        return {
            "reinsurancefirm_id": self.reinsurancefirm_id,
            "initial_capital": self.initial_capital,
            "deductible": self.deductible,
            "market_entry_probability": self.market_entry_probability,
            "exit_capital_threshold": self.exit_capital_threshold,
            "exit_time_limit": self.exit_time_limit,
            "sensitivity_premium": self.sensitivity_premium,
            "reinsurance_list": self.reinsurance_list
        }


    def update_capital(self, ):
        """
        Calculate the current capital after receiving premium from broker, paying each claim, paying premium to reinsurance firms, receiving payments from reinsurance firms, and paying dividends
        """
        self.initial_capital
        self.reinsurancefirm_id

        return current_capital

    def ask_premium_from_syndicate(self, ):
        
        premium = calculate_premium()

    def pay_claim(self, ):
        """
        Pay for claim based on risk region and line size
        """

        return payment

    def pay_dividend(self, ):
        """
        Pay dividends to shareholders if profit
        """
        if self.current_capital > self.initial_capital:

            dividends = self.current_capital * self.dividends_of_profit

            return dividends

    def exit_market(self, ):
        """
        Exit market because of exit time limit reached or bankruptcy
        """
        self.exit_capital_threshold 
        self.exit_time_limit 