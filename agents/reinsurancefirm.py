"""
Contains all the capabilities of Reinsurance Firms
"""

import numpy as np

class ReinsuranceFirm:
    def __init__(self, reinsurancefirm_id, reinsurancefirm_args):
        self.reinsurancefirm_id = reinsurancefirm_id
        self.initial_capital = reinsurancefirm_args['initial_capital']
        self.deductible = reinsurancefirm_args['deductible']
        self.market_entry_probability = reinsurancefirm_args['market_entry_probability']
        self.exit_capital_threshold = reinsurancefirm_args['exit_capital_threshold']
        self.exit_time_limit = reinsurancefirm_args['exit_time_limit']
        self.sensitivity_premium = reinsurancefirm_args['sensitivity_premium']


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