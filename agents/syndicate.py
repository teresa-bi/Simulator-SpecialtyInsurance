"""
Contains all the capabilities of Syndicates
"""

import numpy as np
import os

class Syndicate:
    def __init__(self, syndicate_id, syndicate_args):
        self.syndicate_id = syndicate_id
        self.initial_capital = syndicate_args['initial_capital']
        self.current_capital = syndicate_args['initial_capital']
        self.premium_internal_weight = syndicate_args['actuarial_pricing_internal_weight']
        self.interest_rate = syndicate_args['interest_rate_monthly']
        self.leader = syndicate_args['leader']
        self.lead_line_size = syndicate_args['lead_line_size']
        self.follower = syndicate_args['follower']
        self.follow_line_size = syndicate_args['follow_line_size']
        self.dividends_of_profit = syndicate_args['dividends_of_profit']
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
        self.sensitivity_premium = syndicate_args['sensitivity_premium']
        self.initial_acceptance_threshold = syndicate_args['initial_acceptance_threshold']
        self.acceptance_threshold_friction = syndicate_args['acceptance_threshold_friction']
        self.upper_price_limit = syndicate_args['upper_price_limit']
        self.lower_price_limit = syndicate_args['lower_price_limit']

    def update_capital(self, ):
        """
        Calculate the current capital after receiving premium from broker, paying each claim, paying premium to reinsurance firms, receiving payments from reinsurance firms, and paying dividends
        """
        self.initial_capital
        self.syndicate_id

        return current_capital

    def update_underwrite_risk_regions(self, ):
        """
        Calculate the capitals in each covered risk region
        """
        self.syndicate_id

    def calculate_premium(self, ):
        """
        Calculate the premium based on the past experience and industry statistics
        """
        self.premium_internal_weight

        return premium

    def offer_lead_quote(self, broker_id, risk_id):
        """
        If the syndicate is chosen to be in the list of top k leaders, offer the lead quote depending on its capital and risk balance policy
        """

        return (leader, lead_line_size, broker_id, risk_id)

    def offer_follow_quote(self, broker_id, risk_id):
        """
        If the syndicate is chosen to be in the list of top k followers, offer the follow quote depending on its capital and risk balance policy
        """

        return (follower, follow_line_size, broker_id, risk_id)

    def ask_premium_from_broker(self, ):
        
        premium = calculate_premium()

    def ask_interests(self, ):

        interests = ( 1 + self.interest_rate) * update_capital()

        return current_capital

    def pay_claim(self, ):
        """
        Pay for claim based on risk region and line size
        """
        if self.leader: 
            self.lead_line_size
        elif self.follower:
            self.syndicate_follow_line_size
        else:
            payment = 0

        return payment

    def ask_payment_from_reinsurancefirms(self, ):
        """
        Receive payment from reinsurance firms
        """
        self.syndicate_id

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