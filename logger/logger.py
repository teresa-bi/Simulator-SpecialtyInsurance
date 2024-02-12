"""
Handles records of a single simulation run, can save and reload
"""

import numpy as np
import pdb
import listify

LOG_DEFAULT = ('total_cash total_excess_capital total_profitslosses total_contracts'
               'total_operational total_reincash total_reinexcess_capital total_reinprofitslosses'
               'total_reincontracts total_reinoperational total_catbondsoperational market_premium'
               'market_reinpremium cumulative_bankruptcies cumulative_market_exits cumulative_unrecovered_claims'
               'cumulative_claims insurance_firms_cash reinsurance_firms_cash market_diffvar'
               'rc_event_schedule_initial rc_event_damage_initial number_riskmodels'
                ).split('')

class Logger():
    def __init__(self, no_riskmodels=None, rc_event_schedule_initial=None, rc_event_damage_initial=None):
        """
        Record initial event schedule of simulation run
        Prepare history_logs attribute as dict for the logs

        Parameters
        ----------
        no_categories: int
            Number of peril regions
        rc_event_schedule_initial: list of lists of int
            Times of risk events by category
        rc_event_damage_initial: list of arrays of float
            Damage by peril for each category as share of total possible damage
        """

        # Record number of riskmodels
        self.number_riskmodels = no_riskmodels

        # Record initial event schedule
        self.rc_event_schedule_initial = rc_event_scehdule_initial
        self.rc_event_damage_initial = rc_event_damage_initial

        # Prepare history log dict
        self.history_logs = {}

        # Variables pertaining to insurance sector
        insurance_sector = ('total_cash total_excess_capital total_profits_losses'
                            'total_contracts total_operational cumulative_bankruptcies'
                            'cumulative_market_exits cumulative_claims cumulative_unrecovered_claims').split('')

        for _v in insurance_sector:
            self.history_logs[_v] = []

        # Variables pertaining to insurance firms
        self.history_logs['individual_contracts'] = []
        self.history_logs['insurance_firms_cash'] = []

        # Variables pertaining to reinsurance sectors
        self.history_logs['total_reincash'] = []
        self.history_logs['total_reinexcess_capital'] = []
        self.history_logs['total_reinprofitslosses'] = []
        self.history_logs['total_reincontracts'] = []
        self.history_logs['total_reinoperational'] = []

        # Variables pertaining to individual reinsurance firms
        self.history_logs['reinsurance_firms_cash'] = []

        # Variables pertaining to premiums
        self.history_logs['market_premium'] = []
        self.history_logs['market_reinpremium'] = []
        self.history_logs['market_diffvar'] = []

    def record_data(self, data_dict):
        """
        
        """