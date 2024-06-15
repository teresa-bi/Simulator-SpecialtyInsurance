"""
Handles records of a single simulation run, can save and reload
"""

import numpy as np
import pdb
import listify

"""
LOG_DEFAULT = ('total_cash total_excess_capital total_profits_losses total_contracts'
               'total_operational total_reincash total_reinexcess_capital total_reinprofitslosses'
               'total_reincontracts total_reinoperational total_catbondsoperational market_premium'
               'market_reinpremium cumulative_bankruptcies cumulative_market_exits cumulative_unrecovered_claims'
               'cumulative_claims insurance_firms_cash reinsurance_firms_cash market_diffvar'
               'rc_event_schedule_initial rc_event_damage_initial number_riskmodels'
                ).split('')
"""
LOG_DEFAULT = ('total_cash syndicateA_cash syndicateB_cash syndicateC_cash syndicateD_cash syndicateE_cash syndicateF_cash total_excess_capital total_profits_losses total_contracts syndicateA_contracts syndicateB_contracts syndicateC_contracts syndicateD_contracts syndicateE_contracts syndicateF_contracts uncovered_risks total_operational market_premium cumulative_bankruptcies cumulative_market_exits cumulative_uncovered_claims cumulative_claims insurance_firms_cash risk_event_schedule_initial risk_event_damage_initial number_riskmodels'
                ).split(" ")

class Logger():
    def __init__(self, no_riskmodels=None, risk_event_scehdule_initial=None, risk_event_damage_initial=None, initial_broker=None, initial_syndicate=None):
        """
        Record initial event schedule of simulation run
        Prepare history_logs attribute as dict for the logs

        Parameters
        ----------
        no_categories: int
            Number of peril regions
        risk_event_schedule_initial: list of lists of int
            Times of risk events by category
        initial_broker: data of initial brokers
        initial_syndicate: data of initial syndicates
        """

        # Record number of riskmodels
        self.number_riskmodels = no_riskmodels

        # Record initial event schedule
        self.risk_event_schedule_initial = risk_event_scehdule_initial
        self.risk_event_damage_initial = risk_event_damage_initial

        # Record agent information
        self.broker = initial_broker
        self.syndicate = initial_syndicate

        # Prepare history log dict
        self.history_logs = {}

        # Variables pertaining to insurance firms
        self.history_logs['individual_contracts'] = []
        self.history_logs['insurance_firms_cash'] = []

        # Variables pertaining to premiums
        self.history_logs['total_cash'] = []
        self.history_logs['syndicateA_cash'] = []
        self.history_logs['syndicateB_cash'] = []
        self.history_logs['syndicateC_cash'] = []
        self.history_logs['syndicateD_cash'] = []
        self.history_logs['syndicateE_cash'] = []
        self.history_logs['syndicateF_cash'] = []
        self.history_logs['total_excess_capital'] = []
        self.history_logs['total_profits_losses'] = []
        self.history_logs['total_contracts'] = []
        self.history_logs['syndicateA_contracts'] = []
        self.history_logs['syndicateB_contracts'] = []
        self.history_logs['syndicateC_contracts'] = []
        self.history_logs['syndicateD_contracts'] = []
        self.history_logs['syndicateE_contracts'] = []
        self.history_logs['syndicateF_contracts'] = []
        self.history_logs['uncovered_risks'] = []
        self.history_logs['total_operational'] = []
        self.history_logs['market_premium'] = []
        self.history_logs['cumulative_bankruptcies'] = []
        self.history_logs['cumulative_market_exits'] = []
        self.history_logs['cumulative_uncovered_claims'] = []
        self.history_logs['cumulative_claims'] = []
        self.history_logs["number_riskmodels"] = []
        self.history_logs["risk_event_schedule_initial"] = []
        self.history_logs["risk_event_damage_initial"] = []

    def record_data(self, data_dict):
        """
        Record data for one period

        Parameters
        ----------
        data_dict: dict
            Data with the same keys as used in self.history_log()
        """
        for key in data_dict.keys():
            if key != "individual_contracts":
                self.history_logs[key].append(data_dict[key])
            else:
                for i in range(len(data_dict["individual_contracts"])):
                    self.history_logs["individual_contracts"].append(data_dict["individual_contracts"][i])

    def listify(self, d):
        """Function to convert dict to list with keys in last list element.
        Arguments:
            d: dict - input dict
        Returns:
            list with dict values as elements [:-1] and dict keys as 
                last element."""
                    
        """extract keys"""
        keys = list(d.keys())
    
        """create list"""
        l = [d[key] for key in keys]
        l.append(keys)
    
        return l
    
    def delistify(self, l):
        """Function to convert listified dict back to dict.
        Arguments:
            l: list - input listified dict. This must be a list of dict 
                        elements as elements [:-1] and the corresponding
                        dict keys as list in the last element.
        Returns:
            dict - The restored dict."""
            
        """extract keys"""
        keys = l.pop()
        assert len(keys) == len(l)
    
        """create dict"""
        d = {key: l[i] for i,key in enumerate(keys)}
    
        return d

    def obtain_log(self, requested_logs=LOG_DEFAULT):
        """
        Transfer the log in the cloud
        """
        self.history_logs["number_riskmodels"] = self.number_riskmodels
        self.history_logs["risk_event_schedule_initial"] = self.risk_event_schedule_initial
        self.history_logs["risk_event_damage_initial"] = self.risk_event_damage_initial

        if requested_logs == None:
            requested_logs = LOG_DEFAULT

        log = {name: self.history_logs[name] for name in requested_logs}

        # Convert to list and return
        return self.listify(log)

    def restore_logger_object(self, log):
        """
        A log can be restored later, it can be restored on a different machine. This is useful in the case of ensemble runs to move the log to the master node from the computation nodes
        """

        # Restore dict
        log = self.delistify(log)

        # Extract environment variables
        self.risk_event_schedule_initial = log["risk_event_schedule_initial"]
        self.risk_event_damage_initial = log["risk_event_damage_initial"]
        self.number_riskmodels = log["number_riskmodels"]
        del log["risk_event_schedule_initial"], log["risk_event_damage_initial"], log["number_riskmodels"]

        # Restore history log
        self.history_logs = log
    
    def save_log(self, leadsize, ambiguity):
        """
        Save log to disk of local machine

        Parameter
        ---------
        background_run: bool
            An ensemble run (True) or not (False)
        """
        to_log = self.single_log_prepare(leadsize, ambiguity)

        for filename, data, operation_character in to_log:
            with open(filename, operation_character) as wfile:
                wfile.write(str(data) + "\n")

    def replication_log_prepare(self):
        """
        Prepare writing taks for ensemble run saving
        """
        filename_prefix = {1:"one", 2:"two", 3:"three", 4:"four"}
        fpf = filename_prefix[self.number_riskmodels]
        to_log = []
        to_log.append(("data/" + fpf + "_history_logs.dat", self.history_logs, "a"))
        return to_log

    def single_log_prepare(self, leadsize, ambiguity):
        """
        Prepare writing tasks for single run saving
        """
        to_log = []
        to_log.append(("history_logs_" + str(leadsize) + "_" + str(ambiguity) + ".dat", self.history_logs, "w"))
        return to_log

    def add_insurance_agent(self):
        """
        Add an additional insurer agent to the history log
        """
        if len(self.history_logs['individual_contracts']) > 0:
            zeros_to_append = list(np.zeros(len(self.history_logs["inidividual_contracts"][0]), dtype=int))
        else:
            zeros_to_append = []
        self.history_logs["individual_contracts"].append(zeros_to_append)