from __future__ import annotations
import json
import scipy
import math
import numpy as np
import random
from environment.event import TruncatedDistWrapper
from environment.risk_model import RiskModel

class RiskGenerator:
    """
    Generate all the risks for the simulation in the form of RiskModel
    """
    def __init__(self, num_risk_models, sim_args, broker_args, risk_args, seed):
        """
        Instance of risks for a simulation

        Parameters
        ---------- 
        num_risk_models: int 
        sim_args: dict
        risk_args: dict
        seed: int
        """
        # Get inputs
        self.num_riskmodels = num_risk_models
        self.sim_args = sim_args
        self.broker_args = broker_args
        self.risk_args = risk_args
        self.sim_time_span = sim_args["max_time"]
        self.num_risks = risk_args["num_risks"]
        self.num_categories = risk_args["num_categories"]
        self.riskmodel_margin_of_safety = risk_args["riskmodel_margin_of_safety"]
        self.risk_factor_lower_bound = risk_args["risk_factor_lower_bound"]
        self.risk_factor_upper_bound = risk_args["risk_factor_upper_bound"]
        self.expire_immediately = risk_args["expire_immediately"]
        self.catastrophe_time_mean_separation = risk_args["catastrophe_time_mean_separation"]
        self.mean_contract_runtime = sim_args["mean_contract_runtime"]
        self.norm_profit_markup = risk_args["norm_profit_markup"]
        self.money_supply = risk_args["money_supply"]
        self.inaccuracy_riskmodels = risk_args["riskmodel_inaccuracy"]
        self.norm_profit_markup = risk_args["norm_profit_markup"]
        self.value_at_risk_tail_probability = risk_args["value_at_risk_tail_probability"]
        self.risk_limit = risk_args["risk_limit"]
        self.lambda_attritional_loss = risk_args["lambda_attritional_loss"]
        self.cov_attritional_loss = risk_args["cov_attritional_loss"]
        self.mu_attritional_loss = risk_args["mu_attritional_loss"]
        self.lambda_catastrophe = risk_args["lambda_catastrophe"]
        self.pareto_shape = risk_args["pareto_shape"]
        self.minimum_catastrophe_damage = risk_args["minimum_catastrophe_damage"]
        self.var_em_exceedance_probability = risk_args["var_em_exceedance_probability"]
        self.var_em_safety_factor = risk_args["var_em_safety_factor"]
        self.risk_factor_lower_bound = risk_args["risk_factor_lower_bound"]
        self.risk_factor_upper_bound = risk_args["risk_factor_upper_bound"]
        # Init list of risks
        self.catastrophes = []
        self.catastrophe_time = []
        self.catastrophe_damage = [] 
        self.broker_risks = []
        self.seed  = seed

    def get_all_riskmodel_combinations(self, n, rm_factor):
        riskmodels = []
        for i in range(self.num_categories):
            riskmodel_combination = rm_factor * np.ones(self.num_categories)
            riskmodel_combination[i] = 1/rm_factor
            riskmodels.append(riskmodel_combination.tolist())
        return riskmodels
    
    def generate_risks(self):
        """
        Generate catestrophes

        Returns
        ----------
        dict: risks and risk_model_configs
        """

        # Generate Catastrophes
        non_truncated = scipy.stats.pareto(b=2, loc=0, scale=0.25)
        self.damage_distribution = TruncatedDistWrapper(lower_bound=0.25, upper_bound=1., dist=non_truncated)
        self.catastrophe_separation_distribution = scipy.stats.expon(0, self.catastrophe_time_mean_separation)
        i = 0
        for j in range(self.num_categories):
            total = 0
            while (total < self.sim_time_span):
                separation_time = self.catastrophe_separation_distribution.rvs()
                total += int(math.ceil(separation_time))
                if total < self.sim_time_span:
                    risk_start_time = total
                    damage_value = self.damage_distribution.rvs()
                    self.catastrophes.append({"catastrophe_id": i,
                      "catastrophe_start_time": risk_start_time,
                      "catastrophe_category": j,
                      "catastrophe_value": damage_value[0],
                      })
                    i += 1
        self.catastrophe_time = [0 for x in range(self.sim_time_span)]
        self.catastrophe_damage = [0 for x in range(self.sim_time_span)]
        for i in range(self.sim_time_span):
            for k in range(len(self.catastrophes)):
                if self.catastrophes[k]["catastrophe_start_time"] == i:
                    self.catastrophe_time[i] = 1
                    self.catastrophe_damage[i] = self.catastrophes[k]["catastrophe_value"]

        # Compute remaining parameters
        self.risk_factor_spread = self.risk_factor_upper_bound - self.risk_factor_lower_bound
        self.risk_factor_distribution = scipy.stats.uniform(loc=self.risk_factor_lower_bound, scale=self.risk_factor_spread)
        self.risk_value_distribution = scipy.stats.uniform(loc=1000, scale=0)

        risk_factor_mean = self.risk_factor_distribution.mean()
        if np.isnan(risk_factor_mean):
            risk_factor_mean = self.risk_factor_distribution.rvs()

        if self.expire_immediately:
            assert self.catastrophe_separation_distribution.dist.name == 'expon'
            expected_damage_frequency = 1 - scipy.stats.poisson(1 / self.catastrophe_time_mean_separation * self.mean_contract_runtime).pmf(0)
        else:
            expected_damage_frequency = self.mean_contract_runtime / self.catastrophe_separation_distribution.mean()

        self.norm_premium = expected_damage_frequency * self.damage_distribution.mean() * risk_factor_mean * (1 + self.norm_profit_markup)
        self.market_premium = self.norm_premium

        # Set up risk models
        risk_value_mean = self.risk_value_distribution.mean()
        if np.isnan(risk_value_mean):
            risk_value_mean = self.risk_value_distribution.rvs()
        self.inaccuracy = self.get_all_riskmodel_combinations(self.num_categories, self.inaccuracy_riskmodels)
        self.inaccuracy = random.sample(self.inaccuracy, self.num_riskmodels)

        risk_model_configs = [{"damage_distribution": self.damage_distribution,
                                "expire_immediately": self.expire_immediately,
                                "catastrophe_separation_distribution": self.catastrophe_separation_distribution,
                                "norm_premium": self.norm_premium,
                                "num_categories": self.num_categories,
                                "risk_value_mean": risk_value_mean,
                                "risk_factor_mean": risk_factor_mean,
                                "norm_profit_markup": self.norm_profit_markup,
                                "margin_of_safety": self.riskmodel_margin_of_safety,
                                "var_tail_prob": self.value_at_risk_tail_probability,
                                "inaccuracy_by_categ": self.inaccuracy[i]
                                } for i in range(self.num_riskmodels)]

        """# Generate risks brought by brokers by poisson distribution
        risks_schedule = [[] for x in range(self.broker_args["num_brokers"])]
        risks_categories = [[] for x in range(self.broker_args["num_brokers"])]
        risks_factors = [[] for x in range(self.broker_args["num_brokers"])]
        risks_values = [[] for x in range(self.broker_args["num_brokers"])]
        for i in range(self.broker_args["num_brokers"]):
            risks_schedule[i] = [] # Record the time for broker risks start time
            np.random.seed(self.seed + i)
            risks_categories[i] = np.random.randint(0, self.num_categories, size = self.sim_time_span)
            risks_factors[i] = self.risk_factor_distribution.rvs(size = self.sim_time_span)
            risks_values[i] = self.risk_value_distribution.rvs(size = self.sim_time_span)
            # For each day generate risks according to poisson distribution
            np.random.seed(self.seed + i)
            s = np.random.poisson(self.broker_args["lambda_risks_daily"], self.sim_time_span)
            for k in range(len(s)):
                risks_schedule[i].append(s[k])
            print(len(risks_schedule[i]))
        risk_num = 0
        for i in range(self.broker_args["num_brokers"]):
            for j in range(self.sim_time_span):
                if risks_schedule[i][j] > 0:
                    self.broker_risks.append({"risk_id": risk_num,
                                            "broker_id": i,
                                            "risk_start_time": j,
                                            "risk_factor": risks_factors[i][j],
                                            "risk_category": risks_categories[i][j],
                                            "risk_value": risks_values[i][j]
                                            })
                    risk_num += 1"""
        
        risks_schedule = [[] for x in range(self.broker_args["num_brokers"])]
        risks_categories = [[] for x in range(self.broker_args["num_brokers"])]
        risks_factors = [[] for x in range(self.broker_args["num_brokers"])]
        risks_values = [[] for x in range(self.broker_args["num_brokers"])]
        for i in range(self.broker_args["num_brokers"]):
            risks_schedule[i] = [] # Record the time for broker risks start time
            np.random.seed(self.seed + i)
            risks_categories[i] = np.random.randint(0, self.num_categories, size = self.sim_time_span)
            risks_factors[i] = self.risk_factor_distribution.rvs(size = self.sim_time_span)
            risks_values[i] = self.risk_value_distribution.rvs(size = self.sim_time_span)
        risk_num = 0
        for i in range(self.broker_args["num_brokers"]):
            for j in range(self.sim_time_span):
                for k in range(int(self.broker_args["lambda_risks_daily"])):
                    self.broker_risks.append({"risk_id": risk_num,
                                            "broker_id": i,
                                            "risk_start_time": j,
                                            "risk_factor": risks_factors[i][j],
                                            "risk_category": risks_categories[i][j],
                                            "risk_value": risks_values[i][j]
                                            })
                    risk_num += 1

        return self.catastrophes, self.broker_risks, self.market_premium, risk_model_configs

    def data(self):
        """
        Get the data as a serialisable dictionary.

        Returns
        ----------
        dict
        """
        
        return [{
            "id": self.catastrophes[i].get("risk_id"),
            "time": self.catastrophes[i].get("risk_start_time"),
            "risk_factor": self.catastrophes[i].get("risk_factor"),
            "risk_category": self.catastrophes[i].get("risk_category"),
            "risk_value": self.catastrophes[i].get("risk_value")}
        for i in range(self.num_risks)] 

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
