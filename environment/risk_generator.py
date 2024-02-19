from __future__ import annotations
import json
import scipy
import numpy as np


class RiskGenerator:
    """
    Generate all the risks for the simulation in the form of RiskModel
    """
    def __init__(self, num_risk_models, sim_args, risk_args):
        """
        Instance of risks for a simulation

        Parameters
        ---------- 
        num_risk_models: int 
        sim_args: dict
        risk_args: dict
        """
        # Get inputs
        self.num_riskmodels = num_risk_models
        self.sim_args = sim_args
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
        self.inaccuracy_riskmodels = risk_args["inaccuracy_riskmodels"]
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
        self.risks = {}

    def generate_risks(self):
        """
        Generate risks including catestrophe and attritional loss(add later)

        Returns
        ----------
        dict: risks and risk_model_configs
        """

        # Compute remaining parameters
        non_truncated = scipy.stats.pareto(b=2, loc=0, scale=0.25)
        self.damage_distribution = TruncatedDistWrapper(lower_bound=0.25, upper_bound=1., dist=non_truncated)
        self.risk_factor_spread = self.risk_factor_upper_bound - self.risk_factor_lower_bound
        self.risk_factor_distribution = scipy.stats.uniform(loc=self.risk_factor_lower_bound, scale=self.risk_factor_spread)
        self.risk_value_distribution = scipy.stats.uniform(loc=1000, scale=0)

        risk_factor_mean = self.risk_factor_distribution.mean()
        if np.isnan(risk_factor_mean):
            risk_factor_mean = self.risk_factor_distribution.rvs()

        self.catastrophe_separation_distribution = scipy.stats.expon(0, self.catastrophe_time_mean_separation)
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

        # Set up risks including risk_id, start_time, risk_factor, category, risk_value, riskmodel_configurations
        risks_starttime = np.random.randint(0, self.sim_time_span, size = self.num_risks) # How to set as lambda distribution
        risks_factors = self.risk_factor_distribution.rvs(size = self.num_risks)
        risks_categories = np.random.randint(0, self.num_categories, size = self.num_risks)
        risks_values = self.risk_value_distribution.rvs(size = self.num_risks)
        self.risks = [{"risk_id": i,
                      "risk_start_time": risks_starttime[i],
                      "risk_factor": risks_factors[i],
                      "risk_category": risks_categories[i],
                      "risk_value": risks_values[i],
                      } for i in range(self.num_risks)]

        return self.risks, risk_model_configs

    def data(self):
        """
        Get the data as a serialisable dictionary.

        Returns
        ----------
        dict
        """

        return {
            "id": self.risks[i].get("risk_id"),
            "time": self.risks[i].get("risk_start_time"),
            "risk_factor": self.risks[i].get("risk_factor"),
            "risk_category": self.risks[i].get("risk_category"),
            "risk_value": self.risks[i].get("risk_value")
            for i in range(self.num_risks)}

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
