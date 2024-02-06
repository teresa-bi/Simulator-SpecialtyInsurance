import json        

class CatastropheEvent:
    """
    Generate catastrophes
    """
    def __init__(self, catastrophe_id, catastrophe_riskregions, catastrophe_value):
        self.catastrophe_id = catastrophe_id
        self.catastrophe_riskregions = catastrophe_riskregions
        self.catastrophe_value = catastrophe_value
        self.num_riskregions = risk_args['num_riskregions']
        self.risk_limit = risk_args['risk_limit']
        self.inaccuracy_riskmodels = risk_args['inaccuracy_riskmodels']
        self.margin_of_safety = risk_args['margin_of_safety']
        self.value_at_risk_tail_probability = risk_args['value_at_risk_tail_probability']
        self.catastrophe_time_mean_separation = risk_args['catastrophe_time_mean_separation']
        self.lambda_attritional_loss = risk_args['lambda_attritional_loss']
        self.cov_attritional_loss = risk_args['cov_attritional_loss']
        self.mu_attritional_loss = risk_args['mu_attritional_loss']
        self.lambda_catastrophe = risk_args['lambda_catastrophe']
        self.pareto_shape = risk_args['pareto_shape']
        self.minimum_catastrophe_damage = risk_args['minimum_catastrophe_damage']
        self.var_em_exceedance_probability = risk_args['var_em_exceedance_probability']
        self.var_em_safety_factor = risk_args['var_em_safety_factor']
        self.risk_factor_lower_bound = risk_args['risk_factor_lower_bound']
        self.risk_factor_upper_bound = risk_args['risk_factor_upper_bound']
        
    def run(self, scenario):
        """
        Add catastrophe to the base scenario NoReinsurance_RiskOne

        Parameters
        ----------
        scenario: NoReinsurance_RiskOne
            The scenario to accept catastrophe event

        Returns
        -------
        scenario: NoReinsurance_RiskOne
            The updated scenario
        """

        scenario.catastrophe_event[self.catastrophe_id] = {"catastrophe_id": self.catastrophe_id,
                                        "catastrophe_riskregions": self.catastrophe_riskregions,
                                        "catastrophe_value": self.catastrophe_value
                                        }
        return scenario

    def data(self):
        """
        Get the data as a serialisable dictionary.

        Returns
        --------
        dict
        """

        return {
            self.__class__.__name__: {
                "catastrophe_id": self.catastrophe_id,
                "catastrophe_riskregions": self.catastrophe_riskregions,
                "catastrophe_value": self.catastrophe_value
            }
        }

    def to_json(self):
        """
        Serialise the instance to JSON.

        Returns
        ----------
        str
        """

        return json.dumps(self.data(), indent=4)

    def save(self, filename):
        """
        Write the instance to a log file.

        Parameters
        ----------
        filename: str
            Path to file.
        """

        with open(filename, "w") as file:
            file.write(self.to_json())
