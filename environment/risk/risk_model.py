import math
import numpy as np
import sys, pdb
import scipy.stats

class RiskModel:
    """
    The risk model adopted by syndicates to cope with catastrophe
    The cash in each category, acceptable risk, profits, cash left, value_at_risk, if the risk can be accepted
    """
    def __init__(self, risk_id, risk_args):
        """
        Unpack parameters and set remaining parameters
        """
        self.num_riskmodels = risk_args["num_riskmodels"]

        non_truncated = scipy.stats.pareto(b=2, loc=0, scale=0.25)
        self.damage_distribution = TruncatedDistWrapper(lower_bound=0.25, upper_bound=1., dist=non_truncated)

        self.margin_of_safety = risk_args["riskmodel_margin_of_safety"]
        self.risk_factor_lower_bound = risk_args["risk_factor_lower_bound"]
        self.risk_factor_spread = risk_args["risk_factor_upper_bound"] - risk_args["risk_factor_lower_bound"]
        self.risk_factor_distribution = scipy.stats.uniform(loc=self.risk_factor_lower_bound, scale=self.risk_factor_spread)
        self.risk_value_distribution = scipy.stats.uniform(loc=1000, scale=0)

        risk_factor_mean = self.risk_factor_distribution.mean()
        if np.isnan(risk_factor_mean):
            risk_factor_mean = self.risk_factor_distribution.rvs()

        self.expire_immediately = risk_args["expire_immediately"]
        self.catastrophe_separation_distribution = scipy.stats.expon(0, risk_args["catastrophe_time_mean_separation"])
        if self.expire_immediately:
            assert self.catastrophe_separation_distribution.dist.name == 'expon'
            expected_damage_frequency = 1 - scipy.stats.poisson(1 /risk_args["catastrophe_time_mean_separation"] * risk_args["mean_contract_runtime"]).pmf(0)
        else:
            expected_damage_frequency = risk_args["mean_contract_runtime"] / self.catastrophe_separation_distribution.mean()

        self.norm_premium = expected_damage_frequency * self.damage_distribution.mean() * risk_factor_mean * (1 + risk_args["norm_profit_markup"])
        self.market_premium = self.norm_premium
        self.reinsurance_market_premium = self.market_premium
        self.total_no_risks = risk_args["num_risks"]

        # Set up monetary system can be set in broker
        self.money_supply = risk_args["money_supply"]
        self.obligations = []

        # Set up risk categories
        self.riskcategories = list(range(risk_args["num_categories"]))
        self.rc_event_schedule = []
        self.rc_event_damage = []
        self.rc_event_schedule_initial = []
        self.rc_event_damage_initial = []
        if rc_event_schedule is not None and rc_event_damage is not None:
            self.rc_event_schedule = copy.copy(rc_event_schedule)
            self.rc_event_schedule_initial = copy.copy(rc_event_schedule)
            self.rc_event_damage = copy.copy(rc_event_damage)
            self.rc_event_damage_initial = copy.copy(rc_event_damage)
        else:
            self.setup_risk_categories_caller()

        # Set up risks
        risk_value_mean = self.risk_value_distribution.mean()
        if np.isnan(risk_value_mean):
            risk_value_mean = self.risk_value_distribution.rvs()
        rrisk_factors = self.risk_factor_distribution.rvs(size=risk_args["num_risks"])
        rvalues = self.risk_value_distribution.rvs(size=risk_args["num_risks"])
        rcategories = np.random.randint(0,risk_args["num_categories"],size=risk_args["num_risks"])
        self.risks = [{"risk_factor":rrisk_factors[i], "value":rvalues[i], "category":rcategories[i], "owner":self} for i in range(risk_args["num_risks"])]

        self.risks_counter = [0,0,0,0]

        for item in self.risks:
            self.risks_counter[item["category"]] = self.risks_counter[item["category"]] + 1

        # Set up risk models
        self.inaccuracy = self.get_all_riskmodel_combinations(risk_args["num_categories"], risk_args["inaccuracy_riskmodels"])
        self.inaccuracy = random.sample(self.inaccuracy, risk_args["num_riskmodels"])

        risk_model_configurations = [{"damage_distribution": self.damage_distribution,
                                    "expire_immediately": risk_args["expire_immediately"],
                                    "catastrophe_separation_distribution": self.catastrophe_separation_distribution,
                                    "norm_premium": self.norm_premium,
                                    "num_categories": risk_args["num_categories"],
                                    "risk_value_mean": risk_value_mean,
                                    "risk_factor_mean": risk_factor_mean,
                                    "norm_profit_markup": risk_args["norm_profit_markup"],
                                    "margin_of_safety": risk_args["riskmodel_margin_of_safety"],
                                    "var_tail_prob": risk_args["value_at_risk_tail_probability"],
                                    "inaccuracy_by_categ": self.inaccuracy[i]
                                    } for i in range(risk_args["num_riskmodels"])]

        self.syndicate_id_counter = 0
        for i in range(risk_args["num_syndicates"]):
            insurance_reinsurance_level = syndicate_args["default_non_proportional_reinsurance_deductible"]
            riskmodel_config = risk_model_configurations[i % len(risk_model_configurations)]
            syndicates[i].append({"id": str(i),
                                "initial_cash": syndicate_args["initial_capital"],
                                "riskmodel_config": riskmodel_config,
                                "norm_premium": self.norm_premium,
                                "profit_target": risk_args["norm_profit_markup"],
                                "initial_acceptance_threshold": syndicate_args["initial_acceptance_threshold"],
                                "acceptance_threshold_friction": syndicate_args["acceptance_threshold_friction"],
                                "reinsurance_limit": syndicate_args["reinsurance_limit"],
                                "non_proportional_reinsurance_level": insurance_reinsurance_level,
                                "capacity_target_decrement_threshold": syndicate_args["capacity_target_decrement_threshold"],
                                "capacity_target_increment_threshold": syndicate_args["capacity_target_increment_threshold"],
                                "capacity_target_decrement_factor": syndicate_args["capacity_target_decrement_factor"],
                                "capacity_target_increment_factor": syndicate_args["capacity_target_increment_factor"],
                                "interest_rate": syndicate_args["interest_rate"]})

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
        # Set up remaining list variables
        # Agent lists
        self.reinsurancefirms = []
        self.syndicates = []
        # Lists of agent weights
        self.syndicates_weights = {}
        self.reinsurancefirms_weights = {}
        # Cumulative variables for history and logging
        self.cumulative_bankruptcies = 0
        self.culumative_market_exits = 0
        self.cumulative_unrecovered_claims = 0.0
        self.cumulative_claims = 0.0
        # Lists for logging history
        self.logger = logger.logger(num_riskmodels = risk_args["num_riskmodels"],
                                    rc_event_schedule_initial = self.rc_event_schedule_initial,
                                    rc_event_damage_initial = self.rc_event_damage_initial)
        self.syndicate_models_counter = np.zeros(risk_args["num_categories"])
        self.reinsurancefirms_models_counter = np.zeros(risk_args["num_categories"])

        self.var_tail_prob = 0.02
        
        self.category_number = category_number
        self.init_average_exposure = init_average_exposure
        self.init_average_risk_factor = init_average_risk_factor
        self.init_profit_estimate = init_profit_estimate
        
        self.damage_distribution = [damage_distribution for _ in range(self.category_number)]
        self.damage_distribution_stack = [[] for _ in range(self.category_number)]
        self.reinsurance_contract_stack = [[] for _ in range(self.category_number)]
        self.inaccuracy = inaccuracy

        "num_riskmodels": 4, # Number of risk models in simulation
        "num_risks": 20000, # Number of risks
        "num_riskregions": 10, # Number of peril regions for the catastrophe
        "risk_limit": 10000000, # The maximum value of the risk
        "inaccuracy_riskmodels": 2,
        "riskmodel_margin_of_safety": 2,
        "value_at_risk_tail_probability": 0.005,
        "catastrophe_time_mean_separation": 100/3.,
        "lambda_attritional_loss": 0.1, # Lambda value for the Poisson distribution for the number of attritional claims generated per year
        "cov_attritional_loss": 1, # Coefficient of variation for the gamma distribution which generates the severity of attritional claim event
        "mu_attritional_loss": 3000000, # Mean of the gamma distribution which generates the severity of attritional claim events
        "lambda_catastrophe": 0.05, # Lambda value for the Poisson distribution for the number of catastrophe claims generated per year
        "pareto_shape": 5, # Shape parameter of the Pareto distribution which generates the severity of catastrophe claim events
        "minimum_catastrophe_damage": 0.25, # Minimum value for an event to be considered a catastrophe, fraction of the risk limit
        "var_em_exceedance_probability": 0.05, # The tail probability  used in the VaR calculations
        "var_em_safety_factor": 1, # Scaling safety factor applied to the VaR value, larger values employ more conservative exposure management
        "risk_factor_lower_bound": 0.4,
        "risk_factor_upper_bound": 0.6

    def getPPF(self, categ_id, tailSize):
        """
        Get quantile function of the damage distribution (value at risk) by category

        Parameters
        ----------
        categ_id: int
            Category indentifier
        tailSize: float 0<=tailSize<=1

        Return
        ------
        value-at-risk
        """

        return self.damage_distribution[categ_id].ppf(1-tailSize)

    def get_categ_risks(self, risks, categ_id):
        """
        Choose risk with categ_id
        """
        categ_risks = []
        for risk in risks:
            if risk["category"] == categ_id:
                categ_risks.append(risk)

        return categ_risks

    def compute_expectation(self, categ_risks, categ_id):
        exposures = []
        risk_factors = []
        runtimes = []

        for risk in categ_risks:
            exposures.append(risk["value"]-risk["deductible"])
            risk_factors.append(risk["risk_factor"])
            runtimes.append(risk["runtime"])

        average_exposure = np.mean(exposures)
        average_risk_factor = self.inaccuracy[categ_id] * np.mean(risk_factors)
        mean_runtime = np.mean(runtimes)

        if self.expire_immediately:
            incr_expected_profits = (self.norm_premium - (1 - scipy.stats.poisson(1/self.cat_separation_distribution.mean() * mean_runtime).pmf(0)) * self.damage_distribution[catteg_id].mean() * average_risk_factor) * average_exposure * len(categ_risks)
        else:
            incr_expected_profits = -1

        return average_risk_factor, average_exposure, incr_expected_profits

    def evaluate_proportional(self, risks, cash):
        """
        Calculate the cash left in each risk category
        """
        assert len(cash) == self.category_number

        # Prepare variables
        acceptable_by_category = []
        remaining_acceptable_by_category = []
        cash_left_by_category = np.copy(cash)
        expected_profits = 0
        necessary_liquidity = 0

        var_per_risk_per_categ = np.zeros(self.category_number)

        # Compute acceptable risks by category
        for categ_id in range(self.category_number):
            categ_risks = self.get_categ_risks(risks = risks, categ_id = categ_id)

            if len(categ_risks) > 0:
                average_risk_factor, average_exposure, incr_expected_profits = self.compute_expectation(categ_risks, categ_id)
            else:
                average_risk_factor = self.init_average_risk_factor
                average_exposure = self.init_average_exposure
                incr_expected_profits = -1

            expected_profits += incr_expected_profits

            # Compute value at risk
            var_per_risk = self.getPPF(categ_id, self.var_tail_prob) * average_risk_factor * average_exposure * self.margin_of_safety

            # Record liquidity requirement and apply margin of safety for liquidty requirement
            necessary_liquidity += var_per_risk * self.margin_of_safety * len(categ_risks)

            if isleconfig.verbose:
                print(self.inaccuracy)
                print("RiskModel:", var_per_risk, "=PPF(0.02)*", average_risk_factor, "*", average_exposure," vs.cash:", cash[categ_id], "Total risk in category:", var_per_ris*len(categ_risks))
            #print("RiskModel:", self.getPPF(categ_id, tailSize=0.05) * average_risk_factor * average_exposure, "=PPF(0.05)*", average_risk_factor, "*", average_exposure," vs.cash:", cash[categ_id], "Total risk in category:", self.getPPF(categ_id, tailSize=0.05) * average_risk_factor * average_exposure * len(categ_risks))
            #print("RiskModel:", self.getPPF(categ_id, tailSize=0.1) * average_risk_factor * average_exposure, "=PPF(0.1)*", average_risk_factor, "*", average_exposure," vs.cash:", cash[categ_id], "Total risk in category:", self.getPPF(categ_id, tailSize=0.1) * average_risk_factor * average_exposure * len(categ_risks))
            #print("RiskModel:", self.getPPF(categ_id, tailSize=0.25) * average_risk_factor * average_exposure, "=PPF(0.25)*", average_risk_factor, "*", average_exposure," vs.cash:", cash[categ_id], "Total risk in category:", self.getPPF(categ_id, tailSize=0.25) * average_risk_factor * average_exposure * len(categ_risks))
            #print("RiskModel:", self.getPPF(categ_id, tailSize=0.5) * average_risk_factor * average_exposure, "=PPF(0.5)*", average_risk_factor, "*", average_exposure," vs.cash:", cash[categ_id], "Total risk in category:", self.getPPF(categ_id, tailSize=0.5) * average_risk_factor * average_exposure * len(categ_risks))
            try:
                acceptable = int(math.floor(cash[categ_id]/var_per_risk))
                remaining = acceptable - len(categ_risks)
                cash_left = cash[categ_id] - len(categ_risks) * var_per_risk
            except:
                print(sys.exc_info())
                pdb.set_trace()
            acceptable_by_category.append(acceptable)
            remaining_acceptable_by_category.append(remaining)
            cash_left_by_category[categ_id] = cash_left
            var_per_risk_per_categ[categ_id] = var_per_risk

        if expected_profits < 0:
            expected_profits = None
        else:
            if necessary_liquidity == 0:
                assert expected_profits == 0
                expected_profits = self.init_profit_estimate * cash[0]
            else:
                expected_profits /= necessary_liquidity

        max_cash_by_categ = max(cash_left_by_category)
        floored_cash_by_categ = cash_left_by_category.copy()
        floored_cash_by_categ[floored_cash_by_category<0] = 0
        remaining_acceptable_by_category_old = remianing_acceptable_by_category.copy()
        for categ_id in range(self.category_number):
            remianing_acceptable_by_category[categ_id] = math.floor(remaining_acceptable_by_category[categ_id] * pow(floored_cash_by_categ[categ_id]/max_cash_by_categ, 5))
        if isleconfig.verbose:
            print("Riskmodel returns:", expected_profits, remianing_acceptable_by_category)

        return expected_profits, remianing_acceptable_by_category, cash_left_by_category, var_per_risk_per_categ

    def evaluate_excess_of_loss(self, risks, cash, offered_risk=None):
        cash_left_by_categ = np.copy(cash)
        assert len(cash_left_by_categ) == self.category_number

        # Prepare variables
        additional_required = np.zeros(self.category_number)
        additional_var_per_categ = np.zeros(self.category_number)

        # Values at risk and liquidity requirements by category
        for categ_id in range(self.category_number):
            categ_risks = self.get_categ_risks(risks, categ_id)
            percentage_value_at_risk = self.getPPF(categ_id, tailSize=self.var_tail_prob)

            # Compute liquidity requirements from existing contracts
            for risk in categ_risks:
                expected_damage = percentage_value_at_risk * risk["value"] * risk["risk_factor"] * self.inaccuravy[categ_id]
                expected_claim = min(expected_damage, risk["excess"]) - risk["deductible"]

                # Record liquidity requirement and apply margin of safety for liquidity requirement
                cash_left_by_categ[categ_id] -= expected_claim * self.margin_of_safety

            # Compute additional liquidity requirements from newly offered contract
            if (offered_risk is not None) and (offered_risk.get("category") == categ_id):
                expected_damage_fraction = percentage_value_at_risk * offered_risk["risk_factor"] * self.inaccuracy[categ_id]
                expected_claim_fraction = min(expected_damage_fraction, offered_risk["excess_fraction"] - offered_risk["deductible_fraction"])
                expected_claim_total = expected_claim_fraction * offered_risk["value"]

                # Record liquidity requirement and apply margin of safety for liquidity requirement
                additional_required[categ_id] += expected_claim_total * self.,margin_of_safety
                additional_var_|per_categ[categ_id] += expected_claim_total

        # Additional value at risk should only occur in one category
        assert sum(additional_var_per_categ>0)<=1
        var_this_risk = max(additional_var_per_categ)

        return cash_left_by_categ, additional_required, var_this_risk

    def evaluate(self, risks, cash, offered_risk = None):
        # Ensure that any risk to be considred supplied directly as argument is non-proportional/excess-of-loss
        assert (offered_risk is None) or offered_risk.get("insurancetype") == "excess-of-loss"

        # Construct cash_left_by_categ as a sequence, defining remaining liquidity by category
        if not isinstance(cash, (np.ndarray, list)):
            cash_left_by_categ = np.ones(self.category_number) * cash
        else:
            cash_left_by_categ = np.copy(cash)

        assert len(cash_left_by_categ) == self.category_number

        # Sort current contracts
        el_risks = [risk for risk in risks if risk["insurancetype"] == 'excess_of_loss']
        risks = [risk for risk in risks if risk["insurancetype"] == 'proportional']

        # Compute liquidity requirements and acceptable risks from existing contract
        if (offered_risk is not None) or (len(el_risks) > 0):
            cash_left_by_categ, additional_required, var_this_risk = self.evaluate_excess_of_loss(el_risks, cash_left_by_categ, offered_risk)
        if (offered_risk is None) or (len(risks) > 0):
            expected_profits_proportional, remaining_acceptable_by_categ, cash_left_by_categ, var_per_risk_per_categ = self.evaluate_proportional(risk, cash_left_by_categ)
        if offered_risk is None:
            # Return numbers of remaining acceptable risks by category
            return expected_profits_proportional, remaining_acceptable_by_categ, cash_left_by_categ, var_per_risk_per_categ, min(cash_left_by_categ)
        else:
            # Return boolean value whether the offered excess_of_loss risk can be accepted
            return (cash_left_by_categ - additional_required > 0).all(), cash_left_by_categ, var_this_risk, min(cash_left_by_categ)

    def add_reinsurance(self, categ_id, excess_fraction, deductible_fraction, contract):
        self.damage_distribution_stack[categ_id].append(self.damage_distribution[categ_id])
        self.reinsurance_contract_stack[categ_id].append(contract)
        self.damage_distribution[categ_id] = ReinsuranceDistWrapper(lower_bound=deductible_fraction, upper_bound=excess_fraction, dist=self.damage_distribution[categ_id])

    def delete_reinsurance(self, categ_id, excess_fraction, deductible_fraction, contract):
        assert self.reinsurance_contract_stack[categ_id][-1] == contract
        self.reinsurance_contract_stack[categ_id].pop()
        self.damage_distribution[categ_id] = self.damage_distribution_stack[categ_id].pop()

    def one_risk_model(self, ):

        return

    def four_risk_model(self, ):

        return
