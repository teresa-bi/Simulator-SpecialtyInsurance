import math
import numpy as np
import sys, pdb
import scipy.stats

class RiskModel:
    """
    The risk model adopted by syndicates or reinsurance firms to cope with catestrophes
    """
    def __init__(self, damage_distribution, expire_immediately, catastrophe_separation_distribution, norm_premium, category_number, init_average_exposure, 
                 init_average_risk_factor, init_profit_estimate, margin_of_safety, var_tail_prob, inaccuracy):
        self.expire_immediately = expire_immediately
        self.catastrophe_separation_distribution = catastrophe_separation_distribution
        self.norm_premium = norm_premium
        self.category_number = category_number
        self.init_average_exposure = init_average_exposure
        self.init_average_risk_factor = init_average_risk_factor
        self.init_profit_estimate = init_profit_estimate
        self.margin_of_safety = margin_of_safety
        self.var_tail_prob = var_tail_prob
        self.damage_distribution = [damage_distribution for _ in range(self.category_number)]
        self.damage_distribution_stack = [[] for _ in range(self.category_number)] 
        self.inaccuracy = inaccuracy

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
            if risk["risk_category"] == categ_id:
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
                expected_damage = percentage_value_at_risk * risk["risk_value"] * risk["risk_factor"] * self.inaccuracy[categ_id]
                expected_claim = min(expected_damage, 1.0) - 0.0

                # Record liquidity requirement and apply margin of safety for liquidity requirement
                cash_left_by_categ[categ_id] -= expected_claim * self.margin_of_safety

            # Compute additional liquidity requirements from newly offered contract
            if (offered_risk is not None) and (offered_risk.risk_category == categ_id):
                expected_damage_fraction = percentage_value_at_risk * offered_risk.risk_factor * self.inaccuracy[categ_id]
                expected_claim_fraction = min(expected_damage_fraction, 1.0 - 0.0)
                expected_claim_total = expected_claim_fraction * offered_risk.risk_value

                # Record liquidity requirement and apply margin of safety for liquidity requirement
                additional_required[categ_id] += expected_claim_total * self.margin_of_safety
                additional_var_per_categ[categ_id] += expected_claim_total

        # Additional value at risk should only occur in one category
        assert sum(additional_var_per_categ>0)<=1
        var_this_risk = max(additional_var_per_categ)

        return cash_left_by_categ, additional_required, var_this_risk

    def evaluate(self, risks, cash, offered_risk=None):

        # construct cash_left_by_categ as a sequence, defining remaining liquidity by category
        if not isinstance(cash, (np.ndarray, list)):
            cash_left_by_categ = np.ones(self.category_number) * cash
        else:
            cash_left_by_categ = np.copy(cash)
        assert len(cash_left_by_categ) == self.category_number

        # compute liquidity requirements and acceptable risks from existing contract
        if (offered_risk is not None) or (len(risks) > 0):
            cash_left_by_categ, additional_required, var_this_risk = self.evaluate_excess_of_loss(risks, cash_left_by_categ, offered_risk)
        
            # return boolean value whether the offered excess_of_loss risk can be accepted
            return (cash_left_by_categ - additional_required > 0).all(), cash_left_by_categ, var_this_risk, min(cash_left_by_categ)
    
    def calculate_VaR(self, offered_risk):
        # Prepare variables       
        var_per_risk = self.getPPF(categ_id=offered_risk.get("risk_category"), tailSize=self.var_tail_prob) * self.init_average_risk_factor * self.init_average_exposure * self.margin_of_safety
        return var_per_risk[0]