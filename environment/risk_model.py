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
            exposures.append(risk["risk_value"]-0.06)
            risk_factors.append(risk["risk_factor"])
            runtimes.append(60)

        average_exposure = np.mean(exposures)
        average_risk_factor = self.inaccuracy[categ_id] * np.mean(risk_factors)
        mean_runtime = np.mean(runtimes)

        if self.expire_immediately:
            incr_expected_profits = (self.norm_premium - (1 - scipy.stats.poisson(1/self.cat_separation_distribution.mean() * mean_runtime).pmf(0)) * self.damage_distribution[catteg_id].mean() * average_risk_factor) * average_exposure * len(categ_risks)
        else:
            incr_expected_profits = -1

        return average_risk_factor, average_exposure, incr_expected_profits
    
    def get_var(self, risk):
        # compute var of the new risk
        for categ_id in range(self.category_number):
            # compute number of acceptable risks of this category 
            if risk.risk_category == categ_id:
                exposure = risk.risk_value-0.06
                risk_factor = risk.risk_factor * self.inaccuracy[categ_id]
                # compute value at risk
                var_per_risk = self.getPPF(categ_id=categ_id, tailSize=self.var_tail_prob) * risk_factor * exposure * self.margin_of_safety
                return var_per_risk
    
    def evaluate_proportional(self, risks, cash):
        # prepare variables
        acceptable_by_category = []
        remaining_acceptable_by_category = []
        cash_left_by_category = np.copy(cash)
        expected_profits = 0
        necessary_liquidity = 0
        
        var_per_risk_per_categ = np.zeros(self.category_number)
        
        # compute acceptable risks by category
        for categ_id in range(self.category_number):
            # compute number of acceptable risks of this category 
            
            categ_risks = self.get_categ_risks(risks=risks, categ_id=categ_id)
            #categ_risks = [risk for risk in risks if risk["category"]==categ_id]
            
            if len(categ_risks) > 0:
                average_risk_factor, average_exposure, incr_expected_profits =  self.compute_expectation(categ_risks=categ_risks, categ_id=categ_id)
            else:
                average_risk_factor = self.init_average_risk_factor
                average_exposure = self.init_average_exposure

                incr_expected_profits = -1

            expected_profits += incr_expected_profits
            
            # compute value at risk
            var_per_risk = self.getPPF(categ_id=categ_id, tailSize=self.var_tail_prob) * average_risk_factor * average_exposure * self.margin_of_safety

            # record liquidity requirement and apply margin of safety for liquidity requirement
            necessary_liquidity += var_per_risk * self.margin_of_safety * len(categ_risks)
            
            try:
                acceptable = int(math.floor(cash[categ_id] / var_per_risk))
                remaining = acceptable - len(categ_risks)
                cash_left = cash[categ_id] - len(categ_risks) * var_per_risk
            except:
                print(sys.exc_info())
                pdb.set_trace()
            acceptable_by_category.append(acceptable)
            remaining_acceptable_by_category.append(remaining)
            cash_left_by_category[categ_id] = cash_left
            var_per_risk_per_categ[categ_id] = var_per_risk

        # TODO: expected profits should only be returned once the expire_immediately == False case is fixed; the else-clause conditional statement should then be raised to unconditional
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
        floored_cash_by_categ[floored_cash_by_categ < 0] = 0
        remaining_acceptable_by_category_old = remaining_acceptable_by_category.copy()
        for categ_id in range(self.category_number):
            remaining_acceptable_by_category[categ_id] = math.floor(
                    remaining_acceptable_by_category[categ_id] * pow(
                        floored_cash_by_categ[categ_id] / max_cash_by_categ, 5))
        return expected_profits, remaining_acceptable_by_category, cash_left_by_category, var_per_risk_per_categ

    def evaluate(self, risks, cash, offered_risk=None):

        # construct cash_left_by_categ as a sequence, defining remaining liquidity by category
        if not isinstance(cash, (np.ndarray, list)):
            cash_left_by_categ = np.ones(self.category_number) * cash
        else:
            cash_left_by_categ = np.copy(cash)
        assert len(cash_left_by_categ) == self.category_number

        # compute liquidity requirements and acceptable risks from existing contract
        expected_profits_proportional, remaining_acceptable_by_categ, cash_left_by_categ, var_per_risk_per_categ = self.evaluate_proportional(risks, cash_left_by_categ)
            
        return expected_profits_proportional, remaining_acceptable_by_categ, cash_left_by_categ, var_per_risk_per_categ, min(cash_left_by_categ)