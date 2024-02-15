"""
Contains all the capabilities of Syndicates
"""

import numpy as np
import scipy.stats
import copy
from insurance_contract import InsuranceContract
from reinsurancecontract impoert ReinsuranceContract
from environment.risk.risk_model import RiskModel
import sys, pdb
import uuid

# If not use abce
from genericagent import GenericAgent

class Syndicate:
    def __init__(self, syndicate_id, syndicate_args, num_risk_models, risk_model_configs):
        self.syndicate_id = syndicate_id
        self.initial_capital = syndicate_args['initial_capital']
        self.current_capital = syndicate_args['initial_capital']
        self.premium_internal_weight = syndicate_args['actuarial_pricing_internal_weight']
        self.interest_rate = syndicate_args['interest_rate_monthly']
        self.leader_list = {}  # Include risk_id, broker_id, line_size
        self.lead_line_size = syndicate_args['lead_line_size']
        self.follower_list = {}  # Include risk_id, broker_id, line_size
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

    def data(self):
        """
        Create a dictionary with key/value pairs representing the Syndicate data.

        Returns
        ----------
        dict
        """

        return {
            "syndicate_id": self.syndicate_id,
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "premium_internal_weight": self.premium_internal_weight,
            "interest_rate": self.interest_rate,
            "leader_list": self.leader_list,
            "follower_list": self.follower_list,
            "dividends_of_profit": self.dividends_of_profit,
            "loss_experiency_weight": self.loss_experiency_weight,
            "volatility_weight": self.volatility_weight,
            "underwriter_markup_recency_weight": self.underwriter_markup_recency_weight,
            "upper_premium_limit": self.upper_premium_limit,
            "lower_premium_limit": self.lower_premium_limit,
            "premium_reserve_ratio": self.premium_reserve_ratio,
            "minimum_capital_reserve_ratio": self.minimum_capital_reserve_ratio,
            "maximum_scaling_factor": self.maximum_scaling_factor,
            "market_entry_probability": self.market_entry_probability,
            "exit_capital_threshold": self.exit_capital_threshold,
            "exit_time_limit": self.exit_time_limit,
            "sensitivity_premium": self.sensitivity_premium,
            "initial_acceptance_threshold": self.initial_acceptance_threshold,
            "acceptance_threshold_friction": self.acceptance_threshold_friction
        }

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

    def update_status(self, ):
        self.update_capital
        self.update_underwrite_risk_regions

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
        """
        Total premium from brokers
        """
        
        premium = calculate_premium()

        return total_premium

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



def get_mean(x):
    return sum(x) / len(x)

def get_mean_std(x):
    m = get_mean(x)
    variance = sum((val - m)**2 for val in x)
    return m,np.sqrt(variance / len(x))

class MetaInsuranceOrg(GenericAgent):
    def __init__(self,simulation_parameters, agent_parameters):
        self.simulation = simulation_parameters['simulation']
        self.simulation_parameters = simulation_parameters
        self.contract_runtime_dist = scipy.stats.randin(simulation_parameters["mean_contract_runtime"] - simulation_parameters["contract_runtime_halfspread"], simulation_parameters["mean_contract_runtime"]+simulation_parameters["contract_runtime_halfspread"]+1)
        self.default_contract_payment_period = simulation_parameters["default_contract_payment_period"]
        self.id = agent_parameters['id']
        self.cash = agent_parameters['initial_cash']
        self.capacity_target = self.cash * 0.9
        self.capacity_target_decrement_threshold = agent_parameters['capacity_target_decrement_threshold']
        self.capacity_target_increment_threshold = agent_parameters['capacity_target_increment_threshold']
        self.capacity_target_decrement_factor = agent_parameters['capacity_target_decrement_factor']
        self.capacity_target_increment_factor = agent_parameters['capacity_target_increment_factor']
        self.excess_capital = self.cash
        self.premium = agent_parameters['norm_premium']
        self.profit_target = agent_parameters['profit_target']
        self.acceptance_threshold = agent_parameters['initial_acceptance_threshold']
        self.acceptance_threshold_friction = agent_parameters['initial_acceptance_threshold_friction']
        self.interest_rate = agent_parameters['interest_rate']
        self.reinsurance_limit = agent_parameters['reinsurance_limit']
        self.simulation_no_risk_categories = simulation_parameters["no_categories"]
        self.simulation_reinsurance_type = simulation_parameters["simulation_reinsurance_type"]
        self.dividend_share_of_profits = simulation_parameters["dividend_share_of_profits"]
        self.owner = self.simulation
        self.per_period_dividend = 0
        self.cash_last_periods = list(np.zeros(4,dtype=int)*self.cash)

        rm_config = agent_parameters["riskmodel_config"]

        margin_of_safety_correction = (rm_config["margin_of_safety"] + (simulation_parameters["no_riskmodels"] -1) * simulation_parameters["margin_increase"])

        self.riskmodel = RiskModel(damage_distribution = rm_config["damage_distribution"],
                                expire_immediately = rm_config["expire_immediately"],
                                cat_separation_distribtion = rm_config["cat_separation_distribtion"],
                                norm_premium = rm_config["norm_premium"],
                                category_number = rm_config["no_categories"],
                                init_average_exposure = rm_config["risk_value_mean"],
                                init_average_risk_factor = rm_config["risk_factor_mean"],
                                init_profit_estimate = rm_config["norm_profit_markup"],
                                margin_of_safety = margin_of_safety_correction,
                                var_tail_prob = rm_config["var_tail_prob"],
                                inaccuracy = rm_config["inaccuracy_by_categ"])

        self.category_reinsurance = [None for i in range(self.simulation_no_risk_categories)]

        if self.simulation_reinsurance_type == 'non-proportional':
            if agent_parameters['non-proportional_reinsurance_level'] is not None:
                self.np_reinsurance_deductible_fraction = agent_parameters['non-proportional_reinsurance_level']
            else:
                self.np_reinsurance_deductible_fraction = simulation_parameters["default_non-proportional_reinsurance_deductible"]
                self.np_reinsurance_excess_fraction = simulation_parameters["default_non-proportional_reinsurance_excess"]
                self.np_reinsurance_premium_share = simulation_parameters["default_non-proportional_reinsurance_premium_share"]

        self.obligations = []
        self.underwritten_contracts = []
        self.profits_losses = 0
        self.operational = true

        # Set up risk value estimate variables
        self.var_counter = 0           # Sum over risk model inaccuracies for all contracts
        self.var_counter_per_risk = 0    # Average risk model inaccuracy across contracts
        self.var_sum = 0           # Sum over initial vaR for all contracts
        self.counter_category = np.zeros(self.simulation_no_risk_categories)     # var_counter disaggregated by category
        self.var_category = np.zeros(self.simulation_no_risk_categories)
        self.naccep = []
        self.risk_kept = []
        self.reinrisks_kept = []
        self.balance_ratio = simulation_parameters['insurers_balance_ratio']
        self.recursion_limit = simulation_parameters["insurers_recursion_limit"]
        self.cash_left_by_categ = [self.cash for i in range(self.simulation_parameters["no_categories"])]
        self.market_permanency_counter = 0

    def iterate(self, time):
        # Obtain investments yield
        self.obtain_yield(time)

        # Realise due payments
        self.effect_payments(time)
        if isleconfig.verbose:
            print(time, ":", self.id, len(self.underwrittem_contracts), self.cash, self.operational)

        self.make_reinsurance_claims(time)

        # Mature contracts
        if isleconfig.verbose:
            print("Number of underwritten contracts", len(self.underwritten_contracts))
        maturing = [contract for contract in self.underwritten_contracts if contract.expiration <= time]
        for contract in maturing:
            self.underwritten_contracts.remove(contract)
            contract.mature(time)
        contracts_dissolved = len(maturing)

        # Effect payments from contracts
        [contract.check_payment_due(time) for contract in self.underwritten_contracts]

        if self.operational:
            """
            Request risks to be considered for underwriting in the next period and collect those for this period
            """
            new_risks = []
            if self.is_insurer:
                new_risks += self.simulation.solicit_insurance_requests(self.id, self.cash, self)
            if self.is_insurer:
                new_risks += self.simulation.solicit_reinsurance_requests(self.id, self.cash, self)
            contracts_offered = len(new_risks)

            if isleconfig.verbose and contracts_offered < 2 * contracts_dissolved:
                print("Something wrong: agent {0:d} receives too few new contracts {1:d} <= {2:d}".format(self.id, contracts_offered, 2*contracts_dissolved))

            new_nonproportional_risks = [risk for risk in new_risks if risk.get("insurancetype")=='excess-of-loss' and risk["owner"] is not self]

            new_risks = [risk for risk in new_risks if risk.get("insurancetype") in ['proportional', None] and risk["owner"] is not self]

            # Deal with non-proportionala risks first as they must evaluate each requet separatly, then wwith proportional ones
            [reinrisks_per_categ, number_reinrisks_categ] = self.risks_reinrisks_organizer(new_nonproportional_risks)

            for repetition in range(self.recursion_limit):
                former_reinrisks_per_categ = copy.copy(reinrisks_per_categ)
                [reinrisks_per_categ, not_accepted_reinrisks] = self.process_newrisks_reinsurer(reinrisks_per_categ, number_reinrisks_categ, time)
                if former_reinrisks_per_categ == reinrisks_per_categ:
                    break

            self.simulation.return_reinrisks(not_accepted_reinrisks)

            underwritten_risks = [{"value": contract.value, "category": contract.category, "risk_factor": contract.risk_factor, "deductible": contract.deductible, "excess": contract.excess, "insurancetype": contract.insurancetype, "run_time": contract_runtime} for contract in self.underwritten_contracts if contract.reinsurance_share != 1.0]

            """
            Obtain risk model evaluation (VaR) for underwriting decisions and for capacity specific decisions
            """
            expected_profit, acceptable_by_category, cash_left_by_categ, var_per_risk_per_categ, self.excess_capital = self.riskmodel.evaluate(underwritten_risks, self.cash)

            """
            Handle adjusting capacity target and capacity
            """
            max_var_by_categ = self.cash - self.excess_capital
            self.adjust_capacity_target(max_var_by_categ)
            actual_capacity = self.increase_capacity(time, max_var_by_categ)

            """
            Handle capital market interactions: capital history and dividends
            """
            self.cash_last_periods = [self.cash] + self.cash_last_periods[:3]
            self.adjust_dividends(time, actual_capacity)
            self.pay_dividends(time)

            """
            Make underwriting decisions, category-wise
            """
            growth_limit = max(50,2 * len(self.underwritten_contracts) + contracts_dissolved)
            if sum(acceptable_by_category) > growth_kimit:
                acceptable_by_category = np.asarray(acceptable_by_category).astype(np.double)
                acceptable_by_category = acceptable_by_category * growth_limit / sum(acceptable_by_category)
                acceptable_by_category = np.int64(np.round(acceptable_by_category))

                [risks_per_categ, number_risks_categ] = self.risks_reinrisks_organizer(new_risks)

                for repetition in range(self.recursion_limit):
                    # Find an efficient way to stop the recursion if there are no more risks to accept or if it is not accepting any more over several iterations
                    former_risks_per_categ = copy.copy(risks_per_categ)
                    # Process all the new risks in order to keep the portfolio as balanced as possible
                    [risks_per_categ, not_accepted_risks] = self.process_newrisks_insurer(risks_per_categ, number_risks_caateg, acceptable_by_category, var_per_risk_per_categ, cash_left_by_categ, time)
                    if former_risks_per_categ == risks_per_categ:
                        break

                # Return unacceptables
                self.simulation.return_risks(not_accepted_risks)

        self.market_permanency(time)
        self.toll_over(time)
        self.estimated_var()

    def enter_illiquidity(self, time):
        """
        Enter illiquidity method, this method is called when a firm does not have enough cash to pay all its obligations

        Parameters
        ----------
        time: int
            current time
        """

        self.enter_bankruptcy(time)

    def enter_bankruptcy(self, time):
        """
        Syndicate bankrupt and dissolves the firm through the method self.dissolve()

        Parameters
        ----------
        time: int
            current time
        """

        self.dissolve(time, 'record_bankruptcy')

    def market_exit(self, time):
        """
        Syndicate wants to leave the market because it feels that it has been underperforming for too many periods

        Parameters
        ----------
        time: int
            current time
        """

        due = [item for item in self.obligations]
        for obligation in due:
            self.pay(obligation)
        self.obligations = []
        self.dissolve(time, 'record_market_exit')

    def dissolve(self, time, record):
        """
        It dissolves all the contracts currently held (in self.underwritten_contracts), next all the cash currently available is transferred to insurancesimulation.py through an obligation in the next itertion. finally, the type of dissolution is recorded and the operational state is set to false, different class variables are reset
        Parameters
        ----------
        time: int
            current time
        record: string
            record_bankruptcy or record_market_exit
        """

        # Remove all risks after bankruptcy
        [contract.dissolve(time) for contract in self.underwrittten_contracts]

        self.simualtion.return_risks(self.risks_kept)
        self.risks_kept = []
        self.reinrisks_kept = []
        obligation = {"amount": self.cash, "recipient": self.simulation, "due_time": time, "purpose": "Dissolution"}
        # This must be the last obligation before the dissolution of the firm
        self.pay(obligation)
        # Excess capital is 0 after bakruptcy or market exit
        self.excess_capital = 0
        # Profits and losses are 0 after bankruptcy or market exit
        self.profits_losses = 0
        if self.operational:
            method_to_call = getattr(self.simulation, record)
            method_to_call()
        for category_reinsurance in self.category_reinsurance:
            if category_reinsurnce is not None:
                category_reinsurane.dissolve(time)
        self.operational = False

    def receive_obligation(self, amount, recipient, due_time, purpose):
        obligation = {"amount": amount, "recipient": recipient, "due_time": due_time, "purpose": purpose}
        self.obligations.append(obligation)

    def effect_payments(self, time):
        due = [item for item in self.obligations if item["due_time"]<=time]
        self.obligations = [item for item in self.obligations if item["due_time"]>time]
        sum_due = sum([item["amount"] for item in due])
        if sum_due > self.cash:
            self.obligations += due
            self.enter_illiquidity(time)
            self.simulation.record_unrecovered_claims(sum_due - self.cash)
        else:
            for obligation in due:
                self.pay(obligation)

    def pay(self, obligation):
        amount = obligation["amount"]
        recipient = obligation["recipient"]
        purpose = obligation["purpose"]
        if self.get_operational() and recipient.get_operational()
            self.cash -= amount
            if purpose is not "dividend":
                self.profits_losses -= amount
            recipient.receive(amount)

    def receive(self, amount):
        """
        Accept cash payment
        """

        self.cash += amount
        self.profits_losses += amount

    def pay_dividends(self, time):
        self.receive_obligation(self.per_period_dividend, self.owner, time, 'dividend')

    def obtian_yield(self, time):
        amount = self.cash * self.interest_rate
        self.simulation.receive_obligation(amount, self, time, 'yields')

    def increase_capacity(self):
        raise AttributeError("Method is not implemented in MetaInsuranceOrg, just in inheriting Insurance Firm instances")

    def get_cash(self):
        return self.cash

    def get_excess_capital(self):
        return self.excess_capital

    def logme(self):
        self.log('cash', self.cash)
        self.log('underwritten_contracts', self.underwritten_contracts)
        self.log('operaional', self.operational)

    def len_underwritten_contracts(self):
        return len(self.underwritten_contracts)

    def get_operational(self):
        return self.operational

    def get_profitslosses(self):
        return self.profits_losses

    def get_underwritten_contracts(self):
        return self.underwritten_contracts

    def get_pointer(self):
        return self

    def estimated_var(self):
        self.counter_category = np.zeros(self.simulation_no_risk_categories)
        self.var_category = np.zeros(self.simulation_no_risk_categories)

        self.var_counter = 0
        self.var_counter_per_risk = 0
        self.var_sum = 0

        if self.operational:
            for contract in self.underwritten_contracts:
                self.counter_category[contract.category] = self.counter_category[contract.category] + 1
                self.var_category[contract.category] = self.var_category[contract.category] + contract.initial_VaR

            for category in range(len(self.counter_category)):
                self.var_counter = self.var_counter + self.counter_category[category] * self.riskmodel.inaccuracy[category]
                self.var_sum = self.var_sum + self.var_category[category]

            if not sum(self.counter_category) == 0:
                self.var_counter_per_risk = self.var_counter / sum(self.counter_category)
            else:
                self.var_counter_per_risk = 0

    def increase_capacity(self, time):
        assert False, "Method not implemented, should be implemented in inheriting classses"

    def adjust_dividend(self, time):
        assert False, "Method not implemented, should be implemented in inheriting classses"

    def adjust_capacity_target(self, time):
        assert False, "Method not implemented, should be implemented in inheriting classses"

    def risk_reinrisks_organizer(self, new_risks):
        """
        Organize the new risks received by the insurer or reinsurer
        """
        # Organize the new risks received by the insurer (or reinsurer) by category
        risks_per_categ = [[] for x in range(self.simulation_parameters["no_categories"])]
        # Count the new risks received by the insurer (or reinsurer) by category
        number_risks_categ = [[] for x in range(self.simualtion_parameters["no_categories"])]

        for categ_id in range(self.simulation_parameters["no_categories"]):
            risks_per_categ[categ_id] = [risk for risk in new_risks if risk["category"] == categ_id]
            number_risks_categ[categ_id] = len(risks_per_categ[categ_id])

        return risks_per_categ, number_risks_categ

    def balanced_portfolio(self, risk, cash_left_by_categ, var_per_risk):
        """
        Decide whether the portfolio is balanced enough to accept a new risk or not, if it is balanced enough return True otherwise Flase
        """
        # Return the cash available per category independently the risk is accepted or not
        cash_reserved_by_categ = self.cash - cash_left_by_categ
        _, std_pre = get_mean_std(cash_reserved_by_categ)

        if risk.get("insurancetype") == 'excess-of-loss':
            percentage_value_at_risk = self.riskmodel.getPPF(categ_id=risk["category"], tailSize=self.riskmodel.var_tail_prob)
            expected_damage = percentage_value_at_risk * risk["value"] * risk["risk_factor"] * self.riskmodel.inaccuracy[risk["category"]]
            expected_claim = min(expected_damage, risk["value"] * risk["excess_fraction"]) - risk["value"] * risk["deductible_fraction"]

            # Record liquidity requirement and apply margin of safety for liquidity requirement
            # Compute how cash reserved by category would change if the new reinsurance risk was accepted
            cash_reserved_by_categ_store[risk["category"]] += expected_claim * self.riskmodel.margin_of_safety

        else:
            cash_reserved_by_categ_store[risk["category"]] += var_per_risk[risk["category"]]

            mean, std_post = get_mean_std(cash_reserved_by_categ_store)

            total_cash_reserved_by_categ_post = sum(cash_reserved_by_categ_store)

            if (std_post * total_cash_reserved_by_categ_post/self.cash) <= (self.balance_ratio * mean) or std_post < std_pre
                # The new risk is accepted if the standard deviation is reduced or the cash reserved by category is very well balanced
                for i in range(len(cash_left_by_categ)):
                    cash_left_by_categ[i] = self.cash - cash_reserved_by_categ_store[i]
                return True, cash_left_by_categ
            else:
                for i in range(len(cash_left_by_categ)):
                    cash_left_by_categ[i] = self.cash - cash_reserved_by_categ[i]
                return Flase, cash_left_by_categ

    def process_newrisks_reinsurer(self, reintisks_per_categ, number_reinrisks_categ, time):
        """
        Process one by one reinrisks contained in reinrisks_per_categ
        """
        for iteration in range(max(number_reinrisks_categ)):
            for categ_id in range(self.simulation_parameters["no_categories"]):
                if iterion < number_reinrisks_categ[categ_id] and reinrisks_per_categ[categ_id][iterion] is not None:
                    risk_to_insure = reinrisks_per_categ[categ_id][iterion]
                    underwritten_risks = [{"value": contract.value,
                                           "category": contract.category,
                                           "risk_factor": caontract.risk_factor,
                                           "deductible": contract.deductible,
                                           "excess": contract.excess,
                                           "insurancetype": contract.insurancetype,
                                           "runtime_left": (contract.expiration - time)} for contract in self.underwritten_contracts if contract.insurancetype == "excess-of-loss"]
                    accept, cash_left_by_categ, var_this_risk, self.excess_capital = self.riskmodel.evaluate(underwritten_risks, self.cash, risk_to_insure)
                    if accept:
                        per_value_reinsurance_premium = self.np_reinsurance_premium_share * risk_to_insure["periodized_total_premium"] * risk_to_insure["runtime"] * (self.simulation.get_market_reinpremium()/self.simulation.get_market_premium()) / risk_to_insurer["value"]

                        [condition, cash_left_by_categ] = self.balance_portfolio(risk_to_insure, cash_left_by_categ, None)
                        if condition:
                            contract = ReinsuranceContract(self, risk_to_insure, time, per_value_reinsurance_premium, risk_to_insure["runtime"], self.default_contract_payment_period, expire_immediately = self.simulation_parameters["expire_immediately"], initial_VaR = var_this_risk, insurancetype = risk_to_insure["insurancetype"])
                            self.underwritten_contracts.append(contract)
                            self.cash_left_by_categ = cash_left_by_categ
                            reinrisks_per_categ[categ_id][iterion] = None

        not_accepted_reinrisks = []
        for categ_id in range(self.simulation_parameters["no_categories"]):
            for reinrisk in reinrisks_per_categ[categ_id]:
                if reinrisk is not None:
                    not_accepted_reinrisks.append(reinrisk)

        return reinrisks_per_categ, not_accepted_reinrisks

    def process_newrisks_insurer(self, risks_per_categ, number_risks_categ, acceptable_by_category, var_per_risk_per_categ, cash_left_by_categ, time):
        """
        Process one by one risk contained in risks_per_categ to decided whether they should be underwritten or not
        """
        for iteration in range(max(number_risks_categ)):
            for categ_id in range(len(acceptable_by_category)):
                if iteration < number_risks_categ[categ_id] and acceptable_by_category[categ_id] > 0 and risks_per_categ[categ_id][iteration] is not None:
                    risk_to_insure = risk_per_categ[categ_id][iteration]
                    if risk_to_insure.get("contract") is not None and risk_to_insure["contract"].expiration > time:
                        [condition, cash_left_by_categ] = self.balanced_portfolio(risk_to_insure, cash_left_by_categ, None)
                        if condition:
                            contract = ReinsuranceContract(self, risk_to_insure, time, self.simulation.get_reinsurance_market_premium(), risk_to_insure["expiration"] - time, self.default_contract-payment_period, expire_immediately=self.simulation_parameters["expire_immediately"])
                            self.underwritten_contracts.append(contract)
                            self.cash_left_by_categ = cash_left_by_categ
                            risks_per_categ[categ_id][iteration] = None
                    else:
                        [condition, cash_left_by_categ] = self.balanced_portfolio(risk_to_insurer, cash_left_by_categ, var_per_risk_per_categ)
                        if condition:
                            contract = InsuranceContract(self, risk_to_insure, time, self.simulation.get_market_premium(), _cached_rvs, self.default_contract_payment_period, expire_immediately = self.simualtion_parameters["expire_immediately"], initial_VaR = var_per_risk_per_categ[categ_id])
                            self.underwritten_contracts.append(contract)
                            self.cash_left_by_categ = cash_left_by_categ
                            risks_per_categ[categ_id][iteration] = None
                    acceptable_by_category[categ_id] -= 1

        not_accepted_risks = []
        for categ_id in range(len(acceptable_by_category)):
            for risk in risks_per_categ[categ_id]:
                if risk is not None:
                    not_accepted_risks.append(risk)

        return risks_per_categ, not_accepted_risks

    def market_permanency(self, time):
        """
        Decide whether the insurer or reinsurer stays in the market. If it has very few risks underwritten or too much cash left for too long it will leave the market
        """
        if not self.simualtion_parameters["market_permanency_off"]:
            cash_left_by_categ = np.asarry(self.cash_left_by_categ)
            avg_cash_left = get_mean(cash_left_by_categ)
            if self.cash < self.simualtion_parameters["cash_permanency_limit"]:
                # If their cash is soo low that they cannot underwrite anything and leave the market
                self.market_exit(time)
            else:
                if self.is_insurer:
                    if len(self.underwritten_contracts) < self.simulation_parameters["insurance_permanency_contracts_limit"]or ave_cash_left/self.cash > self.simulation_parameters["insurance_permanency_ratio_limit"]:
                        self.market_permanency_counter += 1
                    else:
                        self.market_permanency_counter = 0

                    if self.market_permanency_counter >= self.simulation_parameters["insurance_permanency_time_constraint"]:
                        self.market_exit(time)
                if self.is_reinsurer:
                    if len(self.underwritten_contracts) < self.simulation_parameters["reinsurance_permanency_contracts_limit"] or avg_cash_left/self.cash > self.simulation_parameters["reinsurance_permanency_ratio_limit"]:
                        # Reinsurers leave the market if they have contracts under the limit or an excess capital over the limit for too long
                        self.market_permanency_counter += 1
                    else:
                        self.market_permanency_counter = 0

                    if self.market_permanency_counter >= self.simulaltion_parameters["reinsurance_permanency_time_constraint"]:
                        self.market_exit(time)

    def register_claim(self, claim):
        """
        Record in insurancesimulation.py every claim made
        """
        self.simulation.record_claims(claim)

    def reset_pl(self):
        """
        Reset the profits and losses variable of each firm at the beginning of every iteration
        """
        self.profits_losses = 0

    def roll_over(self, time):
        """
        Roll over the insurance and reinsurance contracts expiring in the next iteration
        """
        maturing_next = [contract for contract in self.underwritten_contracts if contract.expiration == time + 1]

        if self.is_insurer is True:
            for contract in maturing_next:
                contract.roll_over_flag = 1
                if np.random.uniform(0,1,1) > self.simulation_parameters["insurance_retention"]:
                    self.simulation.return_risks([contract.risk_data])
                else:
                    self.risks_kept.append(contract.risk_data)

        if self.is_reinsurer is True:
            for reincontract in maturing_next:
                if reincontract.property_holder.operational:
                    reincontract.roll_over_flag = 1
                    reinrisk = reincontract.property_holder.create_reinrisk(time, reincontract.category)
                    if np.random.uniform(0,1,1) < self.simulation_parameters["reinsurance_retention"]:
                        if reinrisk is not None:
                            self.reinrisks_kept.append(reinrisk)


     