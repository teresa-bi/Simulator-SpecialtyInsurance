"""
Contains all the capabilities of Syndicates
"""

import numpy as np
import scipy.stats
import copy
from environment.risk_model import RiskModel
import sys, pdb
import uuid

def get_mean(x):
    return sum(x) / len(x)

def get_mean_std(x):
    m = get_mean(x)
    variance = sum((val - m)**2 for val in x)
    return m,np.sqrt(variance / len(x))

class Syndicate:
    """
    Instance for syndicate
    """
    def __init__(self, syndicate_id, syndicate_args, num_risk_models, sim_args, risk_model_configs):
        self.syndicate_id = syndicate_id
        self.initial_capital = syndicate_args['initial_capital']
        self.current_capital = syndicate_args['initial_capital']
        self.current_capital_category = [syndicate_args['initial_capital'] / risk_model_configs[0]["num_categories"] for x in range(risk_model_configs[0]["num_categories"])]
        self.acceptance_threshold = syndicate_args['initial_acceptance_threshold']
        self.acceptance_threshold_friction = syndicate_args['initial_acceptance_threshold_friction']
        self.reinsurance_limit = syndicate_args["reinsurance_limit"]
        self.non_proportional_reinsurance_level = syndicate_args["default_non_proportional_reinsurance_deductible"]
        self.capacity_target_decrement_threshold = syndicate_args["capacity_target_decrement_threshold"]
        self.capacity_target_increment_threshold = syndicate_args["capacity_target_increment_threshold"]
        self.capacity_target_decrement_factor = syndicate_args["capacity_target_decrement_factor"]
        self.capacity_target_increment_factor = syndicate_args["capacity_target_increment_factor"]
        self.interest_rate = syndicate_args['interest_rate']
        self.dividend_share_of_profits = syndicate_args["dividend_share_of_profits"]

        self.contract_runtime_dist = scipy.stats.randint(sim_args["mean_contract_runtime"] - sim_args["contract_runtime_halfspread"], sim_args["mean_contract_runtime"] + sim_args["contract_runtime_halfspread"]+1)
        self.default_contract_payment_period = sim_args["default_contract_payment_period"]
        self.simulation_reinsurance_type = sim_args["simulation_reinsurance_type"]
        self.capacity_target = self.current_capital * 0.9
        self.capacity_target_decrement_threshold = self.capacity_target_decrement_threshold
        self.capacity_target_increment_threshold = self.capacity_target_increment_threshold
        self.capacity_target_decrement_factor = self.capacity_target_decrement_factor
        self.capacity_target_increment_factor = self.capacity_target_increment_factor

        self.riskmodel_config = risk_model_configs[int(syndicate_id) % len(risk_model_configs)]
        self.premium = self.riskmodel_config["norm_premium"]
        self.profit_target = self.riskmodel_config["norm_profit_markup"]
        self.excess_capital = self.current_capital
        self.num_risk_categories = self.riskmodel_config["num_categories"]
        
        self.per_period_dividend = 0
        self.capital_last_periods = list(np.zeros(4,dtype=int)*self.current_capital)

        self.premium_internal_weight = syndicate_args['actuarial_pricing_internal_weight']
        self.play_leader_in_contracts = []  # Include risk_id, broker_id, line_size
        self.lead_line_size = syndicate_args['lead_line_size']
        self.play_follower_in_contracts = []  # Include risk_id, broker_id, line_size
        self.follow_line_size = syndicate_args['follow_line_size']
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
        self.premium_sensitivity = syndicate_args['premium_sensitivity']

        self.status = False   # status True means active, status False means exit (no contract or bankruptcy, at the begining the status is 0 because no syndicate joining the market

        self.current_hold_contracts = []
        # Include paid status, True or False
        self.paid_claim = []

        margin_of_safety_correction = (self.riskmodel_config["margin_of_safety"] + (num_risk_models - 1) * sim_args["margin_increase"])

        self.riskmodel = RiskModel(damage_distribution = self.riskmodel_config["damage_distribution"],
                                expire_immediately = self.riskmodel_config["expire_immediately"],
                                catastrophe_separation_distribution = self.riskmodel_config["catastrophe_separation_distribution"],
                                norm_premium = self.riskmodel_config["norm_premium"],
                                category_number = self.riskmodel_config["num_categories"],
                                init_average_exposure = self.riskmodel_config["risk_value_mean"],
                                init_average_risk_factor = self.riskmodel_config["risk_factor_mean"],
                                init_profit_estimate = self.riskmodel_config["norm_profit_markup"],
                                margin_of_safety = margin_of_safety_correction,
                                var_tail_prob = self.riskmodel_config["var_tail_prob"],
                                inaccuracy = self.riskmodel_config["inaccuracy_by_categ"])

        self.category_reinsurance = [None for i in range(self.num_risk_categories)]

        if self.simulation_reinsurance_type == 'non-proportional':
            if self.non_proportional_reinsurance_level is not None:
                self.np_reinsurance_deductible_fraction = self.non_proportional_reinsurance_level
            else:
                self.np_reinsurance_deductible_fraction = syndicate_args["default_non-proportional_reinsurance_deductible"]
                self.np_reinsurance_excess_fraction = syndicate_args["default_non-proportional_reinsurance_excess"]
                self.np_reinsurance_premium_share = syndicate_args["default_non-proportional_reinsurance_premium_share"]

        self.obligations = []
        self.underwritten_contracts = []
        self.profits_losses = 0

        # Set up risk value estimate variables
        self.var_counter = 0           # Sum over risk model inaccuracies for all contracts
        self.var_counter_per_risk = 0    # Average risk model inaccuracy across contracts
        self.var_sum = 0           # Sum over initial vaR for all contracts
        self.counter_category = np.zeros(self.num_risk_categories)     # var_counter disaggregated by category
        self.var_category = np.zeros(self.num_risk_categories)
        self.naccep = []
        self.risk_kept = []
        self.reinrisks_kept = []
        self.balance_ratio = syndicate_args['insurers_balance_ratio']
        self.recursion_limit = syndicate_args["insurers_recursion_limit"]
        self.capital_left_by_categ = [self.current_capital for i in range(self.num_risk_categories)]
        self.market_permanency_counter = 0
        self.received_risk_list = []

    def received_risk(self, risk_id, broker_id, start_time):
        """
        After broker send risk to the market, all the active syndicate receive risks and update their received_risk list
        """
        self.received_risk_list.append({"risk_id": risk_id,
                                  "broker_id": broker_id,
                                  "start_time": start_time})

    def add_leader(self, risks, line_size, premium):
        self.play_leader_in_contracts.append({"risk_id": risks.get("risk_id"),
                                    "broker_id": risks.get("broker_id"),
                                    "line_size": line_size,
                                    "premium": premium})

    def add_follower(self, risks, line_size, premium):
        self.play_follower_in_contracts.append({"risk_id": risks.get("risk_id"),
                                    "broker_id": risks.get("broker_id"),
                                    "line_size": line_size,
                                    "premium": premium})

    def add_contract(self, risks, broker_id, premium):
        """
        Add new contract to the current_hold_contracts list
        """
        self.current_hold_contracts.append({"risk_id": risks.get("risk_id"),
                                    "broker_id": broker_id,
                                    "risk_start_time": risks.get("risk_start_time"),
                                    "risk_factor": risks.get("risk_factor"),
                                    "risk_category": risks.get("risk_category"),
                                    "risk_value": risks.get("risk_value"),
                                    "syndicate_id": self.syndicate_id,
                                    "premium": premium,
                                    "risk_end_time": risks.get("risk_start_time")+365,
                                    "pay": False})
        for i in range(len(self.current_capital_category)):
            if i == risks.get("risk_category"):
                self.current_capital_category[i] -= risks.get("risk_value")

    def pay_claim(self, risk_id, broker_id, category_id, claim_value):
        if self.current_capital >= claim_value:
            self.current_capital -= claim_value
            for i in range(len(self.current_hold_contracts)):
                if (self.current_hold_contracts[i]["risk_id"] == risk_id) and (self.current_hold_contracts[i]["broker_id"] == broker_id): 
                    self.current_hold_contracts[i]["pay"] == True
        else:
            self.current_capital -= claim_value
        

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
            "play_leader_in_contracts": self.play_leader_in_contracts,
            "play_follower_in_contracts": self.play_follower_in_contracts,
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
            "premium_sensitivity": self.premium_sensitivity,
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

    

    def obtain_yield(self, time):
        amount = self.current_capital * self.interest_rate             # TODO: agent should not award her own interest. This interest rate should be taken from self.simulation with a getter method
        self.simulation.receive_obligation(amount, self, time, 'yields')


    def enter_illiquidity(self, time):
        """
        Enter illiquidity method, this method is called when a firm does not have enough current_capital to pay all its obligations

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
        It dissolves all the contracts currently held (in self.underwritten_contracts), next all the current_capital currently available is transferred to insurancesimulation.py through an obligation in the next itertion. finally, the type of dissolution is recorded and the status is set to false, different class variables are reset
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
        obligation = {"amount": self.current_capital, "recipient": self.simulation, "due_time": time, "purpose": "Dissolution"}
        # This must be the last obligation before the dissolution of the firm
        self.pay(obligation)
        # Excess capital is 0 after bakruptcy or market exit
        self.excess_capital = 0
        # Profits and losses are 0 after bankruptcy or market exit
        self.profits_losses = 0
        if self.status:
            method_to_call = getattr(self.simulation, record)
            method_to_call()
        for category_reinsurance in self.category_reinsurance:
            if category_reinsurnce is not None:
                category_reinsurane.dissolve(time)
        self.status = False

    def pay_dividends(self, time):
        self.receive_obligation(self.per_period_dividend, self.owner, time, 'dividend')

    def obtian_yield(self, time):
        amount = self.current_capital * self.interest_rate
        self.simulation.receive_obligation(amount, self, time, 'yields')

    def increase_capacity(self):
        raise AttributeError("Method is not implemented in MetaInsuranceOrg, just in inheriting Insurance Firm instances")

    def balanced_portfolio(self, risk, capital_left_by_categ, var_per_risk):
        """
        Decide whether the portfolio is balanced enough to accept a new risk or not, if it is balanced enough return True otherwise Flase
        """
        # Return the cash available per category independently the risk is accepted or not
        capital_reserved_by_categ = self.current_capital - capital_left_by_categ
        _, std_pre = get_mean_std(capital_reserved_by_categ)

        if risk.get("insurancetype") == 'excess-of-loss':
            percentage_value_at_risk = self.riskmodel.getPPF(categ_id=risk["category"], tailSize=self.riskmodel.var_tail_prob)
            expected_damage = percentage_value_at_risk * risk["value"] * risk["risk_factor"] * self.riskmodel.inaccuracy[risk["category"]]
            expected_claim = min(expected_damage, risk["value"] * risk["excess_fraction"]) - risk["value"] * risk["deductible_fraction"]

            # Record liquidity requirement and apply margin of safety for liquidity requirement
            # Compute how capital reserved by category would change if the new reinsurance risk was accepted
            capital_reserved_by_categ_store[risk["category"]] += expected_claim * self.riskmodel.margin_of_safety

        else:
            capital_reserved_by_categ_store[risk["category"]] += var_per_risk[risk["category"]]

            mean, std_post = get_mean_std(capital_reserved_by_categ_store)

            total_capital_reserved_by_categ_post = sum(capital_reserved_by_categ_store)

            if (std_post * total_capital_reserved_by_categ_post/self.current_capital) <= (self.balance_ratio * mean) or std_post < std_pre:
                # The new risk is accepted if the standard deviation is reduced or the capital reserved by category is very well balanced
                for i in range(len(capital_left_by_categ)):
                    capital_left_by_categ[i] = self.current_capital - capital_reserved_by_categ_store[i]
                return True, capital_left_by_categ
            else:
                for i in range(len(capital_left_by_categ)):
                    capital_left_by_categ[i] = self.current_capital - capital_reserved_by_categ[i]
                return Flase, capital_left_by_categ

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
                    accept, capital_left_by_categ, var_this_risk, self.excess_capital = self.riskmodel.evaluate(underwritten_risks, self.current_capital, risk_to_insure)
                    if accept:
                        per_value_reinsurance_premium = self.np_reinsurance_premium_share * risk_to_insure["periodized_total_premium"] * risk_to_insure["runtime"] * (self.simulation.get_market_reinpremium()/self.simulation.get_market_premium()) / risk_to_insurer["value"]

                        [condition, capital_left_by_categ] = self.balance_portfolio(risk_to_insure, capital_left_by_categ, None)
                        if condition:
                            contract = ReinsuranceContract(self, risk_to_insure, time, per_value_reinsurance_premium, risk_to_insure["runtime"], self.default_contract_payment_period, expire_immediately = self.simulation_parameters["expire_immediately"], initial_VaR = var_this_risk, insurancetype = risk_to_insure["insurancetype"])
                            self.underwritten_contracts.append(contract)
                            self.capital_left_by_categ = capital_left_by_categ
                            reinrisks_per_categ[categ_id][iterion] = None

        not_accepted_reinrisks = []
        for categ_id in range(self.simulation_parameters["no_categories"]):
            for reinrisk in reinrisks_per_categ[categ_id]:
                if reinrisk is not None:
                    not_accepted_reinrisks.append(reinrisk)

        return reinrisks_per_categ, not_accepted_reinrisks

    def process_newrisks_insurer(self, risks_per_categ, number_risks_categ, acceptable_by_category, var_per_risk_per_categ, capital_left_by_categ, time):
        """
        Process one by one risk contained in risks_per_categ to decided whether they should be underwritten or not
        """
        for iteration in range(max(number_risks_categ)):
            for categ_id in range(len(acceptable_by_category)):
                if iteration < number_risks_categ[categ_id] and acceptable_by_category[categ_id] > 0 and risks_per_categ[categ_id][iteration] is not None:
                    risk_to_insure = risk_per_categ[categ_id][iteration]
                    if risk_to_insure.get("contract") is not None and risk_to_insure["contract"].expiration > time:
                        [condition, capital_left_by_categ] = self.balanced_portfolio(risk_to_insure, capital_left_by_categ, None)
                        if condition:
                            contract = ReinsuranceContract(self, risk_to_insure, time, self.simulation.get_reinsurance_market_premium(), risk_to_insure["expiration"] - time, self.default_contract-payment_period, expire_immediately=self.simulation_parameters["expire_immediately"])
                            self.underwritten_contracts.append(contract)
                            self.capital_left_by_categ = capital_left_by_categ
                            risks_per_categ[categ_id][iteration] = None
                    else:
                        [condition, capital_left_by_categ] = self.balanced_portfolio(risk_to_insurer, capital_left_by_categ, var_per_risk_per_categ)
                        if condition:
                            contract = InsuranceContract(self, risk_to_insure, time, self.simulation.get_market_premium(), _cached_rvs, self.default_contract_payment_period, expire_immediately = self.simualtion_parameters["expire_immediately"], initial_VaR = var_per_risk_per_categ[categ_id])
                            self.underwritten_contracts.append(contract)
                            self.capital_left_by_categ = capital_left_by_categ
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
        Decide whether the insurer or reinsurer stays in the market. If it has very few risks underwritten or too much capital left for too long it will leave the market
        """
        if not self.simualtion_parameters["market_permanency_off"]:
            capital_left_by_categ = np.asarry(self.capital_left_by_categ)
            avg_capital_left = get_mean(capital_left_by_categ)
            if self.current_capital < self.simualtion_parameters["capital_permanency_limit"]:
                # If their capital is soo low that they cannot underwrite anything and leave the market
                self.market_exit(time)
            else:
                if self.is_insurer:
                    if len(self.underwritten_contracts) < self.simulation_parameters["insurance_permanency_contracts_limit"]or ave_capital_left/self.current_capital > self.simulation_parameters["insurance_permanency_ratio_limit"]:
                        self.market_permanency_counter += 1
                    else:
                        self.market_permanency_counter = 0

                    if self.market_permanency_counter >= self.simulation_parameters["insurance_permanency_time_constraint"]:
                        self.market_exit(time)
                if self.is_reinsurer:
                    if len(self.underwritten_contracts) < self.simulation_parameters["reinsurance_permanency_contracts_limit"] or avg_capital_left/self.current_capital > self.simulation_parameters["reinsurance_permanency_ratio_limit"]:
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
                if reincontract.property_holder.status:
                    reincontract.roll_over_flag = 1
                    reinrisk = reincontract.property_holder.create_reinrisk(time, reincontract.category)
                    if np.random.uniform(0,1,1) < self.simulation_parameters["reinsurance_retention"]:
                        if reinrisk is not None:
                            self.reinrisks_kept.append(reinrisk)

    def update_capital(self, ):
        """
        Calculate the current capital after receiving premium from broker, paying each claim, paying premium to reinsurance firms, receiving payments from reinsurance firms, and paying dividends
        """
        self.initial_capital
        self.syndicate_id

        return current_capital

    def update_underwrite_risk_categories(self, ):
        """
        Calculate the capitals in each covered risk category
        """
        self.syndicate_id

    def update_status(self, actions_to_apply):
        self.update_capital
        self.update_underwrite_risk_categories

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


     