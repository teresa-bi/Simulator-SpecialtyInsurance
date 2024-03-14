"""
Contains all the capabilities of Brokers
"""

import json
import random

class Broker:
    def __init__(self, broker_id, broker_args, num_risk_models, sim_args, risk_model_configs):
        self.broker_id = broker_id
        self.broker_lambda_risks = broker_args["lambda_risks_daily"]
        self.deductible = broker_args["decuctible"]
        self.num_risk_models = num_risk_models
        self.sim_time_span = sim_args["max_time"]
        self.risk_model_configs = risk_model_configs
        # Risks brought by broker for the whole time span
        self.risks = []
        # Risks be covered, underwritten
        self.underwritten_contracts = []
        # Risks not be covered
        self.not_underwritten_risks = []
        # Contracts affected by the catastrophe 
        self.affected_contracts = []
        # Claims not being paid
        self.not_paid_claims = []

    def generate_risks(self, risks):
        """
        Return risks brought by broker daily, according to poission distribution
        """
        random.seed(123)
        model_id = random.randint(0,self.num_risk_models-1)
        self.risks = []
        ######TODO: need to be fixed, how to generate poission distribution risks
        #num_risks_daily = self.broker_lambda_risks
        num_risks_daily = len(risks)
        for index in range(num_risks_daily):
            self.risks.append({"risk_id": index,
                               "broker_id": self.broker_id,
                               "risk_start_time": risks[index].get("risk_start_time"),
                               "risk_end_time": risks[index].get("risk_start_time")+12*30,
                               "risk_factor": self.risk_model_configs[model_id].get("risk_factor_mean"),
                               "risk_category": self.risk_model_configs[model_id].get("num_categories"),
                               "risk_value": self.risk_model_configs[model_id].get("risk_value_mean")})

        return self.risks

    def add_contract(self, risks, lead_syndicated_id, lead_line_size, follow_syndicates_id, follow_line_sizes, premium):
        """
        Add new contract to the current underwritten_contracts list
        """
        self.underwritten_contracts.append({"risk_id": risks.get("risk_id"),
                                    "broker_id": self.broker_id,
                                    "risk_start_time": risks.get("risk_start_time"),
                                    "risk_factor": risks.get("risk_factor"),
                                    "risk_category": risks.get("risk_category"),
                                    "risk_value": risks.get("risk_value"),
                                    "lead_syndicate_id": lead_syndicated_id,
                                    "follow_syndicates_id": follow_syndicates_id,
                                    "lead_line_size": lead_line_size,
                                    "follow_line_sizes": follow_line_sizes,
                                    "premium": premium,
                                    "risk_end_time": risks.get("risk_start_time")+365,
                                    "claim": False})

    def delete_contract(self, risks, syndicated_id):
        """
        Delete contract underwritten by bankrupt syndicates
        """
        index = 0
        while index < len(self.underwritten_contracts):
            if self.underwritten_contracts[i].get("syndicated_id") == syndicated_id:
                del self.underwritten_contracts[i]
                index -= 1
            index += 1

    def get_matched_contracts(self, syndicate_id, category):
        """
        Get underwritten contracts with the same category
        """
        matched_contracts = []
        for i in range(len(self.underwritten_contracts)):
            if (int(self.underwritten_contracts[i].get("syndicated_id")) == int(syndicated_id)) and (int(self.underwritten_contracts[i].get("risk_category")) == int(category)):
                matched_contracts.append(self.underwritten_contracts[i])

        return matched_contracts

    def not_underwritten_risk(self, risks):
        """
        Risks not being covered
        """
        if risks["risk_id"] not in self.underwritten_contracts:
            self.not_underwritten_risks.append(risks["risk_id"])

    def pay_premium(self, contract):
        """
        Pay premium to the syndicates

        Return
        ------
        total_premium: int
            All the money should be paied to the specific syndicate
        """
        
        premium = contract.get("premium")
        lead_syndicate_id = contract.get("lead_syndicate_id")
        follow_syndicates_id = contract.get("follow_syndicates_id")
        risk_category = contract.get("risk_category")
        
        return premium, lead_syndicate_id, follow_syndicates_id, risk_category

    def ask_claim(self, syndicate_id, follow_syndicates_id, category_id):
        """
        Ask payment from syndicate for covered risks

        Return
        ------
        claim_value: int
            All the money should be paied by the specific syndicate
        """
        matched_contracts = []
        claim_value = 0
        for i in range(len(self.underwritten_contracts)):
            if (int(self.underwritten_contracts[i].get("lead_syndicate_id")) == int(syndicate_id)) and (int(self.underwritten_contracts[i].get("risk_category")) == int(category_id)):
                matched_contracts.append(self.underwritten_contracts[i])
            for num in range(len(follow_syndicates_id)):
                if (follow_syndicates_id[num] in self.underwritten_contracts[i].get("follow_syndicates_id")) and (int(self.underwritten_contracts[i].get("risk_category")) == int(category_id)):
                    matched_contracts.append(self.underwritten_contracts[i])
        for i in range(len(matched_contracts)):
            claim_value += matched_contracts[i].get("risk_value") * ( 1 - self.deductible)

        return claim_value

    def end_contract_ask_claim(self, contract):

        claim = contract.get("risk_value")
        lead_syndicate_id = contract.get("lead_syndicate_id")
        follow_syndicates_id = contract.get("follow_syndicates_id")
        risk_category = contract.get("risk_category")

        return claim, lead_syndicate_id, follow_syndicates_id, risk_category

    def receive_claim(self, syndicate_id, category_id, require_claim_value, receive_claim_value):
        """
        Receive payment from syndicate for covered risks

        Return
        ------
        claim_value: int
            All the money should be paied by the specific syndicate
        """
        for contract in range(len(self.underwritten_contracts)):
            if (int(self.underwritten_contracts[contract].get("lead_syndicate_id")) == int(syndicate_id)) and (int(self.underwritten_contracts[contract].get("risk_category")) == int(category_id)):
                if require_claim_value == receive_claim_value:
                    self.underwritten_contracts[contract]["claim"] = True
                else:
                    self.not_paid_claims.append(self.underwritten_contracts[contract])

    def data(self):
        """
        Create a dictionary with key/value pairs representing the Broker data.

        Returns
        ----------
        dict
        """

        return {
            "broker_id": self.broker_id,
            "broker_risk": self.risks,
            "broker_quote": self.underwritten_contracts
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