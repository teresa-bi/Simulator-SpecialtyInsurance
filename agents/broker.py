"""
Contains all the capabilities of Brokers
"""

import json
import math

class Broker:
    def __init__(self, broker_id, broker_args, num_risk_models, sim_args, risk_model_configs):
        self.broker_id = broker_id
        self.broker_lambda_risks = broker_args["lambda_risks_daily"]
        self.deductible = broker_args["decuctible"]
        self.sim_time_span = sim_args["max_time"]
        self.risk_model_configs = risk_model_configs
        # Risks brought by broker
        self.risks = []
        # Risks be covered, underwritten
        self.underwritten_contracts = []

    def generate_risks(self, time):
        """
        Return risks brought by broker daily, according to poission distribution
        """
        if time > self.sim_time_span:
            Exception("Sorry, simulation stopped")
        random.seed(time + 123)
        model_id = random.randint(0,len(self.risk_model_configs)-1)
        self.risks = []
        ######TODO: need to be fixed, how to generate poission distribution risks
        num_risks_daily = self.broker_lambda_risks
        for index in range(num_risks_daily):
            self.risks.append({"risk_id": index,
                               "broker_id": self.broker_id,
                               "risk_start_time": time,
                               "risk_factor": self.risk_model_configs[model_id].get("risk_factor"),
                               "risk_category": self.risk_model_configs[model_id].get("risk_category"),
                               "risk_value": self.risk_model_configs[model_id].get("risk_value")})

        return self.risks

    def add_contract(self, risks, syndicated_id, premium):
        """
        Add new contract to the current underwritten_contracts list
        """
        self.underwritten_contracts.append({"risk_id": risks.get("risk_id"),
                                    "risk_start_time": risks.get("risk_start_time"),
                                    "risk_factor": risks.get("risk_factor"),
                                    "risk_category": risks.get("risk_category"),
                                    "risk_value": risks.get("risk_value"),
                                    "syndicated_id": syndicated_id,
                                    "premium": premium,
                                    "risk_end_time": risks.get("risk_start_time")+365})

    def delete_contract(self, risks, syndicated_id):
        """
        Delete contract underwritten by bankrupt syndicates
        """
        index = 0
        while index < len(self.underwritten_contracts)):
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
            if (self.underwritten_contracts[i].get("syndicated_id") == syndicated_id) and (self.underwritten_contracts[i].get("risk_category") == category):
                matched_contracts.append(self.underwritten_contracts[i])

        return matched_contracts

    def pay_premium(self, syndicate_id):
        """
        Pay premium to the syndicates

        Return
        ------
        total_premium: int
            All the money should be paied to the specific syndicate
        """
        matched_contracts = []
        total_premium = 0
        for i in range(len(self.underwritten_contracts)):
            if self.underwritten_contracts[i].get("syndicated_id") == syndicated_id:
                matched_contracts.append(self.underwritten_contracts[i])
        for i in range(len(matched_contracts)):
            total_premium += matched_contracts.get("premium")
        
        return total_premium

    def ask_claim(self, syndicate_id):
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
            if self.underwritten_contracts[i].get("syndicated_id") == syndicated_id:
                matched_contracts.append(self.underwritten_contracts[i])
        for i in range(len(matched_contracts)):
            claim_value += matched_contracts.get("risk_value") * ( 1 - self.deductible)

        return claim_value

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
            "broker_quote": self.underwritten_contracts,
            "broker_claim": self.claims
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