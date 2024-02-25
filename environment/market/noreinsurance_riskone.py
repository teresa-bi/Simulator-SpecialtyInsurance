from __future__ import annotations
import numpy as np
import json
import time
from collections import defaultdict

class NoReinsurance_RiskOne:
    """
    Basic insurance market including brokers, syndicates, sharholders, and one risk model
    """
    def __init__(self, time, sim_maxstep, manager_args, brokers, syndicates, shareholders, risks, risk_model_configs):
        """
        Construct a new instance.

        Parameters
        ----------
        time: float
            Current time in the Market.
        sim_maxstep: int
            Simulation time span.
        manager_args: dict
        brokers: list of Broker
        syndicates: list of Syndicate
        shareholders: list of Shareholder
        risks: list of catastrophe
        risk_model_configs: risk model configuration
        """

        if time < 0.0:
            raise ValueError("Time must be non-negative.")

        self.time = time
        self.sim_maxstep = sim_maxstep
        self.manager_args = manager_args
        self.brokers = brokers
        self.syndicates = syndicates
        self.shareholders = shareholders
        self.risks = risks
        self.risk_model_configs = risk_model_configs
        
        # Status of risks and claims
        # Broker_bring_risk 
        self.broker_bring_risk = self.brokers.risks
        self.underwritten_risk = self.brokers.underwritten_contracts
        self.not_underwritten_risk = not_underwritten_risk()
        # Broker_pay_premium
        self.broker_pay_premium = self.syndicates.receive_premium
        # Broker_bring_claim
        self.broker_bring_claim = self.brokers.ask_claim
        # Not paid claim
        self.not_paid_claim = not_paid_claim()

    def data(self):
        """
        Get the data as a serialisable dictionary.

        Returns
        ----------
        dict
        """

        timestamp = time.gmtime(self.time)
        time_str = time.strftime("%H:%M:%S", timestamp)

        return {
            "time": time_str,
            "brokers": self.brokers.data(),
            "syndicates": self.syndicates.data(),
            "shareholders": self.shareholders.data(),
            "underwritten_risk": self.underwritten_risk,
            "not_paid_claims": self.not_paid_claim
        }

    def to_json(self) -> str:
        """
        Serialise the instance to JSON.

        Returns
        ----------
        str
        """

        return json.dumps(self.data(), indent=4)

    def save(self, filename):
        """
        Write the instance to a file.

        Parameters
        ----------
        filename: str
            Path to file.
        """

        with open(filename, "w") as file:
            file.write(self.to_json())

    def not_underwritten_risk(self):
        not_underwritten_risks = []
        under_written_riskid = []
        for written_risk_id in range(len(self.brokers.underwritten_contracts)):
            under_written_riskid.append(self.brokers.underwritten_contracts[written_risk_id]["risk_id"])
        for bring_risk_id in range(len(self.brokers.risks)):
            if self.brokers.risks[bring_risk_id]["risk_id"] not in under_written_riskid:
                not_underwritten_risks.append(self.brokers.risks[bring_risk_id])
        return not_underwritten_risks

    def not_paid_claim(self):
        not_paid_claims = []
        for contract in len(self.underwritten_contracts):
            if self.underwritten_contracts[contract]["claim"] == False:
                not_paid_claims.append(self.underwritten_contracts[contract])
        return not_paid_claims


