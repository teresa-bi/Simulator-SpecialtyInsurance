from __future__ import annotations
import numpy as np
import json
import time
from collections import defaultdict

class NoReinsurance_RiskOne:
    """
    Basic insurance market including brokers, syndicates, sharholders, and one risk model
    """
    def __init__(self, time, sim_maxstep, manager_args, brokers, syndicates, shareholders, risks, risk_model_configs, catastrophe_event, attritional_loss_event, broker_risk_event, broker_premium_event, broker_claim_event):
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
        catastrophe_event: list of catastrophes
        attritional_loss_event: list of attritional loss events
        broker_risk_event: list of risk events brought by broker
        broker_premium_event: list of premium events
        broker_claim_event: list of claim events
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
        self.catastrophe_event = catastrophe_event
        self.attritional_loss_event = attritional_loss_event
        self.broker_bring_risk = broker_risk_event
        self.broker_pay_premium = broker_premium_event
        self.broker_bring_claim = broker_claim_event

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


