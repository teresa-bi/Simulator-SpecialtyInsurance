from __future__ import annotations
import json
import time
import numpy as np

from .agents import Broker, Syndicate, ReinsuranceFirm, Shareholder
from environment.risk import RiskModle


class NoReinsurance_RiskOne:
    """
    Scenario with broker, syndicate, shareholder in the market, and all the syndicates use one risk modle
    """
    def __init__(self):

        # With or without reinsurance firms in the scenario
        self.reinsurance = False
        # Use one risk model
        self.one_risk = True
        # Use four risk models
        self.four_risk = False
        # List of agents
        self.brokers = {}
        self.syndicates = {}
        self.reinsurancefirms = {}
        self.shareholders = {}
        self.risk_models = {}

    def create(self, broker_args, syndicate_args, reinsurancefirm_args, shareholder_args, risk_args):
        """
        Create agents including brokers, syndicates, reinsurancefirms, shareholders, risk_models.

        Parameters
        ----------
        broker_args, syndicate_args, reinsurancefirm_args, shareholder_args, risk_args

        Returns
        ----------
        dict
        """
        for i in range(broker_args["num_brokers"]):
            self.brokers[str(i)] = Broker(i,broker_args)
        for i in range(syndicate_args["num_syndicates"])]:
            self.syndicates[str(i)] = Syndicate(i,syndicate_args)
        if self.reinsurance == True:
            for i in range(reinsurancefirm_args["num_reinsurancefirms"])]:
                self.reinsurancefirms[str(i)] = ReinsuranceFirm(i,reinsurancefirm_args)
        for i in range(shareholder_args["num_shareholders"]):
            self.shareholders[str(i)] = Shareholder(i,shareholder_args)
        for i in range(risk_args["num_risks"]):
            if self.one_risk:
                self.risk_models[str(i)] = RiskModel.one_risk_model(risk_args) 
            elif self.four_risk:
                self.risk_models[str(i)] = RiskModel.four_risk_model(risk_args)
            else:
                self.risk_models[str(i)] = RiskModel.other_risk_model(risk_args)

        return self.brokers, self.syndicates, self.reinsurancefirms, self.shareholders, self.risk_models

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
            "risk_models": self.risk_models.data(),
            "active_syndicates": {syndicate_id: syndicate.data() for (syndicate_id, syndicate) in self.active_syndicate.items()},
            "broker_bring_risk": {risk_id: risk.data() for (risk_id, risk) in self.broker_bring_risk.items()},
            "broker_bring_claim": {risk_id: risk.data() for (risk_id, risk) in self.broker_bring_claim.items()},
            "catastrophe_event": {catastrophe_id: catastrophe_event.data() for (catastrophe_id, catastrophe_event) in self.catastrophe_event.items()}
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

    