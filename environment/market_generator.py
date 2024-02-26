from __future__ import annotations
import json
import time
import numpy as np

from agents import Broker, Syndicate, Shareholder, ReinsuranceFirm

class MarketGenerator:
    """
    Market with broker, syndicate, reinsurance firms, and shareholder in the market
    """
    def __init__(self, with_reinsurance, num_risk_models, sim_args, broker_args, syndicate_args, reinsurancefirm_args, shareholder_args, risk_model_configs):
        """
        Instance of an insurance market

        Parameters
        ----------
        with_reinsurance: bool
            True will generate reinsurance firms and include reinsurance risks, while Flase will not include reinsurance
        num_risk_models: int
            1 to 4 risk models
        sim_args: dict
        broker_args: dict
        syndicate_args: dict
        reinsurancefirm_args: dict
        shareholder_args: dict
        risk_model_configs: list of dict
            risk model configurations to be passed to RiskModel instance
        """
        # With or without reinsurance firms in the market
        self.with_reinsurance = with_reinsurance
        self.num_risk_models = num_risk_models
        # Get inputs
        self.sim_args = sim_args
        self.broker_args = broker_args
        self.syndicate_args = syndicate_args
        self.reinsurancefirm_args = reinsurancefirm_args
        self.shareholder_args = shareholder_args
        self.risk_model_configs = risk_model_configs
        # Init list of agents
        self.brokers = {}
        self.syndicates = {}
        self.reinsurancefirms = {}
        self.shareholders = {}

    def generate_agents(self):
        """
        Generate brokers, syndicates, reinsurancefimrs, shareholders, risk_models
        """
        # Generate brokers
        for i in range(self.broker_args["num_brokers"]):
            self.brokers[i] = Broker(i, self.broker_args, self.num_risk_models, self.sim_args, self.risk_model_configs)

        # Generate syndicates
        for i in range(self.syndicate_args["num_syndicates"]):
            self.syndicates[i] = Syndicate(i, self.syndicate_args, self.num_risk_models, self.sim_args, self.risk_model_configs)

        # Generate reinsurancefirms
        if self.with_reinsurance:
            for i in range(self.reinsurancefirm_args["num_reinsurancefirms"]):
                self.reinsurancefirms[i] = ReinsuranceFirm(i, self.reinsurancefirm_args, self.num_risk_models, self.risk_model_configs)

        # Generate shareholders
        for i in range(self.shareholder_args["num_shareholders"]):
            self.shareholders[i] = Shareholder(i, self.shareholder_args, self.num_risk_models, self.risk_model_configs)

        return self.brokers, self.syndicates, self.reinsurancefirms, self.shareholders

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

    