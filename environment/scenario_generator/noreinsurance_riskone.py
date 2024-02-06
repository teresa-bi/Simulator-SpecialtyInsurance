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
    def __init__(self, time, brokers, syndicates, reinsurancefirms, shareholders, risk_models):

        if time < 0.0:
            raise ValueError("Time must be non-negative.")

        # With or without reinsurance firms in the scenario
        self.reinsurance = False
        # Use one risk models
        self.one_risk = True
        # Use four risk models
        self.four_risk = False
        # List of brokers
        self.brokers = brokers
        # List of syndicates
        self.syndicates = syndicates
        # List of shareholders
        self.shareholders = shareholders
        # List of risk models used by synidacates
        self.risk_models = risk_models
        # Time Step
        self.time = time
        # Add event to the scenario
        self.active_syndicate = {}
        self.broker_bring_risk = {}
        self.broker_bring_claim = {}
        self.catastrophe_event= {}

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

    def is_active(self)
        """
        Determine which Syndicates are active (i.e., are not bankrupt).

        Returns
        ----------
        dict[str, bool]
            {Syndicate_id: whether Syndicate is active}
        """

        active = {}

        for syndicate in self.sydicates.items():
            if syndicate.current_capital >= 0:
                active[syndicate] = True
            else:
                active[syndicate] = False

        return active

    