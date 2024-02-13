from __future__ import annotations
import json
from environment.scenario_generator import NoReinsurance_RiskOne

class AddClaimEvent(Event):
    """
    Add claim event caused by catastrophe event
    """
    def __init__(self, risk_id, broker_id, risk_regions, risk_value):
        """
        Construct a new claim instance

        Parameters
        ----------
        risk_id: str
            The risk identifier
        broker_id: int
            The broker who bring this claim to the market
        risk_regions: int
            The risk region the event belongs to (from 0 to 9)
        risk_value: int
            The risk amount (<= risk_limit 10000000)
        """

        self.risk_id = risk_id
        self.broker_id = broker_id
        self.risk_regions = risk_regions
        self.risk_value = risk_value

    def run(self, scenario):
        """
        Add claim to the base scenario NoReinsurance_RiskOne

        Parameters
        ----------
        scenario: NoReinsurance_RiskOne
            The scenario to accept risk event

        Returns
        -------
        scenario: NoReinsurance_RiskOne
            The updated scenario
        """

        scenario.broker_bring_claim[self.risk_id] = {"risk_id": self.risk_id,
                                                "broker_id": self.broker_id,
                                                "risk_regions": self.risk_regions,
                                                "risk_value": self.risk_value
                                                }
        return scenario

    def data(self):
        """
        Get the data as a serialisable dictionary.

        Returns
        --------
        dict
        """

        return {
            self.__class__.__name__: {
                "risk_id": self.risk_id,
                "broker_id": self.broker_id,
                "risk_regions": self.risk_regions,
                "risk_value": self.risk_value
            }
        }

    def to_json(self):
        """
        Serialise the instance to JSON.

        Returns
        ----------
        str
        """

        return json.dumps(self.data(), indent=4)


    def save(self, filename):
        """
        Write the instance to a log file.

        Parameters
        ----------
        filename: str
            Path to file.
        """

        with open(filename, "w") as file:
            file.write(self.to_json())
