from __future__ import annotations
import json
from environment.scenario_generator import NoReinsurance_RiskOne

class AddRiskEvent(Event):
    """
    Add risk event brought by the broker to the market
    """

    def __init__(self, risk_id, broker_id, risk_categories, risk_value, start_time, end_time):
        """
        Construct a new insurable risk instance brought by the broker to the market

        Parameters
        ----------
        risk_id: str
            The risk identifier
        broker_id: int
            The broker who bring this risk to the market
        risk_categories: int
            The risk region the event belongs to (from 0 to 9)
        risk_value: int
            The risk amount (<= risk_limit 10000000)
        start_time: int
            The time in days on which the risk brought to the market
        end_time: int
            The time insurance contract ends, usually one contract lasts for 12 months
        """

        self.risk_id = risk_id
        self.broker_id = broker_id
        self.risk_categories = risk_categories
        self.risk_value = risk_value
        self.start_time = start_time
        self.end_time = end_time

    def run(self, scenario):
        """
        Add risk to the base scenario NoReinsurance_RiskOne

        Parameters
        ----------
        scenario: NoReinsurance_RiskOne
            The scenario to accept risk event

        Returns
        -------
        scenario: NoReinsurance_RiskOne
            The updated scenario
        """

        scenario.broker_bring_risk[self.risk_id] = {"risk_id": self.risk_id,
                                                    "broker_id": self.broker_id,
                                                    "risk_categories": self.risk_categories,
                                                    "risk_value": self.risk_value,
                                                    "start_time": self.start_time,
                                                    "end_time": self.end_time
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
                "risk_value": self.risk_value,
                "start_time": self.start_time,
                "end_time": self.end_time
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
