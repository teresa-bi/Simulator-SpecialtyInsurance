from __future__ import annotations
import json
import warnings
from agents import Broker
from environment.event.event import Event
from environment.market import NoReinsurance_RiskOne, NoReinsurance_RiskFour, Reinsurance_RiskOne, Reinsurance_RiskFour

class AddRiskEvent(Event):
    """
    Add risk event brought by the broker to the market
    """

    def __init__(self, risk_id, broker_id, risk_start_time, risk_end_time, risk_factor, risk_category, risk_value):
        """
        Construct a new insurable risk instance brought by the broker to the market

        Parameters
        ----------
        risk_id: str
            The risk identifier for all the risks generated by broker_id
        broker_id: int
            The broker who bring this risk to the market
        risk_start_time: int
            The time in days on which the risk brought to the market
        risk_end_time: int
            The time insurance contract ends, usually one contract lasts for 12 months
        risk_factor: int
        risk_category: int
            The risk categories the event belongs to 
        risk_value: int
            The risk amount (<= risk_limit 10000000)
        """
        Event.__init__(self, start_time=risk_start_time, repeated=False)

        self.risk_id = risk_id
        self.broker_id = broker_id
        self.risk_start_time = risk_start_time
        self.risk_end_time = risk_end_time
        self.risk_factor = risk_factor
        self.risk_category = risk_category
        self.risk_value = risk_value

    def run(self, market, step_time):
        """
        Add risk brought bt brokers to the insruance market 

        Parameters
        ----------
        market: NoReinsurance_RiskOne
            The scenario to accept risk event

        Returns
        -------
        market: NoReinsurance_RiskOne
            The updated market
        """

        # Set time the market should be at once MarketManager.evolve(step_time) is complete 
        if step_time is None:
            end_time = market.time
        else:
            end_time = market.time + step_time

        # If CatastropheEvent not yet initiated in the Market - add it
        if self.risk_id not in market.broker_bring_risk:
            warnings.warn(f"{self.risk_id} Event not in the market, cannot update...", UserWarning)
            return market

        # Add risk will influence broker contract
        for broker_id in range(len(market.brokers)):
            for syndicate_id in range(len(market.syndicates)):
                claim_value[broker_id][syndicate_id] = market.brokers[broker_id].ask_claim(syndicate_id, self.risk_category)
        

        return market

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
                "risk_start_time": self.risk_start_time,
                "risk_end_time": self.risk_end_time,
                "risk_factor": self.risk_factor,
                "risk_category": self.risk_category,
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




    

    

   