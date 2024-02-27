from __future__ import annotations
import json
import warnings
from environment.event.event import Event
from environment.market import NoReinsurance_RiskOne, NoReinsurance_RiskFour, Reinsurance_RiskOne, Reinsurance_RiskFour

class AddPremiumEvent(Event):
    """
    Add claim event caused by catastrophe event
    """
    def __init__(self, risk_id, broker_id, risk_start_time, risk_end_time, risk_category, risk_value, syndicate_id, premium):
        """
        Construct a new claim instance

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
        risk_category: int
            The risk categories the event belongs to 
        risk_value: int
            The risk amount (<= risk_limit 10000000)
        syndicate_id: str
            The syndicate who underwrite this risk
        premium: int
            The cash paied to the syndicate each month
        """

        Event.__init__(self, start_time=risk_start_time, repeated=False)

        self.risk_id = risk_id
        self.broker_id = broker_id
        self.risk_start_time = risk_start_time
        self.risk_end_time = risk_end_time
        self.risk_category = risk_category
        self.risk_value = risk_value
        self.syndicate_id = syndicate_id
        self.premium = premium

    def run(self, market, step_time):
        """
        Add claim to the insurance market 

        Parameters
        ----------
        market: NoReinsurance_RiskOne
            The insurance market to accept payment event

        Returns
        -------
        market: NoReinsurance_RiskOne
            The updated market
        """

        market.broker_pay_premium[self.risk_id] = {"risk_id": self.risk_id,
                                                "broker_id": self.broker_id,
                                                "risk_start_time": self.risk_start_time,
                                                "risk_end_time": self.risk_end_time,
                                                "risk_category": self.risk_category,
                                                "risk_value": self.risk_value,
                                                "syndicate_id": self.syndicate_id,
                                                "premium": self.premium
                                                }

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
                "risk_category": self.risk_category,
                "risk_value": self.risk_value,
                "syndicate_id": self.syndicate_id,
                "premium": self.premium
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

    def get_syndicate_status(self, syndicates):
        """
        Update the syndicate status after the add premium event

        Return
        -------
        All Syndicate status, TODO: include current capital, current capital in risk category
        """
        for sy_id in range(len(syndicates)):
            syndicates[sy_id].receive(self.premium)

