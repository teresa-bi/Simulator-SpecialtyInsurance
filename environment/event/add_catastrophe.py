from __future__ import annotations
import warnings
from environment.event.event import Event

class AddCatastropheEvent(Event):
    """
    Generate catastrophes
    """
    def __init__(self, risk_id, risk_start_time, risk_factor, risk_category, risk_value):
        """
        Construct a new catastrophe event

        Parameters
        ----------
        risk_id: str
            The risk identifier for all the risks generated by broker_id
        risk_start_time: int
            The time in days on which the risk brought to the market
        risk_factor: int
        risk_category: int
            The risk categories the event belongs to 
        risk_value: int
            The risk amount (<= risk_limit 10000000)
        """
        Event.__init__(self, start_time=risk_start_time, repeated=False)

        self.risk_id = risk_id
        self.risk_start_time = risk_start_time
        self.risk_factor = risk_factor
        self.risk_category = risk_category
        self.risk_value = risk_value
        
    def run(self, market, step_time):
        """
        Add catastrophe to the insruance market

        Parameters
        ----------
        market: NoReinsurance_RiskOne
            The insurance market to accept catastrophe event

        Returns
        -------
        market: NoReinsurance_RiskOne
            The updated insurance market
        """
        n = len(market.catastrophe_event)
        count = 0
        for i in range(n):
            if self.risk_id != market.catastrophe_event[i].risk_id:
                count += 1
        if count == n:
            warnings.warn(f"{self.risk_id} Catastrophe not in the Market, cannot update...", UserWarning)
        
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
                "catastrophe_id": self.risk_id,
                "catastrophe_start_time": self.risk_start_time,
                "catastrophe_factor": self.risk_factor,
                "catastrophe_category": self.risk_category,
                "catastrophe_value": self.risk_value
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
