from __future__ import annotations
import warnings
from environment.event.event import Event

class AddCatastropheEvent(Event):
    """
    Generate catastrophes
    """
    def __init__(self, catastrophe_id, catastrophe_start_time, catastrophe_category, catastrophe_value):
        """
        Construct a new catastrophe event

        Parameters
        ----------
        catastrophe_id: str
            The catastrophe identifier for all the catastrophes
        catastrophe_start_time: int
            The time in days on which the catastrophe brought to the market
        catastrophe_category: int
            The catastrophe categories the event belongs to 
        catastrophe_value: int
            The catastrophe damage amount (<= catastrophe_limit 10000000)
        """
        Event.__init__(self, start_time=catastrophe_start_time, repeated=False)

        self.catastrophe_id = catastrophe_id
        self.catastrophe_start_time = catastrophe_start_time
        self.catastrophe_category = catastrophe_category
        self.catastrophe_value = catastrophe_value
        
    def run(self, market):
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
            if self.catastrophe_id != market.catastrophe_event[i].catastrophe_id:
                count += 1
        if count == n:
            warnings.warn(f"{self.catastrophe_id} Catastrophe not in the Market, cannot update...", UserWarning)
        
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
                "catastrophe_id": self.catastrophe_id,
                "catastrophe_start_time": self.catastrophe_start_time,
                "catastrophe_category": self.catastrophe_category,
                "catastrophe_value": self.catastrophe_value
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
