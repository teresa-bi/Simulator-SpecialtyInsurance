from __future__ import annotations
import json
import warnings
from environment.event.event import Event

class AddClaimEvent(Event):
    """
    Add claim event caused by contracts end
    """
    def __init__(self, risk_id, risk_start_time):
        """
        Construct a new claim instance

        Parameters
        ----------
        risk_id: str
            The risk identifier for all the risks generated by broker_id
        risk_start_time: int
            The time in days on which the risk brought to the market
        """

        Event.__init__(self, start_time=risk_start_time, repeated=False)

        self.risk_id = risk_id
        self.risk_start_time = risk_start_time
      
    def run(self, market, step_time):
        """
        Add claim to the insurance market

        Parameters
        ----------
        market: NoReinsurance_RiskOne
            The insurance market to accept claim event

        Returns
        -------
        market: NoReinsurance_RiskOne
            The updated market
        """
        n = len(market.broker_bring_claim)
        count = 0
        for i in range(n):
            if self.risk_id != market.broker_bring_claim[i].risk_id:
                count += 1
        if count == n:
            warnings.warn(f"{self.risk_id} Claim Event not in the Market, cannot update...", UserWarning)

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
                "risk_start_time": self.risk_start_time,
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
