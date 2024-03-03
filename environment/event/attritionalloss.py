import warnings
from environment.event.event import Event

class AttritionalLossEvent(Event):
    """
    Generate daily attritional loss
    """
    def __init__(self, risk_id, risk_start_time, risk_factor, risk_category, risk_value):
        """
        Construct a new attritional loss event

        Parameters
        ----------
        risk_id: str
            The risk identifier for all the attritional loss 
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

    def data(self):
        """
        Get the data as a serialisable dictionary.

        Returns
        --------
        dict
        """

        return {
            self.__class__.__name__: {
                "attritional_loss_id": self.risk_id,
                "attritional_loss_start_time": self.risk_start_time,
                "attritional_loss_factor": self.risk_factor,
                "attritional_loss_category": self.risk_category,
                "attritional_loss_value": self.risk_value
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
    
