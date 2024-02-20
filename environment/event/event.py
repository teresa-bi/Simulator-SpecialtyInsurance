from __future__ import annotations
from abc import ABC, abstractmethod

from environment.market import NoReinsurance_RiskOne, NoReinsurance_RiskFour, Reinsurance_RiskOne, Reinsurance_RiskFour


class Event(ABC):
    """
    A time bounded change to the Environment.
    """

    def __init__(self, start_time, repeated=False):
        """
        Instantiate a new event.

        Parameters
        ----------
        start_time: float
            The (Environment) time in days at which the event starts.
        repeated : bool
            Whether or not to repeatedly apply the event at each time-step.
        """

        self.start_time = start_time
        self.repeated = repeated

    def __lt__(self, other_event):
        """
        Less than method to compare start_time between Events. Used to sort Events by start_time.

        Parameters
        ----------
        other_event: Event

        Returns
        --------
        bool
        """

        return self.start_time < other_event.start_time

    @abstractmethod
    def run(self, environment, **kwargs):
        """
        Apply Event to the Environment.

        Parameters
        ----------
        environment: SpecialtyInsuranceMarketEnv
            The Environment to act upon.
        **kwargs : Dict[str, Any]
            A dictionary of key word arguments. These can vary per event.

        Returns
        --------
        SpecialtyInsuranceMarketEnv
            The updated Environment.
        """

        pass

    @abstractmethod
    def data(self) -> dict:
        """
        Get the data as a serialisable dictionary.

        Returns
        --------
        dict
        """

        pass

