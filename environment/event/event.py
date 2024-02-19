from __future__ import annotations
from abc import ABC, abstractmethod

from environment.env import SpecialtyInsuranceMarketEnv


class Event(ABC):
    """
    A time bounded change to the Environment.
    """

    def __init__(self, start_time: float, repeated: bool = False):
        """
        Instantiate a new event.

        Parameters
        ----------
        start_time: float
            The (Environment) time in seconds at which the event starts.
        repeated : bool
            Whether or not to repeatedly apply the event at each time-step.
        """

        self.start_time = start_time
        self.repeated = repeated

    def __lt__(self, other_event: Event) -> bool:
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

