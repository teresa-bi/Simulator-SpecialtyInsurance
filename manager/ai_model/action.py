from __future__ import annotations

import json
import typing


class Action:
    """
    Actions taken by syndicates.
    """

    def __init__(self, syndicate: int, premium: float, risk: int, broker: int):
        """
        Construct a new instance.

        Parameters
        ----------
        syndicate: int
            The Syndicate identifier
        premium: float
            The premium offered by the syndicate
        risk: int
            The risk identifier brought by the broker into the insurance market
        broker: int
            The Broker identifier.

        Examples
        --------
        >>> action = Action(0, 10000, 1, 1) Syndicate 0 will offer 10000 premium of the risk 1 brought by Broker 1
        """

        self.syndicate = syndicate
        self.premium = premium
        self.risk = risk
        self.broker = broker

    @staticmethod
    def from_str(s: str) -> Action:
        """
        Construct a new instance from a string representation.

        Parameters
        ----------
        s: str
            A string representation of Action.

        Returns
        ----------
        Action

        Examples
        ----------
        >>> Action.from_str("0 10000 1 1")
        """

        s.strip()

        s = s.split()
        syndicate = s[0]
        premium = s[1]
        risk = s[2]
        broker = s[3]

        return Action(syndicate, premium, risk, broker)

    def __str__(self) -> str:
        """
        Create a human readable representation of the instance.

        Returns
        ----------
        str
            A string representation of Action (e.g., "Syndicate 0 will offer 10000 premium 0f risk 1 brought by broker 1")
        """

        return f"{self.syndicate} {self.premium} {self.risk} {self.broker}"