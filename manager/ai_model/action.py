from __future__ import annotations

import json
import typing


class Action:
    """
    Actions taken by syndicates.
    """

    def __init__(self, syndicate: int, line_size: float, risk: int, broker: int):
        """
        Construct a new instance.

        Parameters
        ----------
        syndicate: int
            The Syndicate identifier
        broker: int
            The Broker identifier.
        risk: int
            The risk identifier brought by the broker into the insurance market
        value: float
            The line size covered by the syndicate

        Examples
        --------
        >>> action = Action(0, 0.9, 1, 1) Syndicate 0 will cover 0.9 line size of the risk 1 brought by Broker 1
        """

        self.syndicate = syndicate
        self.line_size = line_size
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
        >>> Action.from_str("0 0.9 1 1")
        """

        s.strip()

        s = s.split()
        syndicate = s[0]
        line_size = s[1]
        risk = s[2]
        broker = s[3]

        return Action(syndicate, line_size, risk, broker)

    def __str__(self) -> str:
        """
        Create a human readable representation of the instance.

        Returns
        ----------
        str
            A string representation of Action (e.g., "Syndicate 0 covers 0.9 line size 0f risk 1 brought by broker 1")
        """

        return f"{self.syndicate} {self.line_size} {self.risk} {self.broker}"