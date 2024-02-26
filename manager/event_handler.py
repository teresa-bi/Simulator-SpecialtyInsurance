from __future__ import annotations
import json

from environment.market import NoReinsurance_RiskOne, NoReinsurance_RiskFour, Reinsurance_RiskOne, Reinsurance_RiskFour


class EventHandler:
    """
    The EventHandler applies events to the internal environment (market) of the Environment Manager
    """

    def __init__(self, maxstep, catastrophe_events, attritional_loss_events, broker_risk_events, broker_premium_events, broker_claim_events):
        """
        Construct a new instance.

        Parameters
        ----------
        maxstep: int
            Simulation time span.
        catastrophe_events: List[Event]
            List of catastrophe events to apply to the environment.
        attritional_loss_events: List[Event]
            List of attritional_loss events to apply to the environment.
        broker_risk_events: List[Event]
            List of broker_risk_events to apply to the environment.
        broker_premium_events: List[Event]
            List of broker_premium_events to apply to the environment.
        broker_claim_events: List[Event]
            List of broker_claim_events to apply to the environment.
        """

        # Events to be carried out some time in the future
        # Events are stored internally in a dictionary to allow for O(1) access/removal
        self.upcoming_catastrophe = {event_id: event for (event_id, event) in enumerate(catastrophe_events)}
        self.upcoming_attritional_loss = {event_id: event for (event_id, event) in enumerate(attritional_loss_events)}
        self.upcoming_broker_risk = {event_id: event for (event_id, event) in enumerate(broker_risk_events)}
        self.upcoming_broker_premium = {event_id: event for (event_id, event) in enumerate(broker_premium_events)}
        self.upcoming_broker_claim = {event_id: event for (event_id, event) in enumerate(broker_claim_events)}

        # Events that are currently underway and should be carried out at each time-step
        self.ongoing_catastrophe = {}
        self.ongoing_attritional_loss = {}
        self.ongoing_broker_risk = {}
        self.ongoing_broker_premium = {}
        self.ongoing_broker_claim = {}

        # Events that have been completed
        self.completed_catastrophe = {}
        self.completed_attritional_loss = {}
        self.completed_broker_risk = {}
        self.completed_broker_premium = {}
        self.completed_broker_claim = {}

    def data(self):
        """
        Get the data as a serialisable dictionary.

        Returns
        --------
        dict
        """

        return {
            "upcoming_catastrophe": [(event_id, event.data()) for (event_id, event) in self.upcoming_catastrophe.items()],
            "upcoming_attritional_loss": [(event_id, event.data()) for (event_id, event) in self.upcoming_attritional_loss.items()],
            "upcoming_broker_risk": [(event_id, event.data()) for (event_id, event) in self.upcoming_broker_risk.items()],
            "upcoming_broker_premium": [(event_id, event.data()) for (event_id, event) in self.upcoming_broker_premium.items()],
            "upcoming_broker_claim": [(event_id, event.data()) for (event_id, event) in self.upcoming_broker_claim.items()],
            "ongoing_catastrophe": [(event_id, event.data()) for (event_id, event) in self.ongoing_catastrophe.items()],
            "ongoing_attritional_loss": [(event_id, event.data()) for (event_id, event) in self.ongoing_attritional_loss.items()],
            "ongoing_broker_risk": [(event_id, event.data()) for (event_id, event) in self.ongoing_broker_risk.items()],
            "ongoing_broker_premium": [(event_id, event.data()) for (event_id, event) in self.ongoing_broker_premium.items()],
            "ongoing_broker_claim": [(event_id, event.data()) for (event_id, event) in self.ongoing_broker_claim.items()],
            "completed_catastrophe": [(event_id, event.data()) for (event_id, event) in self.completed_catastrophe.items()],
            "completed_attritional_loss": [(event_id, event.data()) for (event_id, event) in self.completed_attritional_loss.items()],
            "completed_broker_risk": [(event_id, event.data()) for (event_id, event) in self.completed_broker_risk.items()],
            "completed_broker_premium": [(event_id, event.data()) for (event_id, event) in self.completed_broker_premium.items()],
            "completed_broker_claim": [(event_id, event.data()) for (event_id, event) in self.completed_broker_claim.items()],
        }

    def to_json(self) -> str:
        """
        Serialise the instance to JSON string.

        Returns
        ----------
        str
        """

        return json.dumps(self.data(), indent=4)

    def save(self, filename: str):
        """
        Write the instance to a file.

        Parameters
        ----------
        filename: str
            Path to file.
        """

        with open(filename, "w") as file:
            file.write(self.to_json())

    def forward(self, market, step_time):
        """
        Evolve Market for the given time step [day] by applying Events.

        Parameters
        ----------
        market: NoReinsurance_RiskOne
            The insurance market to apply Events to.
        step_time: int
            Amount of time in days the market is evolving for.
        """

        episode_start = market.time
        episode_end = episode_start + step_time

        # Temporary dict to store which events to remove from the ongoing dict
        catastrophe_events_to_remove_from_ongoing = {}

        # Carry out ongoing (repeating) events
        for event_id, event in self.ongoing_catastrophe.items():
            # step_time is needed here for the Event
            # to ensure we get market status at the market.time after evolve(step_time) is completed i.e., current market.time + step_time
            market = event.run(market, step_time=step_time)

            # Check if the event is no longer to be repeated
            if not event.repeated:
                self.completed_catastrophe[event_id] = event

                # Store the event_id to be removed
                catastrophe_events_to_remove_from_ongoing[event_id] = 1

        # Attritional loss event
        attritional_loss_events_to_remove_from_ongoing = {}
        for event_id, event in self.ongoing_attritional_loss.items():
            market = event.run(market, step_time=step_time)
            if not event.repeated:
                self.attritional_loss_catastrophe[event_id] = event
                attritional_loss_events_to_remove_from_ongoing[event_id] = 1

        # Broker brought risk event
        broker_risk_events_to_remove_from_ongoing = {}
        for event_id, event in self.ongoing_broker_risk.items():
            market = event.run(market, step_time=step_time)
            if not event.repeated:
                self.broker_risk_catastrophe[event_id] = event
                broker_risk_events_to_remove_from_ongoing[event_id] = 1

        # Broker brought premium event
        broker_premium_events_to_remove_from_ongoing = {}
        for event_id, event in self.ongoing_broker_premium.items():
            market = event.run(market, step_time=step_time)
            if not event.repeated:
                self.broker_premium_catastrophe[event_id] = event
                broker_premium_events_to_remove_from_ongoing[event_id] = 1

        # Broker brought claim event
        broker_claim_events_to_remove_from_ongoing = {}
        for event_id, event in self.ongoing_broker_claim.items():
            market = event.run(market, step_time=step_time)
            if not event.repeated:
                self.broker_claim_catastrophe[event_id] = event
                broker_claim_events_to_remove_from_ongoing[event_id] = 1

        # Temporary dict to store which events to remove from the upcoming dict
        catastrophe_events_to_remove_from_upcoming = {}

        # Carry out the new events
        for event_id, event in self.upcoming_catastrophe.items():
            if event.start_time >= episode_start and event.start_time <= episode_end:
                # step_time is needed here for the event
                # to ensure we get market status at the market.time after
                # evolve(step_time) is completed i.e., current market.time + step_time
                market = event.run(market, step_time=step_time)

                # Move the event to its new status
                if event.repeated:
                    self.ongoing_catastrophe[event_id] = event
                else:
                    self.completed_catastrophe[event_id] = event

                # Store the event_id to be removed
                catastrophe_events_to_remove_from_upcoming[event_id] = 1

        # Remove no longer repeating events from the ongoing event dict
        for event_id in catastrophe_events_to_remove_from_ongoing:
            del self.ongoing_catastrophe[event_id]

        # Remove newly-started events from the upcoming event dict
        for event_id in catastrophe_events_to_remove_from_upcoming:
            del self.upcoming_catastrophe[event_id]

        # Attritional loss event
        attritional_loss_events_to_remove_from_upcoming = {}
        for event_id, event in self.upcoming_attritional_loss.items():
            if event.start_time >= episode_start and event.start_time <= episode_end:
                market = event.run(market, step_time=step_time)
                if event.repeated:
                    self.ongoing_attritional_loss[event_id] = event
                else:
                    self.completed_attritional_loss[event_id] = event
                attritional_loss_events_to_remove_from_upcoming[event_id] = 1
        for event_id in attritional_loss_events_to_remove_from_ongoing:
            del self.ongoing_attritional_loss[event_id]
        for event_id in attritional_loss_events_to_remove_from_upcoming:
            del self.upcoming_attritional_loss[event_id]

        # Broker brought risk event
        broker_risk_events_to_remove_from_upcoming = {}
        for event_id, event in self.upcoming_broker_risk.items():
            if event.start_time >= episode_start and event.start_time <= episode_end:
                market = event.run(market, step_time=step_time)
                if event.repeated:
                    self.ongoing_broker_risk[event_id] = event
                else:
                    self.completed_broker_risk[event_id] = event
                broker_risk_events_to_remove_from_upcoming[event_id] = 1
        for event_id in broker_risk_events_to_remove_from_ongoing:
            del self.ongoing_broker_risk[event_id]
        for event_id in broker_risk_events_to_remove_from_upcoming:
            del self.upcoming_broker_risk[event_id]

        # Broker pay premium event
        broker_premium_events_to_remove_from_upcoming = {}
        for event_id, event in self.upcoming_broker_premium.items():
            if event.start_time >= episode_start and event.start_time <= episode_end:
                market = event.run(market, step_time=step_time)
                if event.repeated:
                    self.ongoing_broker_premium[event_id] = event
                else:
                    self.completed_broker_premium[event_id] = event
                broker_premium_events_to_remove_from_upcoming[event_id] = 1
        for event_id in broker_premium_events_to_remove_from_ongoing:
            del self.ongoing_broker_premium[event_id]
        for event_id in broker_premium_events_to_remove_from_upcoming:
            del self.upcoming_broker_premium[event_id]

        # Broker brought claim event
        broker_claim_events_to_remove_from_upcoming = {}
        for event_id, event in self.upcoming_broker_claim.items():
            if event.start_time >= episode_start and event.start_time <= episode_end:
                market = event.run(market, step_time=step_time)
                if event.repeated:
                    self.ongoing_broker_claim[event_id] = event
                else:
                    self.completed_broker_claim[event_id] = event
                broker_claim_events_to_remove_from_upcoming[event_id] = 1
        for event_id in broker_claim_events_to_remove_from_ongoing:
            del self.ongoing_broker_claim[event_id]
        for event_id in broker_claim_events_to_remove_from_upcoming:
            del self.upcoming_broker_claim[event_id]

        return market

