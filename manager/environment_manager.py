from __future__ import annotations
import typing
import warnings
from collections import defaultdict

import numpy as np
from agents import Broker, Syndicate, Shareholder, ReinsuranceFirm
from environment.scenario_generator import NoReinsurance_RiskOne, NoReinsurance_RiskFour, Reinsurance_RiskOne, Reinsurance_RiskFour
from environment.risk import RiskEvent, CatastropheEvent, AttritionalLossEvent, AddRiskEvent, AddClaimEvent, RiskModel
from manager.event_handler import EventHandler


class EnvironmentManager:
    """
    Manage and evolve the Environment.
    """

    def __init__(
        self,
        brokers,
        syndicates,
        reinsurancefirms,
        shareholders,
        risk_models,
        event_handler,
        logger,
        time = 0,
    ):
        """
        Construct a new instance.

        Parameters
        ----------
        event_handler: EventHandler
            The EventHandler applies events to the internal Environment of the EM.
        logger: Logger
            An optional Starling Logger.
        time: float
            Environment start time.
        """

        self.brokers = brokers
        self.syndicates = syndicates
        self.reinsurancefirms = reinsurancefirms
        self.shareholders = shareholders
        self.risk_models = risk_models

        self.environment = NoReinsurance_RiskOne(time, self.brokers, self.syndicates, self.reinsurancefirms, self.shareholders, self.risk_models)

        self.min_step_time = 1  # Day Event

        self.actions_to_apply = []
        # for logging keep track of all Actions ever received
        # and whether they were accepted or refused by the manager
        self.actions_accepted = {}
        self.actions_refused = {}

        # logging
        self.logger = logger
        if self.logger is not None:
            self.logger._store_metadata(
                self.environment.time, self.environment.brokers, self.environment.syndicates, self.environment.reinsurancefirms, self.environment.shareholders, self.event_handler
            )

    def observe(self):
        """
        Get the current state of the Environment.

        Returns
        ----------
        Environment
        """

        return self.environment

    def get_time_to_next_event(self, event_type):
        """
        Get time to the next Event.

        Parameters
        ----------
        event_type: List[Type[Event]]
            Optional list of Event types to restrict to (i.e., skip over Event types not in the list).

        Returns
        ----------
        float
           Time [day] until the next upcoming Event.
        """

        # Sort upcoming events by start_time
        upcoming_events = list(self.event_handler.upcoming.values()) + list(self.event_handler.ongoing.values())
        upcoming_events.sort()

        # Filter for correct type of events if required
        if len(event_type) != 0:
            upcoming_events = [e for e in upcoming_events if isinstance(e, tuple(event_type))]

        # Check if there are any upcoming events left
        if len(upcoming_events) == 0:
            return None

        # Get time to next event
        next_event = upcoming_events[0]
        time_to_next_event = next_event.start_time - self.environment.time

        return time_to_next_event

    def _get_replay_status(self):
        """
        Extract status in last evolve() for Syndicates replayed from data.

        Returns
        ----------
        Dict[str, Syndicates]
            The status of all syndicates. Dictionary keys are Syndicate identifier.
        """

        syndicates_status = {}

        for event in self.event_handler.ongoing.values():
            if isinstance(event, UpdateSyndicateEvent):
                syndicates_status = event.get_synidate_status()

                if syndicates_status is not None:
                    syndicates_status[event.id] = syndicates_status
                    self.control_points[event.id].append(syndicates_status)

        return syndicates_status

    def evolve_action_syndicate(self, syndicates, step_time):
        """
        Evolve the specified Syndicate in the Environment for step_time [day].

        Parameters
        ----------
        syndicates: List[str]
            A list of Syndicate identifiers to evolve in time.
        step_time: float
            Amount of time in days to evolve the Environment for.

        Returns
        ----------
        Dict[str, Syndicate]
            The syndicate status during step_time. The dictionary keys are Syndicate identifiers.
        """

        syndicates_status = {}

        for syndicate in syndicates:

            syndicates_status = self.syndicates[syndicate].update_status(syndicate, self.environment, step_time, self.actions_to_apply)

        return syndicates_status

    def evolve(self, step_time):
        """
        Evolve the Environment: apply Events and update all agents status, including attritional loss event

        Parameters
        ----------
        step_time: float
            Amount of time [day] to evolve the Environment for.

        Returns
        ----------
        The updated Environment.
        """

        # Storage for all the syndicates' status
        syndicates_status = {}

        # The time the environment will have after being evolved
        env_start_time = self.environment.time
        env_end_time = self.environment.time + step_time

        # Enact the events
        self.environment = self.event_handler.forward(self.environment, step_time)

        # Track any newly-added broker_bring_risk events
        upcoming_broker_bring_risk = [
            e.broker_id for e in self.event_handler.upcoming.values() if isinstance(e, AddRiskEvent)
        ]
        newly_added_risk_events = {
            e.syndicate: e.start_time
            for e in self.event_handler.completed.values()
            if isinstance(e, AddRiskEvent) and (e.syndicate in upcoming_broker_bring_risk)
        }

        # Track any newly-added broker_bring_risk events
        upcoming_broker_bring_risk = [
            e.id for e in self.event_handler.upcoming.risk_values() if isinstance(e, AddRiskEvent)
        ]
        newly_added_risk_events = {
            e.id: e.start_time
            for e in self.event_handler.completed.risk_values()
            if isinstance(e, AddRiskEvent) and (e.id in upcoming_broker_bring_risk)
        }

        # Track any newly-added broker_bring_claim events
        upcoming_broker_bring_claim = [
            e.id for e in self.event_handler.upcoming.claim_values() if isinstance(e, AddClaimEvent)
        ]
        newly_added_claim_events = {
            e.id: e.start_time
            for e in self.event_handler.completed.claim_values()
            if isinstance(e, AddClaimEvent) and (e.id in upcoming_broker_bring_claim)
        }

        # Track any newly-added catastrophe events
        upcoming_catastrophe = [
            e.id for e in self.event_handler.upcoming.catastrophe_values() if isinstance(e, CatastropheEvent)
        ]
        newly_added_catastrophe_events = {
            e.id: e.start_time
            for e in self.event_handler.completed.catastrophe_values()
            if isinstance(e, CatastropheEvent) and (e.id in upcoming_catastrophe)
        }

        # Track any newly-added attritional loss events
        upcoming_attritionalloss = [
            e.id for e in self.event_handler.upcoming.attritionalloss_values() if isinstance(e, AttritionalLossEvent)
        ]
        newly_added_attritionalloss_events = {
            e.id: e.start_time
            for e in self.event_handler.completed.attritionalloss_values()
            if isinstance(e, AttritionalLossEvent) and (e.id in upcoming_attritionalloss)
        }

        events_start_times = np.array(
            [
                self.environment.time
                if risk_id not in newly_added_risk_events
                else newly_added_risk_events[risk_id]
                for risk_id in self.brokers.bring_risk()
            ]
        )

        for claim_id in self.brokers.bring_claim():
            if claim_id not in newly_added_claim_events:
                events_start_times.append(self.environment.time)
            else:
                events_start_times.(newly_added_claim_events[claim_id])

        for catastrophe_id in self.catastrophe():
            if catastrophe_id not in newly_added_catastrophe_events:
                events_start_times.append(self.environment.time)
            else:
                events_start_times.(newly_added_catastrophe_events[catastrophe_id])

        for attritionalloss_id in self.attritionalloss():
            if attritionalloss_id not in newly_added_attritionalloss_events:
                events_start_times.append(self.environment.time)
            else:
                events_start_times.(newly_added_attritionalloss_events[attritionalloss_id])

        # Get the unique start times and sort
        sorted_unique_start_times = np.sort(np.unique(events_start_times))

        # Update all the agents, run the event at the same start time
        for start_time in sorted_unique_start_times:
            # Move along the environment's time
            self.environment.time = start_time

            # Find all the events starting at this time

            # Risk events triger lead, follow line size of syndicates, accpted triger premium from broker

            # Catastrophe events triger claim

            # Claim events triger payment

            # Attritional Loss events triger loss

            # Events like interest receive and dividend payment

        # Finally, save issued Actions and move the environment time to the end time
        if len(self.actions_to_apply) > 0:
            self.actions_issued[env_start_time] = self.actions_to_apply

            # Empty all the actions to apply to syndicates
            self.actions_to_apply = []

        self.environment.time += 1

        return self.environment

    def syndicate_exit(self):
        """
        Syndicates exit the market because of 
        """

    def finished(self):
        """
        Determine whether the scenario is finished.
        Finished here means
        - there are no upcoming events
        - there aren't any Syndicates left in the Environment

        Returns
        ----------
        bool
        """

        # If there are any upcoming events or any syndicates active in the market,
        # then we aren't finished.
        any_events_left = len(self.event_handler.upcoming) > 0
        any_syndicate_left = len(self.environment.syndicate) > 0

        return not (any_events_left or any_syndicate_left)

    def receive_actions(self, actions):
        """
        Store Actions to apply when self.evolve() is called. Actions issued to Syndicate that exit the market are discarded.

        Parameters
        ----------
        actions: List[Action]
            List of Actions to issue to Syndicate.
        """

        all_syndicates = [action.syndicate for action in actions]
        exit_syndicates = [syndicate for syndicate in all_syndicates if syndicate not in self.environment.syndicate]
        active_syndicates = [syndicate for syndicate in all_syndicates if syndicate not in exit_syndicates]
        uncontrollable_syndicates = [
            syndicate for syndicate in active_syndicates if not self.environment.syndicates[syndicate].controllable
        ]

        # save Actions that cannot be issued and warn
        if len(exit_syndicates) > 0:
            warnings.warn(
                f"Cannot issue actions to {', '.join(exit_syndicates)} (syndicate not recognized)...", UserWarning
            )
        if len(uncontrollable_syndicates) > 0:
            warnings.warn(
                f"Cannot issue actions to {', '.join(uncontrollable_syndicates)} (Syndicate not controllable)...",
                UserWarning,
            )
        refused_actions = [
            action
            for action in actions
            if (action.syndicate in exit_syndicates) or (action.syndicate in uncontrollable_syndicates)
        ]

        # store the not allowed actions if there are any
        if len(refused_actions) > 0:
            self.actions_not_issued[self.environment.time] = refused_actions

        # save Actions to issue
        accept_actions = [action for action in actions if action not in refused_actions]
        self.actions_to_apply = accept_actions

    def log_data(self, file_prefix, environment, issued_actions, not_issued_actions
    ):
        """
        Save log of specified data to a file in json format.

        Parameters
        ----------
        file_prefix: str
            Prefix to include in the log file name
        environment: bool
            Include observation of the current environment
        issued_actions: bool
            Include all Actions requested by the manager and issued to Syndicate
        not_issued_actions: bool
            Include all Actions requested by the manager but not issued to Syndicate
        """

        if self.logger is None:
            warnings.warn("Logger has not been instantiated...", UserWarning)
        else:

            env = self.environment if environment else None
            iss_act = self.actions_issued if issued_actions else None
            not_iss_act = self.actions_not_issued if not_issued_actions else None

            self.logger.log_to_json(
                self.environment.time,
                file_prefix=file_prefix,
                environment=env,
                issued_actions=iss_act,
                not_issued_actions=not_iss_act,
            )

    def log_all_data(self, file_prefix):
        """
        Save all the data to file.

        Parameters
        ----------
        file_prefix: str
            Prefix to include in the log file name.
        """

        self.log_data(
            file_prefix,
            environment=True,
            issued_actions=True,
            not_issued_actions=True,
        )

