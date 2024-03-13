from __future__ import annotations
import warnings
from environment.event.add_catastrophe import AddCatastropheEvent
from environment.event.add_attritionalloss import AddAttritionalLossEvent
from environment.event.add_risk import AddRiskEvent
from environment.event.add_premium import AddPremiumEvent
from environment.event.add_claim import AddClaimEvent
import numpy as np
from environment.market import NoReinsurance_RiskOne, NoReinsurance_RiskFour, Reinsurance_RiskOne, Reinsurance_RiskFour
from manager.event_handler import EventHandler

class MarketManager:
    """
    Manage and evolve the market.
    """

    def __init__(self, maxstep, manager_args, brokers, syndicates, reinsurancefirms, shareholders, risks, risk_model_configs, with_reinsurance, num_risk_models, catastrophe_events, attritional_loss_events, broker_risk_events, broker_premium_events, broker_claim_events, event_handler, logger = None, time = 0):
        """
        Construct a new instance.

        Parameters
        ----------
        maxstep: int
            Simulation time span.
        manager_args
        brokers: list of Broker
        syndicates: list of Syndicate
        reinsurancefirms: list of Reinsurance Firms
        shareholders: list of sharholders
        risks: list of catastrophe
        risk_model_configs
        with_reinsurance: bool, True, involves reinsurance in the market, False, not involves reinsurance in the market
        num_risk_models: risk model arguments
        catastrophe_events: list of CatastropheEvent
        attritional_loss_events: list of AttritionalLossEvent
        broker_risk_events: list of AddRiskEvent
        broker_premium_events: list of AddPremiumEvent
        broker_claim_events: list of AddClaimEvent
        event_handler: EventHandler
            The EventHandler applies events to the internal Market of the EM.
        logger: Logger
            An optional Logger.
        time: int
            Market start time.
        """
        self.maxstep = maxstep
        self.manager_args = manager_args
        self.brokers = brokers
        self.syndicates = syndicates
        self.reinsurancefirms = reinsurancefirms
        self.shareholders = shareholders
        self.risks = risks
        self.risk_model_configs = risk_model_configs
        self.with_reinsurance = with_reinsurance
        self.num_risk_models = num_risk_models
        self.catastrophe_events = catastrophe_events
        self.attritional_loss_events = attritional_loss_events
        self.broker_risk_events = broker_risk_events
        self.broker_premium_events = broker_premium_events
        self.broker_claim_events = broker_claim_events
        self.event_handler = event_handler

        """
        if self.with_reinsurance == False:
            if self.num_risk_models == 1:
                self.market = NoReinsurance_RiskOne(time, self.maxstep, self.manager_args, self.brokers, self.syndicates, self.shareholders, self.risks, self.risk_model_configs, self.catastrophe_events, self.attritional_loss_events, self.broker_risk_events, self.broker_premium_events, self.broker_claim_events)
            else:
                self.market = NoReinsurance_RiskFour(time, self.maxstep, self.manager_args, self.brokers, self.syndicates, self.shareholders, self.risks, self.risk_model_configs, self.catastrophe_events, self.attritional_loss_events, self.broker_risk_events, self.broker_premium_events, self.broker_claim_events)
        else:
            if self.num_risk_models == 1:
                self.market = Reinsurance_RiskOne(time, self.maxstep, self.manager_args, self.brokers, self.syndicates, self.reinsurancefirms, self.shareholders, self.risks, self.risk_model_configs, self.catastrophe_events, self.attritional_loss_events, self.broker_risk_events, self.broker_premium_events, self.broker_claim_events)
            else:
                self.market = Reinsurance_RiskFour(time, self.maxstep, self.manager_args, self.brokers, self.syndicates, self.reinsurancefirms, self.shareholders, self.risks, self.risk_model_configs, self.catastrophe_events, self.attritional_loss_events, self.broker_risk_events, self.broker_premium_events, self.broker_claim_events)
        """

        self.market = NoReinsurance_RiskOne(time, self.maxstep, self.manager_args, self.brokers, self.syndicates, self.shareholders, self.risks, self.risk_model_configs, self.catastrophe_events, self.attritional_loss_events, self.broker_risk_events, self.broker_premium_events, self.broker_claim_events)

        self.min_step_time = 1  # Day Event

        self.actions_to_apply = []
        # For logging keep track of all Actions ever received and whether they were accepted or refused by the manager
        self.actions_accepted = {}
        self.actions_refused = {}

        # Logging
        self.logger = logger
        if self.logger is not None:
            self.logger._store_metadata(
                self.market.time, self.market.brokers, self.market.syndicates, self.market.reinsurancefirms, self.market.shareholders, self.event_handler
            )

    def observe(self):
        """
        Get the current state of the Market.

        Returns
        ----------
        Market
        """

        return self.market

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
        upcoming_catastrophe_events = list(self.event_handler.upcoming_catastrophe.values()) + list(self.event_handler.ongoing_catastrophe.values())
        upcoming_catastrophe_events.sort()
        upcoming_attritional_loss_events = list(self.event_handler.upcoming_attritional_loss.values()) + list(self.event_handler.ongoing_attritional_loss.values())
        upcoming_attritional_loss_events.sort()
        upcoming_broker_risk_events = list(self.event_handler.upcoming_broker_risk.values()) + list(self.event_handler.ongoing_broker_risk.values())
        upcoming_broker_risk_events.sort()
        upcoming_broker_premium_events = list(self.event_handler.upcoming_broker_premium.values()) + list(self.event_handler.ongoing_broker_premium.values())
        upcoming_broker_premium_events.sort()
        upcoming_broker_claim_events = list(self.event_handler.upcoming_broker_claim.values()) + list(self.event_handler.ongoing_broker_claim.values())
        upcoming_broker_claim_events.sort()

        # Check if there are any upcoming events left
        #if ((len(upcoming_catastrophe_events) == 0) and (len(upcoming_attritional_loss_events) == 0) and (len(upcoming_broker_risk_events) == 0)
            #and (len(upcoming_broker_premium_events) == 0) and (len(upcoming_broker_claim_events) == 0)):
            #return None

        # Get time to next event
        next_catastrophe_event = upcoming_catastrophe_events[0]
        time_to_next_catastrophe_event = next_catastrophe_event.start_time - self.market.time
        next_attritional_loss_event = upcoming_attritional_loss_events[0]
        time_to_next_attritional_loss_event = next_attritional_loss_event.start_time - self.market.time
        next_broker_risk_event = upcoming_broker_risk_events[0]
        time_to_next_broker_risk_event = next_broker_risk_event.start_time - self.market.time
        next_broker_premium_event = upcoming_broker_premium_events[0]
        time_to_next_broker_premium_event = next_broker_premium_event.start_time - self.market.time
        next_broker_claim_event = upcoming_broker_claim_events[0]
        time_to_next_broker_claim_event = next_broker_claim_event.start_time - self.market.time

        #return time_to_next_catastrophe_event, time_to_next_attritional_loss_event, time_to_next_broker_risk_event, time_to_next_broker_premium_event, time_to_next_broker_claim_event
        return time_to_next_broker_risk_event

    def _get_syndicate_replay_status(self):
        """
        Extract status in last evolve() for Syndicates replayed from data.

        Returns
        ----------
        Dict[str, Syndicates]
            The status of all syndicates. Dictionary keys are Syndicate identifier.
        """

        syndicates_status = {}

        for event in self.event_handler.ongoing.values():
            if isinstance(event, AddClaimEvent) or isinstance(event, AddPremiumEvent) or isinstance(event, AddAttritionalLossEvent):
                # Update capital list
                syndicates_status = event.get_synidate_status(self.syndicates)
                if syndicates_status is not None:
                    syndicates_status[event.risk_id] = syndicates_status

        return syndicates_status

    def evolve_action_market(self, starting_broker_risk):
        """
        Evolve the syndicate, broker, risk in the market for step_time [day].

        Parameters
        ----------
        starting_broker_risk: AddRiskEvent
            The current risk event.
        step_time: float
            Amount of time in days to evolve the Market for.
        """

        # Update the status of brokers and syndicates in the market
        risk_id = starting_broker_risk.risk_id
        broker_id = starting_broker_risk.broker_id
        risks = {"risk_id": starting_broker_risk.risk_id,
                "risk_start_time": starting_broker_risk.risk_start_time,
                "risk_factor": starting_broker_risk.risk_factor,
                "risk_category": starting_broker_risk.risk_category,
                "risk_value": starting_broker_risk.risk_value}
        if len(self.actions_to_apply) > 0:
            lead_syndicate_id = self.actions_to_apply[0].syndicate
            follow_syndicates_id = [self.actions_to_apply[i].syndicate for i in range(1,len(self.actions_to_apply))]
            premium = starting_broker_risk.risk_value # TODO: will be changed in the future
            self.market.brokers[broker_id].add_contract(risks, lead_syndicate_id, follow_syndicates_id, premium)
            self.market.syndicates[lead_syndicate_id].add_leader(risks, self.actions_to_apply[0].line_size, premium)
            self.market.syndicates[lead_syndicate_id].add_contract(risks, broker_id, premium)
            for sy in range(len(follow_syndicates_id)):
                self.market.syndicates[follow_syndicates_id[sy]].add_follower(risks, self.actions_to_apply[1+sy].line_size, premium)
                self.market.syndicates[follow_syndicates_id[sy]].add_contract(risks, broker_id, premium)
        else:
            self.market.brokers[0].not_underwritten_risk(risks)

    def evolve(self, step_time):
        """
        Evolve the Market: apply Events and update all agents status

        Parameters
        ----------
        step_time: float
            Amount of time [day] to evolve the Market for.

        Returns
        ----------
        The updated Market.
        """

        # Storage for all the syndicates' status
        syndicates_status = {}

        # The time the market will have after being evolved
        market_start_time = self.market.time
        market_end_time = self.market.time + step_time

        upcoming_broker_risk = [
            e.risk_id for e in self.event_handler.upcoming_broker_risk.values() if isinstance(e, AddRiskEvent)
        ]

        # Enact the events
        self.market = self.event_handler.forward(self.market, step_time)

        # Track any newly-added broker_risk events

        newly_added_broker_risk_events = {
            e.risk_id: e.risk_start_time
            for e in self.event_handler.completed_broker_risk.values()
            if isinstance(e, AddRiskEvent) and (e.risk_id in upcoming_broker_risk)
        }

        broker_risk_event_start_times = np.array(
            [
                newly_added_broker_risk_events.get(risk_id)
                for risk_id in upcoming_broker_risk
                if newly_added_broker_risk_events.get(risk_id) != None
            ]
        )

        # Get the unique start times and sort
        sorted_unique_start_times = np.sort(np.unique(broker_risk_event_start_times))

        # Update all the agents, run the event at the same start time
        for start_time in sorted_unique_start_times:
            # Move along the market's time
            self.market.time = start_time

            # Get all the events starting at this time
            starting_broker_risk = None
            for i in range(len(self.broker_risk_events)):
                if self.broker_risk_events[i].risk_start_time == start_time:
                    starting_broker_risk = self.broker_risk_events[i]

            # Move along the corresponding syndicates
            self.evolve_action_market(starting_broker_risk)

            # Empty all the actions to apply to syndicates
            self.actions_to_apply = []

        self.market.time = market_end_time

    def syndicate_exit(self):
        """
        Syndicates exit the market because of 
        """

    def finished(self):
        """
        Determine whether the scenario is finished.
        Finished here means
        - there are no upcoming events
        - there aren't any Syndicates left in the Market

        Returns
        ----------
        bool
        """

        # If there are any upcoming events or any syndicates active in the market,
        # then we aren't finished.
        any_events_left = len(self.event_handler.upcoming) > 0
        any_syndicate_left = len(self.market.syndicates) > 0

        return not (any_events_left or any_syndicate_left)

    def receive_actions(self, actions):
        """
        Store Actions to apply when self.evolve() is called. 
        Actions issued to Syndicate that exit the market are discarded 
        (only happens for centralised control).
        Syndicate who issues actions which cover the highest line size will be chosen as the leader
        and execute its action, the others will be assigned the rest until the whole size equals one.

        Parameters
        ----------
        actions: List[Action]
            List of Actions to issue to Syndicate.
        """

        # Choose the leader and save its action, the first syndicate with the highest line size wins 
        # TODO: will add selection algorithm in the future
        sum_line_size = 0
        for sy in range(len(self.market.syndicates)):
            sum_line_size += actions[sy].line_size
        
        if sum_line_size < 1:
            # Refuse the quote
            accept_actions = []
        else:
            # Accept the quote
            accept_actions = []
            line_size = 0
            syndicate_id = []
            for sy in range(len(self.market.syndicates)):
                if actions[sy].line_size > line_size:
                    line_size = actions[sy].line_size
                    syndicate_id.append(sy)

            accept_actions.append(actions[syndicate_id[0]])
            # Assign line size to the rest syndicates, FIFO
            rest_line_size = 1 - line_size
            while rest_line_size > 0:
                for sy in range(len(self.market.syndicates)):
                    if sy not in  syndicate_id:
                        if actions[sy].line_size > rest_line_size:
                            actions[sy].line_size = rest_line_size
                            accept_actions.append(actions[sy])
                            syndicate_id.append(sy)
                            break
                        else:
                            rest_line_size -= actions[sy].line_size
                            accept_actions.append(actions[sy])
                            syndicate_id.append(sy)
        # Save Actions to issue
        self.actions_to_apply = accept_actions

    def log_data(self, file_prefix, market, issued_actions, not_issued_actions
    ):
        """
        Save log of specified data to a file in json format.

        Parameters
        ----------
        file_prefix: str
            Prefix to include in the log file name
        market: bool
            Include observation of the current market
        issued_actions: bool
            Include all Actions requested by the manager and issued to Syndicate
        not_issued_actions: bool
            Include all Actions requested by the manager but not issued to Syndicate
        """

        if self.logger is None:
            warnings.warn("Logger has not been instantiated...", UserWarning)
        else:

            market = self.market if market else None
            iss_act = self.actions_issued if issued_actions else None
            not_iss_act = self.actions_not_issued if not_issued_actions else None

            self.logger.log_to_json(
                self.market.time,
                file_prefix=file_prefix,
                market=market,
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
            market=True,
            issued_actions=True,
            not_issued_actions=True,
        )

