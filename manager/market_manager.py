from __future__ import annotations
import warnings

import numpy as np
from agents import Broker, Syndicate, Shareholder, ReinsuranceFirm
from environment.market import NoReinsurance_RiskOne, NoReinsurance_RiskFour, Reinsurance_RiskOne, Reinsurance_RiskFour
from environment.event import CatastropheEvent, AttritionalLossEvent, AddRiskEvent, AddPremiumEvent, AddClaimEvent
from manager.event_handler import EventHandler

class MarketManager:
    """
    Manage and evolve the market.
    """

    def __init__(self, maxstep, manager_args, brokers, syndicates, reinsurancefirms, shareholders, risks, risk_model_configs, with_reinsurance, num_risk_models, 
                catastrophe_events, attritional_loss_events, broker_risk_events, broker_premium_events, broker_claim_events, event_handler, logger, time = 0):
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

        if self.with_reinsurance == False:
            if self.num_risk_models == 1:
                self.market = NoReinsurance_RiskOne(time, self.maxstep, self.manager_args, self.brokers, self.syndicates, self.shareholders, self.risks, self.risk_model_configs)
            else:
                self.market = NoReinsurance_RiskFour(time, self.maxstep, self.manager_args, self.brokers, self.syndicates, self.shareholders, self.risks, self.risk_model_configs)
        else:
            if self.num_risk_models == 1:
                self.market = Reinsurance_RiskOne(time, self.maxstep, self.manager_args, self.brokers, self.syndicates, self.reinsurancefirms, self.shareholders, self.risks, self.risk_model_configs)
            else:
                self.market = Reinsurance_RiskFour(time, self.maxstep, self.manager_args, self.brokers, self.syndicates, self.reinsurancefirms, self.shareholders, self.risks, self.risk_model_configs)

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
        if ((len(upcoming_catastrophe_events) == 0) and (len(upcoming_attritional_loss_events) == 0) and (len(upcoming_broker_risk_events) == 0)
            and (len(upcoming_broker_premium_events) == 0) and (len(upcoming_broker_claim_events) == 0)):
            return None

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

        return time_to_next_catastrophe_event, time_to_next_attritional_loss_event, time_to_next_broker_risk_event, time_to_next_broker_premium_event, time_to_next_broker_claim_event

    def _get_broker_replay_status(self):
        """
        Extract status in last evolve() for Broker replayed from data.

        Returns
        ----------
        Dict[str, Broker]
            The status of all brokers. Dictionary keys are Broker identifier.
        """

        brokers_status = {}
        
        for event in self.event_handler.ongoing.values():
            if isinstance(event, AAddRiskEvent) or isinstance(event, AddClaimEvent) or isinstance(event, AddPremiumEvent):
                # Update risk list and underwritten_contract list
                brokers_status = event.get_broker_status()
                if brokers_status is not None:
                    broker_status[event.id] = brokers_status

        return brokers_status
    
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
            if isinstance(event, AAddRiskEvent) or isinstance(event, AddClaimEvent) or isinstance(event, AddPremiumEvent) 
            or isinstance(event, CatastropheEvent) or isinstance(event, AttritionalLossEvent):
                # Update capital list
                syndicates_status = event.get_synidate_status()
                if syndicates_status is not None:
                    syndicates_status[event.id] = syndicates_status

        return syndicates_status

    def evolve_action_syndicate(self, syndicates, step_time):
        """
        Evolve the specified Syndicate in the Market for step_time [day].

        Parameters
        ----------
        syndicates: List[str]
            A list of Syndicate identifiers to evolve in time.
        step_time: float
            Amount of time in days to evolve the Market for.

        Returns
        ----------
        Dict[str, Syndicate]
            The syndicate status during step_time. The dictionary keys are Syndicate identifiers.
        """

        syndicates_status = {}

        for syndicate in syndicates:

            syndicates_status = self.syndicates[syndicate].update_status(syndicate, self.market, step_time, self.actions_to_apply)

        return syndicates_status

    def evolve(self, step_time):
        """
        Evolve the Market: apply Events and update all agents status, including attritional loss event

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

        # Enact the events
        self.market = self.event_handler.forward(self.market, step_time)

        # Track any newly-added catastrophe events
        upcoming_catastrophe = [
            e.id for e in self.event_handler.upcoming_catastrophe.values() if isinstance(e, CatastropheEvent)
        ]
        newly_added_catastrophe_events = {
            e.id: e.start_time
            for e in self.event_handler.completed_catastrophe.values()
            if isinstance(e, CatastropheEvent) and (e.id in upcoming_catastrophe)
        }

        # Track any newly-added attritional loss events
        upcoming_attritionalloss = [
            e.id for e in self.event_handler.upcoming_attritionalloss.values() if isinstance(e, AttritionalLossEvent)
        ]
        newly_added_attritionalloss_events = {
            e.id: e.start_time
            for e in self.event_handler.completed_attritionalloss.values()
            if isinstance(e, AttritionalLossEvent) and (e.id in upcoming_attritionalloss)
        }

        # Track any newly-added broker_bring_risk events
        upcoming_broker_bring_risk = [
            e.broker_id for e in self.event_handler.upcoming_broker_risk.values() if isinstance(e, AddRiskEvent)
        ]
        newly_added_risk_events = {
            e.syndicate: e.start_time
            for e in self.event_handler.completed_broker_risk.values()
            if isinstance(e, AddRiskEvent) and (e.syndicate in upcoming_broker_risk)
        }

        # Track any newly-added broker_pay_premium events
        upcoming_broker_pay_premium = [
            e.id for e in self.event_handler.upcoming_broker_premium.values() if isinstance(e, AddPremiumEvent)
        ]
        newly_added_premium_events = {
            e.id: e.start_time
            for e in self.event_handler.completed_broker_premium.values()
            if isinstance(e, AddPremiumEvent) and (e.id in upcoming_broker_premium)
        }

        # Track any newly-added broker_bring_claim events
        upcoming_broker_bring_claim = [
            e.id for e in self.event_handler.upcoming_broker_claim.values() if isinstance(e, AddClaimEvent)
        ]
        newly_added_claim_events = {
            e.id: e.start_time
            for e in self.event_handler.completed_broker_claim.values()
            if isinstance(e, AddClaimEvent) and (e.id in upcoming_broker_claim)
        }

        events_start_times = np.array(
            [
                self.market.time
                if risk_id not in newly_added_risk_events
                else newly_added_risk_events[risk_id]
                for risk_id in self.brokers.bring_risk()
            ]
        )

        for claim_id in self.brokers.bring_claim():
            if claim_id not in newly_added_claim_events:
                events_start_times.append(self.market.time)
            else:
                events_start_times.(newly_added_claim_events[claim_id])

        for catastrophe_id in self.catastrophe():
            if catastrophe_id not in newly_added_catastrophe_events:
                events_start_times.append(self.market.time)
            else:
                events_start_times.(newly_added_catastrophe_events[catastrophe_id])

        for attritionalloss_id in self.attritionalloss():
            if attritionalloss_id not in newly_added_attritionalloss_events:
                events_start_times.append(self.market.time)
            else:
                events_start_times.(newly_added_attritionalloss_events[attritionalloss_id])

        # Get the unique start times and sort
        sorted_unique_start_times = np.sort(np.unique(events_start_times))

        # Update all the agents, run the event at the same start time
        for start_time in sorted_unique_start_times:
            # Move along the market's time
            self.market.time = start_time

            # Find all the events starting at this time

            # Risk events triger lead, follow line size of syndicates, accpted triger premium from broker

            # Catastrophe events triger claim

            # Claim events triger payment

            # Attritional Loss events triger loss

            # Events like interest receive and dividend payment

        # Finally, save issued Actions and move the market time to the end time
        if len(self.actions_to_apply) > 0:
            self.actions_issued[market_start_time] = self.actions_to_apply

            # Empty all the actions to apply to syndicates
            self.actions_to_apply = []

        self.market.time += 1

        return self.market


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
        any_syndicate_left = len(self.market.syndicate) > 0

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
        exit_syndicates = [syndicate for syndicate in all_syndicates if syndicate not in self.market.syndicate]
        active_syndicates = [syndicate for syndicate in all_syndicates if syndicate not in exit_syndicates]
        uncontrollable_syndicates = [
            syndicate for syndicate in active_syndicates if not self.market.syndicates[syndicate].controllable
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
            self.actions_not_issued[self.market.time] = refused_actions

        # save Actions to issue
        accept_actions = [action for action in actions if action not in refused_actions]
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

    def iterate(self,t):
        """
        Can be merged with evolve function
        """
        # Set up risk categories
        self.riskcategories = list(range(risk_args["num_categories"]))
        self.rc_event_schedule = []
        self.rc_event_damage = []
        self.rc_event_schedule_initial = []
        self.rc_event_damage_initial = []
        if rc_event_schedule is not None and rc_event_damage is not None:
            self.rc_event_schedule = copy.copy(rc_event_schedule)
            self.rc_event_schedule_initial = copy.copy(rc_event_schedule)
            self.rc_event_damage = copy.copy(rc_event_damage)
            self.rc_event_damage_initial = copy.copy(rc_event_damage)
        else:
            self.setup_risk_categories_caller()

        # Set up monetary system can be set in broker
        self.obligations = []

        self.risks_counter = [0,0,0,0]

        for item in self.risks:
            self.risks_counter[item["category"]] = self.risks_counter[item["category"]] + 1

        # Adjust market premiums
        sum_capital = sum([agent.get_cash() for agent in self.syndicates])
        self.adjust_market_premium(capital=sum_capital)
        sum_capital = sum([agent.get_cash() for agent in self.reinsurancefirms])
        self.adjust_reinsurance_market_premium(capital=sum_capital)

        # Pay obligations
        self.effect_payments(t)

        # Identify perils and effect claims
        for categ_id in range(len(self.rc_event_schedule)):
            try:
                if len(self.rc_event_schedule[categ_id]) > 0:
                    assert self.rc_event_schedule[categ_id][0] >= t
            except:
                print("Somthing wrong, past events not deleted")
            if len(self.rc_event_schedule[categ_id]) > 0 and self.rc_event_schedule[categ_id][0] == t:
                self.rc_event_schedule[categ_id] = self.rc_event_schedule[categ_id][1:]
            else:
                if isleconfig.verbose:
                    print("Next peril", self.rc_event_schedule[categ_id])

        # Shuffle risks
        self.shuffle_risks()

        # Reset weights
        self.reset_syndicate_weights()

        # Iterate syndicate agents
        for agent in self.syndicates:
            agent.iterate(t)

        self.syndicate_models_counter = np.zeros(risk_args["num_categories"])

        for syndicate in self.syndicates:
            for i in range(len(self.inaccuracy)):
                if syndicate.operational:
                    if syndicate.riskmodel.inaccuracy == self.inaccuracy[i]:
                        self.syndicate_models_counter[i] += 1

    def save_data(self):
        """
        Collect statistics about the current state of the simualtion, will pass these to the logger
        """

        # Collect data
        total_cash_no = sum([syndicate.cash for syndicate in self.syndicates])
        total_excess_capital = sum([syndicate.get_excess_capital() for syndicate in self.syndicates])
        total_profitslosses = sum([syndicate.get_profitslosses() for syndicate in self.syndicates])
        total_contracts_no = sum([len(syndicate.underwritten_contracts) for syndicate in self.syndicates])
        operational_no = sum([syndicate.operational for syndicate in self.syndicates])

        # Collect agent level data
        syndicates = [(syndicate.cash, syndicate.id, syndicate.operational) for syndicate in self.syndicates]

        # Prepare dict
        current_log = {}
        current_log["total_cash"] = total_cash_no
        current_log["total_excess_capital"] = total_excess_capital
        current_log["total_profitslosses"] = total_profitslosses
        current_log["total_contracts"] = total_contracts_no
        current_log["total_operational"] = operational_no
        current_log["market_premium"] = self.market_premium
        current_log["cumulative_bankruptcies"] = self.cumulative_bankruptcies
        current_log["cumulative_market_exits"] = self.cumulative_market_exits
        current_log["cumulative_unrecovered_claims"] = self.cumulative_unrecovered_claims
        current_log["cumulative_claims"] = self.cumulative_claims

        # Add agent level data to dict
        current_log["syndicate_cash"] = syndicates
        current_log["market_diffvar"] = self.compute_market_diffvar()

        current_log["individual_contracts"] = []
        individual_contracts_no = [len(syndicate.underwritten_contracts) for syndicate in self.syndicates]
        for i in range(len(individual_contracts_no)):
            current_log["individual_contracts"].append(individual_contracts_no[i])

        # Call to logger object
        self.logger.record_data(current_log)

    def obtain_log(self, requested_logs = None):
        """
        Return in a list all the data generated by the model
        """
        return self.logger.obtain_log(requested_logs)

    def advance_round(self, *args):
        pass

    def finalize(self, *args):
        pass

    def inflict_peril(self, categ_id, damage, t):
        affected_contracts = [contract for syndicate in self.syndicates for contract in syndicate.underwritten_contracts if contract.category==categ_id]
        damagevalues = np.random.beta(1, 1./damage-1, size=self.risks_counter[categ_id])
        uniformvalues = np.random.uniform(0, 1, size=self.risks_counter[categ_id])
        [contract.explode(t, uniformvalues[i], damagevalues[i]) for i, contract in enumerate(affected_contracts)]

    def receive_obligation(self, amount, recipient, due_time, purpose):
        obligation = {"amount": amount, "recipient": recipient, "due_time": due_time, "purpose": purpose}
        self.obligations.append(obligation)

    def effect_payments(self, time):
        due = [item for item in self.obligations if item["due_time"]<=time]
        self.obligations = [item for item in self.obligations if item["due_time"]>time]
        sum_due = sum([item["amount"] for item in due])
        for obligation in due:
            self.pay(obligation)

    def pay(self, obligation):
        amount = obligation["amount"]
        recipient = obligation["recipient"]
        purpose = obligation["purpose"]

        try:
            assert self.money_supply > amount
        except:
            print("Something wrong: economy out of money")
        if self.get_operational() and recipient.get_operational():
            self.money_supply -= amount
            recipient.receive(amount)

    def receive(self, amount):
        """
        Accept cash payment
        """
        self.money_supply += amount

    def reduce_money_supply(self, amount):
        """
        Reduce money supply immediately and without payment recipient
        """
        self.money_supply -= amount
        assert self.mooney_supply >= 0

    def reset_syndicate_weights(self):
        operational_no = sum([syndicate.operational for syndicate in self.syndicates])
        operational_firms = [syndicate for syndicate in self.syndicates if syndicate.operational]
        risk_no = len(self.risks)
        self.syndicates_weights = {}

        for syndicate in self.syndicates:
            self.syndicates_weights[syndicate.id] = 0

        if operational_no > 0:
            if risks_no / operational_no > 1:
                weights = risks_no / operational_no
                for syndicate in self.syndicates:
                    self.syndicates_weights[syndicate.id] = math.floor(weights)
            else:
                for i in range(len(self.risks)):
                    s = math.floor(np.random.uniform(0, len(operational_firms), 1))
                    self.syndicates_weights[operational_firms[s].id] += 1

    def shuffle_risks(self):
        np.random.shuffle(self.risks)

    def adjust_market_premium(self, capital):
        """
        Adjust the premium charged by syndicates for the risks covered. The premium reduces linearly with the capital available in the insurance market, the premium reduces until the minimum below which no synidaicate is willing to reduce the price

        Parameters
        ----------
        capital: float
            The total capital available in the insurance market
        """

        self.market_premium = self.norm_premium * (syndicate_args["upper_premium_limit"] - syndicate_args["premium_sensitivity"] * capital / (syndicate_args["initial_capital"] * self.damage_distribution.mean() * risk_args["num_risks"]))

        if self.market_premium < self.norm_premium * syndicate_args["lower_premium_limit"]:
            self.market_premium = self.norm_premium * syndicate_args["lower_premium_limit"]

    def get_market_premium(self):

        return self.market_premium

    def solicit_insurance_requests(self, id, cash, syndicate):
        risks_to_be_sent = self.risks[:int(self.syndicates_weights[syndicate.id])]
        self.risks = self.risks[int(self.syndicates_weights[syndicate.id]):]
        for risk in syndicate.risks_kept:
            risks_to_be_sent.append(risk)

        syndicate.risks_kept = []
        np.random.shuffle(risks_to_be_sent)

        return risks_to_be_sent

    def return_risks(self, not_accepted_risks):
        self.risks += not_accepted_risks

    def get_all_riskmodel_combinations(self, n, rm_factor):
        riskmodels = []
        for i in range(risk_args["num_categories"]):
            riskmodel_combination = rm_factor * np.ones(risk_args["num_categories"])
            riskmodel_combination[i] = 1/rm_factor
            riskmodels.append(tiskmodel_combination.tolist())

        return riskmodels

    def setup_risk_categories(self):
        for i in self.riskcategories:
            event_schedule = []
            event_damage = []
            total = 0
            while (total < sim_args["max_time"]):
                separation_time = self.catastrophe_separation_distribution.rvs()
                total += int(math.ceil(separation_time))
                if total < sim_args["max_time"]:
                    event_schedule.append(total)
                    event_damage.append(self.damage_distribution.rvs())
            self.rc_event_schedule.append(event_schedule)
            self.rc_event_damage.append(event_damage)
        self.rc_event_schedule_initial = copy.copy(self.rc_event_schedule)
        self.rc_event_damage_initial = copy.copy(self.rc_event_damage)

    def record_bankruptcy(self):
        """
        When a firm files for bankruptcy
        """
        self.cumulative_bankruptcies += 1

    def record_market_exit(self):
        """
        Record the firms that leave the market due to underperforming conditions
        """
        self.cumulative_market_exits += 1

    def record_unrecovered_claims(self, loss):
        self.cumulative_uncovered_claims += loss

    def record_claims(self, claims):
        """
        Record every claim made to syndicates and reinsurancefirms
        """
        self.cumulative_claims += claims

    def log(self):
        self.logger.save_log(self.background_run)

    def compute_market_diffvar(self):
        varsfirms = []
        for syndicate in self.syndicates:
            if syndicate.operational:
                varsfirms.append(syndicate.var_counter_per_risk)
        totalina = sum(varsfirms)

        varsfirms = []
        for syndicate in self.syndicates:
            if syndicate.operational:
                varsfirms.append(1)
        totalreal = sum(varsfirms)

        totaldiff = totalina - totalreal

        return totaldiff

    def count_underwritten_and_reinsured_risks_by_category(self):
        underwritten_risks = 0
        reinsured_risks = 0
        underwritten_per_category = np.zeros(risk_args["num_categories"])
        reinsured_per_category = np.zeros(risk_args["num_categories"])
        for syndicate in self.syndicates:
            if syndicate.operational:
                underwritten_by_category += syndicate.counter_category
                if risk_args["simulation_reinsurance_type"] == "non-proportional":
                    reinsured_per_category += syndicate.counter_category * syndicate.category_reinsurance
        if risk_args["simulation_reinsurance_type"] == "proportional":
            for syndicate in self.syndicates:
                if syndicate.operational:
                    reinsured_per_category += syndicate.counter_category

    def get_unique_syndicate_id(self):
        current_id = self.syndicate_id_counter
        self.syndicate_id_counter += 1
        return current_id

    def syndicate_entry_index(self):
        return self.syndicate_models_counter[0:risk_args["num_riskmodels"]].argmin()

    def get_operational(self):
        return True

    def reset_pls(self):
        """
        Reset all the profits and losses of all syndicates, reinsurance firms
        """
        for syndicate in self.syndicates:
            syndiacte.reset_pl()


