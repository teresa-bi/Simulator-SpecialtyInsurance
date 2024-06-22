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

    def __init__(self, maxstep, sim_args, manager_args, syndicate_args, brokers, syndicates, reinsurancefirms, shareholders, catastrophes, fair_market_premium, risk_model_configs, with_reinsurance, num_risk_models, 
                 catastrophe_events, attritional_loss_events, broker_risk_events, broker_premium_events, broker_claim_events, event_handler, logger = None, time = 0):
        self.maxstep = maxstep
        self.sim_args = sim_args
        self.manager_args = manager_args
        self.syndicate_args = syndicate_args
        self.brokers = brokers
        self.syndicates = syndicates
        self.lead_line_size = self.syndicate_args["lead_line_size"]
        self.follow_line_sizes = self.syndicate_args["follow_line_size"]
        self.reinsurancefirms = reinsurancefirms
        self.shareholders = shareholders
        self.catastrophes = catastrophes
        self.fair_market_premium = fair_market_premium
        self.risk_model_configs = risk_model_configs
        self.with_reinsurance = with_reinsurance
        self.num_risk_models = num_risk_models
        self.catastrophe_events = catastrophe_events
        self.attritional_loss_events = attritional_loss_events
        self.broker_risk_events = broker_risk_events
        self.broker_premium_events = broker_premium_events
        self.broker_claim_events = broker_claim_events
        self.event_handler = event_handler

        self.market = NoReinsurance_RiskOne(time, self.maxstep, self.manager_args, self.brokers, self.syndicates, self.shareholders, self.catastrophes, self.risk_model_configs, 
                                            self.catastrophe_events, self.attritional_loss_events, self.broker_risk_events, self.broker_premium_events, self.broker_claim_events)

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

    def evolve_action_market(self, starting_broker_risk, time):
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
        num_risk = len(self.actions_to_apply)
        for num in range(num_risk):
            broker_id = starting_broker_risk[num].broker_id
            risks = {"risk_id": starting_broker_risk[num].risk_id,
                "risk_start_time": starting_broker_risk[num].risk_start_time,
                "risk_end_time": starting_broker_risk[num].risk_end_time+self.sim_args["mean_contract_runtime"],
                "risk_factor": starting_broker_risk[num].risk_factor,
                "risk_category": starting_broker_risk[num].risk_category,
                "risk_value": starting_broker_risk[num].risk_value}
            if len(self.actions_to_apply[num]) == 0:
                self.market.brokers[int(broker_id)].not_underwritten_risk(risks)
            elif len(self.actions_to_apply[num]) > 0:
                lead_syndicate_id = self.actions_to_apply[num][0].syndicate
                lead_syndicate_premium = self.actions_to_apply[num][0].premium * self.lead_line_size * risks["risk_value"] / 30
                premium = lead_syndicate_premium
                follow_syndicates_id = [None for i in range(len(self.market.syndicates))]
                follow_syndicates_premium = [None for i in range(len(self.market.syndicates))]
                for i in range(1,len(self.actions_to_apply[num])):
                    follow_syndicates_id[i-1] = self.actions_to_apply[num][i].syndicate
                    follow_syndicates_premium[i-1] = self.actions_to_apply[num][i].premium * self.follow_line_sizes * risks["risk_value"] / 30
                    premium += follow_syndicates_premium[i-1]
                self.market.brokers[int(broker_id)].add_contract(risks, lead_syndicate_id, self.lead_line_size, lead_syndicate_premium, follow_syndicates_id, self.follow_line_sizes, follow_syndicates_premium, premium)
                self.market.syndicates[int(lead_syndicate_id)].add_leader(risks, self.lead_line_size, lead_syndicate_premium)
                self.market.syndicates[int(lead_syndicate_id)].add_contract(risks, broker_id, lead_syndicate_premium)
                for sy in range(len(follow_syndicates_id)):
                    if follow_syndicates_id[sy] != None:
                        self.market.syndicates[int(follow_syndicates_id[sy])].add_follower(risks, self.follow_line_sizes, follow_syndicates_premium[sy])
                        self.market.syndicates[int(follow_syndicates_id[sy])].add_contract(risks, broker_id, follow_syndicates_premium[sy])
        # Update syndicates' status and contracts
        for broker_id in range(len(self.market.brokers)):
            num = 0
            while num < len(self.market.brokers[broker_id].underwritten_contracts):
                if self.market.brokers[broker_id].underwritten_contracts[num]["risk_end_time"] < time:
                    lead_syndicate_id = self.market.brokers[broker_id].underwritten_contracts[num]["lead_syndicate_id"]
                    follow_syndicates_id = self.market.brokers[broker_id].underwritten_contracts[num]["follow_syndicates_id"]
                    i = 0
                    while i < len(self.market.syndicates[int(lead_syndicate_id)].current_hold_contracts):
                        if (self.market.syndicates[int(lead_syndicate_id)].current_hold_contracts[i]["broker_id"] == broker_id) and (self.market.syndicates[int(lead_syndicate_id)].current_hold_contracts[i]["risk_end_time"] < time):
                            del self.market.syndicates[int(lead_syndicate_id)].current_hold_contracts[i]
                        else:
                            i += 1
                    for j in range(len(follow_syndicates_id)):
                        if follow_syndicates_id[j] != None:
                            k = 0
                            while k < len(self.market.syndicates[int(follow_syndicates_id[j])].current_hold_contracts):
                                if (self.market.syndicates[int(follow_syndicates_id[j])].current_hold_contracts[k]["broker_id"] == broker_id) and (self.market.syndicates[int(follow_syndicates_id[j])].current_hold_contracts[k]["risk_end_time"] < time):
                                    del self.market.syndicates[int(follow_syndicates_id[j])].current_hold_contracts[k]
                                else:
                                    k += 1
                    del self.market.brokers[broker_id].underwritten_contracts[num]
                else:
                    num += 1

        # Update Syndicates status and capital from interest
        for i in range(len(self.market.syndicates)):
            if self.market.syndicates[i].status == True:
                amount = self.market.syndicates[i].current_capital * self.syndicate_args["interest_rate"]
                self.market.syndicates[i].current_capital += amount
                for j in range(len(self.market.syndicates[i].current_capital_category)):
                    self.market.syndicates[i].current_capital_category[j] += amount / len(self.market.syndicates[i].current_capital_category)
        if time % 12 == 0:
            for i in range(len(self.market.syndicates)):
                if self.market.syndicates[i].status == True:
                    if self.market.syndicates[i].current_capital > self.market.syndicates[i].initial_capital:
                        self.market.syndicates[i].current_capital = self.market.syndicates[i].current_capital - (self.market.syndicates[i].current_capital - self.market.syndicates[i].initial_capital) * self.market.syndicates[i].dividend_share_of_profits
                    self.market.syndicates[i].market_permanency()
                    
    def run_attritional_loss(self, starting_attritional_loss):
        """
        Update market with attritional loss event

        Parameters
        ----------
        starting_attritional_loss: AddAttritionalLossEvent
            The current attritional loss event
        """
        for i in range(len(self.market.syndicates)):
            if self.market.syndicates[i].status == True:
                self.market.syndicates[i].current_capital -= starting_attritional_loss.risk_value * 0.0001
                self.market.syndicates[i].current_capital -= (self.market.syndicates[i].initial_capital-self.market.syndicates[i].current_capital) * self.market.syndicates[i].cost_of_capital
                self.market.syndicates[i].profits_losses -= starting_attritional_loss.risk_value * 0.0001

    def run_broker_premium(self, starting_broker_premium):
        """
        Update market with premium event

        Parameters
        ----------
        starting_broker_premium: AddPremiumEvent
            The current premium event
        """
        for broker_id in range(len(self.market.brokers)):
            affected_contract = []
            for num in range(len(self.market.brokers[broker_id].underwritten_contracts)):
                if self.market.brokers[broker_id].underwritten_contracts[num]["risk_end_time"] >= starting_broker_premium.risk_start_time:
                    affected_contract.append(self.market.brokers[broker_id].underwritten_contracts[num])
            for num in range(len(affected_contract)):
                premium, lead_syndicate_premium, follow_syndicates_premium, lead_syndicate_id, follow_syndicates_id, risk_category = self.market.brokers[broker_id].pay_premium(affected_contract[num])
                if self.market.syndicates[int(lead_syndicate_id)].status == True:
                    self.market.syndicates[int(lead_syndicate_id)].receive_premium(lead_syndicate_premium, risk_category)
                    self.market.syndicates[int(lead_syndicate_id)].current_capital += lead_syndicate_premium
                    self.market.syndicates[int(lead_syndicate_id)].current_capital_category[risk_category] += lead_syndicate_premium
                    self.market.syndicates[int(lead_syndicate_id)].profits_losses += lead_syndicate_premium
                for follow_id in range(len(follow_syndicates_id)):
                    if (follow_syndicates_id[follow_id] != None) and (self.market.syndicates[int(follow_syndicates_id[follow_id])].status == True):
                        self.market.syndicates[int(follow_syndicates_id[follow_id])].receive_premium(follow_syndicates_premium[follow_id], risk_category)
                        self.market.syndicates[int(follow_syndicates_id[follow_id])].current_capital += follow_syndicates_premium[follow_id]
                        self.market.syndicates[int(follow_syndicates_id[follow_id])].current_capital_category[risk_category] += follow_syndicates_premium[follow_id]
                        self.market.syndicates[int(follow_syndicates_id[follow_id])].profits_losses += follow_syndicates_premium[follow_id]

    def run_broker_claim(self, starting_broker_claim):
        """
        Update market with claim event

        Parameters
        ----------
        starting_broker_claim: AddClaimEvent
            The current claim event
        """
        for broker_id in range(len(self.market.brokers)):
            affected_contract = []
            for num in range(len(self.market.brokers[broker_id].underwritten_contracts)):
                if self.market.brokers[broker_id].underwritten_contracts[num]["risk_end_time"] == starting_broker_claim.risk_start_time+1:
                    affected_contract.append(self.market.brokers[broker_id].underwritten_contracts[num])
            """
            for num in range(len(affected_contract)):
                claim, lead_syndicate_id, follow_syndicates_id, risk_category, lead_claim_value, follow_claim_values = self.market.brokers[broker_id].end_contract_ask_claim(affected_contract[num])
                if self.market.syndicates[int(lead_syndicate_id)].current_capital >= lead_claim_value:
                    # TODO: now pay claim according to broker id, can add other mechanism in the future
                    self.market.syndicates[int(lead_syndicate_id)].pay_claim(broker_id, risk_category, lead_claim_value)
                    self.market.syndicates[int(lead_syndicate_id)].profits_losses -= lead_claim_value
                    self.market.syndicates[int(lead_syndicate_id)].current_capital -= lead_claim_value
                    self.market.syndicates[int(lead_syndicate_id)].current_capital_category[risk_category] -= lead_claim_value
                    self.market.brokers[broker_id].receive_claim(lead_syndicate_id, risk_category, lead_claim_value, lead_claim_value)
                else:
                    if self.market.syndicates[int(lead_syndicate_id)].status == True:
                        self.market.syndicates[int(lead_syndicate_id)].pay_claim(broker_id, risk_category, lead_claim_value)
                        self.market.syndicates[int(lead_syndicate_id)].current_capital -= lead_claim_value
                        self.market.syndicates[int(lead_syndicate_id)].current_capital_category[risk_category] -= lead_claim_value
                        self.market.syndicates[int(lead_syndicate_id)].profits_losses -= lead_claim_value
                        self.market.brokers[broker_id].receive_claim(lead_syndicate_id, risk_category, lead_claim_value, self.market.syndicates[int(lead_syndicate_id)].current_capital)
                        self.market.syndicates[int(lead_syndicate_id)].bankrupt() 
                        self.market.syndicates[int(lead_syndicate_id)].excess_capital = 0
                for follow_num in range(len(follow_syndicates_id)): 
                    if follow_syndicates_id[follow_num] != None: 
                        if self.market.syndicates[int(follow_syndicates_id[follow_num])].current_capital >= follow_claim_values[follow_num]:
                            self.market.syndicates[int(follow_syndicates_id[follow_num])].pay_claim(broker_id, risk_category, follow_claim_values[follow_num])
                            self.market.syndicates[int(follow_syndicates_id[follow_num])].current_capital -= follow_claim_values[follow_num]
                            self.market.syndicates[int(follow_syndicates_id[follow_num])].current_capital_category[risk_category] -= follow_claim_values[follow_num]
                            self.market.syndicates[int(follow_syndicates_id[follow_num])].profits_losses -= follow_claim_values[follow_num]
                            self.market.brokers[broker_id].receive_claim(follow_syndicates_id[follow_num], risk_category, follow_claim_values[follow_num], follow_claim_values[follow_num])
                        else:
                            if self.market.syndicates[int(follow_syndicates_id[follow_num])].status == True:
                                self.market.syndicates[int(follow_syndicates_id[follow_num])].pay_claim(broker_id, risk_category, follow_claim_values[follow_num])
                                self.market.syndicates[int(follow_syndicates_id[follow_num])].current_capital -= follow_claim_values[follow_num]
                                self.market.syndicates[int(follow_syndicates_id[follow_num])].current_capital_category[risk_category] -= follow_claim_values[follow_num]
                                self.market.syndicates[int(follow_syndicates_id[follow_num])].profits_losses -= follow_claim_values[follow_num]
                                self.market.brokers[broker_id].receive_claim(follow_syndicates_id[follow_num], risk_category, follow_claim_values[follow_num], self.market.syndicates[int(follow_syndicates_id[follow_num])].current_capital)
                                self.market.syndicates[int(follow_syndicates_id[follow_num])].bankrupt()
                                self.market.syndicates[int(follow_syndicates_id[follow_num])].excess_capital = 0"""

    def run_catastrophe(self, starting_catastrophe):
        """
        Update market with catastrophe event

        Parameters
        ----------
        starting_catastrophe: AddCatastropheEvent
            The current catastrophe event
        """
        # Catastrophe will influce the broker claim
        for broker_id in range(len(self.market.brokers)):
            affected_contract = []
            for num in range(len(self.market.brokers[broker_id].underwritten_contracts)):
                if (starting_catastrophe.catastrophe_category - self.market.brokers[broker_id].underwritten_contracts[num]["risk_category"] == 0):
                #if starting_catastrophe.catastrophe_category - self.market.brokers[broker_id].underwritten_contracts[num]["risk_category"] <= 1:
                    affected_contract.append(self.market.brokers[broker_id].underwritten_contracts[num])
            for num in range(len(affected_contract)):
                claim, lead_syndicate_id, follow_syndicates_id, catastrophe_category, lead_claim_value, follow_claim_values = self.market.brokers[broker_id].end_contract_ask_claim(affected_contract[num])
                lead_claim_value *= starting_catastrophe.catastrophe_value
                if self.market.syndicates[int(lead_syndicate_id)].current_capital >= lead_claim_value:
                    # TODO: now pay claim according to broker id, can add other mechanism in the future
                    self.market.syndicates[int(lead_syndicate_id)].pay_claim(broker_id, catastrophe_category, lead_claim_value)
                    self.market.syndicates[int(lead_syndicate_id)].current_capital -= lead_claim_value
                    self.market.syndicates[int(lead_syndicate_id)].current_capital_category[starting_catastrophe.catastrophe_category] -= lead_claim_value
                    self.market.syndicates[int(lead_syndicate_id)].profits_losses -= lead_claim_value
                    self.market.brokers[broker_id].receive_claim(lead_syndicate_id, catastrophe_category, lead_claim_value, lead_claim_value)
                else:
                    if self.market.syndicates[int(lead_syndicate_id)].status == True:
                        self.market.syndicates[int(lead_syndicate_id)].pay_claim(broker_id, catastrophe_category, lead_claim_value)
                        self.market.syndicates[int(lead_syndicate_id)].current_capital -= lead_claim_value
                        self.market.syndicates[int(lead_syndicate_id)].current_capital_category[starting_catastrophe.catastrophe_category] -= lead_claim_value
                        self.market.syndicates[int(lead_syndicate_id)].profits_losses -= lead_claim_value
                        self.market.brokers[broker_id].receive_claim(lead_syndicate_id, catastrophe_category, lead_claim_value, self.market.syndicates[int(lead_syndicate_id)].current_capital)
                        self.market.syndicates[int(lead_syndicate_id)].bankrupt() 
                        self.market.syndicates[int(lead_syndicate_id)].excess_capital = 0
                for follow_num in range(len(follow_syndicates_id)): 
                    if follow_syndicates_id[follow_num] != None: 
                        follow_claim_values[follow_num] *= starting_catastrophe.catastrophe_value
                        if self.market.syndicates[int(follow_syndicates_id[follow_num])].current_capital >= follow_claim_values[follow_num]:
                            self.market.syndicates[int(follow_syndicates_id[follow_num])].pay_claim(broker_id, catastrophe_category, follow_claim_values[follow_num])
                            self.market.syndicates[int(follow_syndicates_id[follow_num])].current_capital -= follow_claim_values[follow_num]
                            self.market.syndicates[int(follow_syndicates_id[follow_num])].current_capital_category[starting_catastrophe.catastrophe_category] -= follow_claim_values[follow_num]
                            self.market.syndicates[int(follow_syndicates_id[follow_num])].profits_losses -= follow_claim_values[follow_num]
                            self.market.brokers[broker_id].receive_claim(follow_syndicates_id[follow_num], catastrophe_category, follow_claim_values[follow_num], follow_claim_values[follow_num])
                        else:
                            if self.market.syndicates[int(follow_syndicates_id[follow_num])].status == True:
                                self.market.syndicates[int(follow_syndicates_id[follow_num])].pay_claim(broker_id, catastrophe_category, follow_claim_values[follow_num])
                                self.market.syndicates[int(follow_syndicates_id[follow_num])].current_capital -= follow_claim_values[follow_num]
                                self.market.syndicates[int(follow_syndicates_id[follow_num])].current_capital_category[starting_catastrophe.catastrophe_category] -= follow_claim_values[follow_num]
                                self.market.syndicates[int(follow_syndicates_id[follow_num])].profits_losses -= follow_claim_values[follow_num]
                                self.market.brokers[broker_id].receive_claim(follow_syndicates_id[follow_num], catastrophe_category, follow_claim_values[follow_num], self.market.syndicates[int(follow_syndicates_id[follow_num])].current_capital)
                                self.market.syndicates[int(follow_syndicates_id[follow_num])].bankrupt()
                                self.market.syndicates[int(follow_syndicates_id[follow_num])].excess_capital = 0

    def evolve(self, step_time):

        # Storage for all the syndicates' status
        syndicates_status = {}

        # The time the market will have after being evolved
        market_start_time = self.market.time
        market_end_time = self.market.time + step_time

        upcoming_catastrophe = [
            e.catastrophe_id for e in self.event_handler.upcoming_catastrophe.values() if isinstance(e, AddCatastropheEvent)
        ]

        upcoming_attritional_loss = [
            e.risk_id for e in self.event_handler.upcoming_attritional_loss.values() if isinstance(e, AddAttritionalLossEvent)
        ]

        upcoming_broker_risk = [
            e.risk_id for e in self.event_handler.upcoming_broker_risk.values() if isinstance(e, AddRiskEvent)
        ]

        upcoming_broker_premium = [
            e.risk_id for e in self.event_handler.upcoming_broker_premium.values() if isinstance(e, AddPremiumEvent)
        ]

        upcoming_broker_claim = [
            e.risk_id for e in self.event_handler.upcoming_broker_claim.values() if isinstance(e, AddClaimEvent)
        ]

        # Enact the events
        self.event_handler.forward(self.market, step_time)

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
        sorted_broker_risk_start_times = np.sort(np.unique(broker_risk_event_start_times))

        # Update all the agents, run the event at the same start time
        for start_time in sorted_broker_risk_start_times:
            # Move along the market's time
            self.market.time = start_time
            # Get all the events starting at this time
            starting_broker_risk = []
            for i in range(len(self.broker_risk_events)):
                if self.broker_risk_events[i].risk_start_time == start_time-2:
                    starting_broker_risk.append(self.broker_risk_events[i])

            # Move along the corresponding syndicates
            self.evolve_action_market(starting_broker_risk, self.market.time)

            # Empty all the actions to apply to syndicates
            self.actions_to_apply = []

        # Track any newly-added attritional_loss events and execute
        newly_added_attritional_loss_events = {
            e.risk_id: e.risk_start_time
            for e in self.event_handler.completed_attritional_loss.values()
            if isinstance(e, AddAttritionalLossEvent) and (e.risk_id in upcoming_attritional_loss)
        }
        attritional_loss_event_start_times = np.array(
            [
                newly_added_attritional_loss_events.get(risk_id)
                for risk_id in upcoming_attritional_loss
                if newly_added_attritional_loss_events.get(risk_id) != None
            ]
        )
        sorted_attritional_loss_start_times = np.sort(np.unique(attritional_loss_event_start_times))
        follow_id = []
        for start_time in sorted_attritional_loss_start_times:
            starting_attritional_loss = None
            for i in range(len(self.attritional_loss_events)):
                if self.attritional_loss_events[i].risk_start_time == start_time:
                    starting_attritional_loss = self.attritional_loss_events[i]
            self.run_attritional_loss(starting_attritional_loss)

        # Track any newly-added broker_premium events and execute
        newly_added_broker_premium_events = {
            e.risk_id: e.risk_start_time
            for e in self.event_handler.completed_broker_premium.values()
            if isinstance(e, AddPremiumEvent) and (e.risk_id in upcoming_broker_premium)
        }
        broker_premium_event_start_times = np.array(
            [
                newly_added_broker_premium_events.get(risk_id)
                for risk_id in upcoming_broker_premium
                if newly_added_broker_premium_events.get(risk_id) != None
            ]
        )
        sorted_broker_premium_start_times = np.sort(np.unique(broker_premium_event_start_times))
        for start_time in sorted_broker_premium_start_times:
            starting_broker_premium = None
            for i in range(len(self.broker_premium_events)):
                if self.broker_premium_events[i].risk_start_time == start_time:
                    starting_broker_premium = self.broker_premium_events[i]
            self.run_broker_premium(starting_broker_premium)

        # Track any newly-added broker_claim events and execute
        newly_added_broker_claim_events = {
            e.risk_id: e.risk_start_time
            for e in self.event_handler.completed_broker_claim.values()
            if isinstance(e, AddClaimEvent) and (e.risk_id in upcoming_broker_claim)
        }
        broker_claim_event_start_times = np.array(
            [
                newly_added_broker_claim_events.get(risk_id)
                for risk_id in upcoming_broker_claim
                if newly_added_broker_claim_events.get(risk_id) != None
            ]
        )
        sorted_broker_claim_start_times = np.sort(np.unique(broker_claim_event_start_times))
        for start_time in sorted_broker_claim_start_times:
            starting_broker_claim = None
            for i in range(len(self.broker_claim_events)):
                if self.broker_claim_events[i].risk_start_time == start_time:
                    starting_broker_claim = self.broker_claim_events[i]
            self.run_broker_claim(starting_broker_claim)        

        # Track any newly-added catastrophe events and execute
        newly_added_catastrophe_events = {
            e.catastrophe_id: e.catastrophe_start_time
            for e in self.event_handler.completed_catastrophe.values()
            if isinstance(e, AddCatastropheEvent) and (e.catastrophe_id in upcoming_catastrophe)
        }
        catastrophe_event_start_times = np.array(
            [
                newly_added_catastrophe_events.get(catastrophe_id)
                for catastrophe_id in upcoming_catastrophe
                if newly_added_catastrophe_events.get(catastrophe_id) != None
            ]
        )
        sorted_catastrophe_start_times = np.sort(np.unique(catastrophe_event_start_times))
        for start_time in sorted_catastrophe_start_times:
            starting_catastrophe = None
            for i in range(len(self.catastrophe_events)):
                if self.catastrophe_events[i].catastrophe_start_time == start_time:
                    starting_catastrophe = self.catastrophe_events[i]
            self.run_catastrophe(starting_catastrophe)

        self.market.time = market_end_time

    def receive_actions(self, actions):

        # Choose the leader and save its action, the first syndicate with the highest line size wins 
        # TODO: will add selection algorithm in the future
        num_risk = len(actions)
        accept_actions = [[] for x in range(num_risk)]
        min_premium = [0 for x in range(num_risk)]
        lead_syndicate_id = [0 for x in range(num_risk)]
        sum_line_size = [0 for x in range(num_risk)]
        # Accept the quote
        syndicate_list = [[] for x in range(num_risk)]
        for num in range(num_risk):
            # If no one offer premium, no action
            sum = 0
            for j in range(len(actions[num])):
                sum += actions[num][j].premium
            if sum > 0:
                # Find the leader
                for sy in range(len(self.market.syndicates)):
                    if (actions[num][sy].premium != 0):
                        min_premium[num] = actions[num][sy].premium
                        lead_syndicate_id[num] = sy
                for sy in range(len(self.market.syndicates)):
                    if (actions[num][sy].premium != 0) and (actions[num][sy].premium < min_premium[num]):
                        min_premium[num] = actions[num][sy].premium
                        lead_syndicate_id[num] = sy
                    elif (actions[num][sy].premium != 0) and (actions[num][sy].premium == min_premium[num]):
                        np.random.seed(1234+sy)
                        a = np.random.randint(1)
                        if a == 0:
                            min_premium[num] = actions[num][sy].premium
                            lead_syndicate_id[num] = sy
                syndicate_list[num].append(lead_syndicate_id[num])
                accept_actions[num].append(actions[num][lead_syndicate_id[num]])
                sum_line_size[num] += self.lead_line_size
                # Sort the premium
                premium_sort = []
                for sy in range(len(self.market.syndicates)):
                    premium_sort.append(actions[num][sy].premium)
                premium_sort = np.array(premium_sort)
                premium_sort = np.sort(premium_sort)
                # Assign line size to the rest syndicates, min premium win
                rest_line_size = 1 - sum_line_size[num]
                for p in range(len(premium_sort)):
                    for sy in range(len(self.market.syndicates)):
                        if (rest_line_size > 0) and (sy not in syndicate_list[num]) and (actions[num][sy].premium == premium_sort[p]) and (premium_sort[p] != 0):
                            rest_line_size -= self.follow_line_sizes
                            accept_actions[num].append(actions[num][sy])
                            syndicate_list[num].append(sy)
                if rest_line_size > 0:
                    accept_actions[num].append(actions[num][lead_syndicate_id[num]])
                    syndicate_list[num].append(lead_syndicate_id[num])
                    
        # Save Actions to issue
        self.actions_to_apply = accept_actions