from __future__ import annotations
from environment.event.catastrophe import CatastropheEvent
from environment.event.attritionalloss import AttritionalLossEvent
from environment.event.add_risk import AddRiskEvent
from environment.event.add_premium import AddPremiumEvent
from environment.event.add_claim import AddClaimEvent
from environment.market import NoReinsurance_RiskOne

class EventGenerator():
    """
    Generate different types of events
    """
    def __init__(self, risk_model_configs):
        """
        Parameters
        ----------
        risk_model_configs: dict
            Risk model configurations.
        """
        self.damage_distribution = []
        self.expire_immediately = []
        self.catastrophe_separation_distribution = []
        self.norm_premium = []
        self.num_categories = []
        self.risk_value_mean = []
        self.risk_factor_mean = []
        self.norm_profit_markup = []
        self.margin_of_safety = []
        self.var_tail_prob = []
        self.inaccuracy_by_categ = []
        self.num_riskmodels = len(risk_model_configs)
        for i in range(self.num_riskmodels):
            self.damage_distribution.append(risk_model_configs[i]["damage_distribution"])
            self.expire_immediately.append(risk_model_configs[i]["expire_immediately"])
            self.catastrophe_separation_distribution.append(
                risk_model_configs[i]["catastrophe_separation_distribution"])
            self.norm_premium.append(risk_model_configs[i]["norm_premium"])
            self.num_categories.append(risk_model_configs[i]["num_categories"])
            self.risk_value_mean.append(risk_model_configs[i]["risk_value_mean"])
            self.risk_factor_mean.append(risk_model_configs[i]["risk_factor_mean"])
            self.norm_profit_markup.append(risk_model_configs[i]["norm_profit_markup"])
            self.margin_of_safety.append(risk_model_configs[i]["margin_of_safety"])
            self.var_tail_prob.append(risk_model_configs[i]["var_tail_prob"])
            self.inaccuracy_by_categ.append(risk_model_configs[i]["inaccuracy_by_categ"])

    def generate_catastrophe_events(self, risks):
        """
        Generate a set of CatastropheEvent for an insurance market.

        Parameters
        ----------
        risks: list
            All the generated catastrophe risks

        Returns
        ----------
        List[CatastropheEvent]
            A list of CatastropheEvents
        """
        catastrophe_events = []
        for i in range(len(risks)):
            catastrophe_event = CatastropheEvent(risks[i].get("risk_id"), risks[i].get("risk_start_time"),
                                                risks[i].get("risk_factor"), risks[i].get("risk_category"),
                                                risks[i].get("risk_value"))
            catastrophe_events.append(catastrophe_event)

        return catastrophe_events

    def generate_attritional_loss_events(self, sim_args, risks):
        """
        Generate a set of AttritionalLossEvent for an insurance market.

        Returns
        ----------
        List[AttritionalLossEvent]
            A list of AttritionalLossEvents
        """
        attritional_loss_events = []
        for time in range(sim_args["max_time"]):
            attritional_loss_event = AttritionalLossEvent(time, time, risks[0].get("risk_factor"), risks[0].get("risk_category"), risks[0].get("risk_value"))
            attritional_loss_events.append(attritional_loss_event)

        return attritional_loss_events

    def generate_risk_events(self, brokers):
        """
        Generate a set of AddRiskEvent for an insurance market.

        Parameters
        ----------
        brokers: a list of Broker

        Returns
        ----------
        List[AddRiskEvent]
            A list of AddRiskEvents
        """
        add_risk_events = []
        for broker_id in range(len(brokers)):
            num_total_risks = len(brokers[broker_id].risks)
            for k in range(num_total_risks):
                risk = brokers[broker_id].risks[k]
                add_risk_event = AddRiskEvent(risk["risk_id"], risk["broker_id"], risk["risk_start_time"], risk["risk_end_time"], risk["risk_factor"], risk["risk_category"], risk["risk_value"])
                add_risk_events.append(add_risk_event)

        return add_risk_events

    def generate_premium_events(self, brokers):
        """
        Generate a set of AddPremiumEvent for an insurance market. 

        Parameters
        ----------
        brokers: a list of Broker

        Returns
        ----------
        List[AddPremiumEvent]
            A list of AddPremiumEvents
        """
        add_premium_events = []
        for broker_id in range(len(brokers)):
            num_total_premiums = len(brokers[broker_id].underwritten_contracts)
            for k in range(num_total_premiums):
                premium = brokers[broker_id].underwritten_contracts[k]  #TODO: cannot be got from the begining, updated during the simulation, related to underwritten_contracts
                add_premium_event = AddPremiumEvent(premium["risk_id"], premium["broker_id"], premium["risk_start_time"], premium["risk_end_time"], premium["risk_category"], premium["risk_value"], premium["syndicate_id"], premium["premium"])
                add_premium_events.append(add_premium_event)

        return add_premium_events

    def generate_claim_events(self, brokers):
        """
        Generate a set of AddClaimEvent for an insurance market.

        Parameters
        ----------
        brokers: a list of Broker

        Returns
        ----------
        List[AddClaimEvent]
            A list of AddClaimEvents
        """
        add_claim_events = []
        for broker_id in range(len(brokers)):
            for k in range(len(brokers[broker_id].underwritten_contracts)):
                claim = brokers[broker_id].underwritten_contracts[k]
                if claim["claim"] == True:  #TODO: after the claim, it will be false
                    add_claim_event = AddClaimEvent(claim["risk_id"], claim["broker_id"], claim["risk_start_time"], claim["risk_end_time"], claim["risk_category"], claim["risk_value"], claim["syndicate_id"])
                    add_claim_events.append(add_claim_event)    

        return add_claim_events