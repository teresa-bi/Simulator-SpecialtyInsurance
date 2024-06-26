from __future__ import annotations
from environment.event.add_catastrophe import AddCatastropheEvent
from environment.event.add_attritionalloss import AddAttritionalLossEvent
from environment.event.add_risk import AddRiskEvent
from environment.event.add_premium import AddPremiumEvent
from environment.event.add_claim import AddClaimEvent
from environment.event.distributiontruncated import TruncatedDistWrapper
import scipy.stats

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

    def generate_catastrophe_events(self, catastrophes):
        """
        Generate a set of CatastropheEvent for an insurance market.

        Parameters
        ----------
        catastrophes: list
            All the generated catastrophe catastrophes

        Returns
        ----------
        List[AddCatastropheEvent]
            A list of AddCatastropheEvent
        """
        
        catastrophe_events = []
        for i in range(len(catastrophes)):
            add_catastrophe_event = AddCatastropheEvent(catastrophes[i].get("catastrophe_id"), catastrophes[i].get("catastrophe_start_time"),
                                                catastrophes[i].get("catastrophe_category"), catastrophes[i].get("catastrophe_value"))
            catastrophe_events.append(add_catastrophe_event)

        return catastrophe_events

    def generate_attritional_loss_events(self, attritional_losses):
        """
        Generate a set of AttritionalLossEvent for an insurance market.

        Returns
        ----------
        List[AddAttritionalLossEvent]
            A list of AddAttritionalLossEvent
        """
        attritional_loss_events = []
        for i in range(len(attritional_losses)):
            add_attritional_loss_event = AddAttritionalLossEvent(attritional_losses[i].get("attritional_loss_id"), attritional_losses[i].get("attritional_loss_start_time"),
                                                attritional_losses[i].get("attritional_losses_value"))
            attritional_loss_events.append(add_attritional_loss_event)

        return attritional_loss_events

    def generate_risk_events(self, sim_args, broker_risks):
        """
        Generate a set of AddRiskEvent for an insurance market.

        Parameters
        ----------
        brokers: a list of Broker

        Returns
        ----------
        List[AddRiskEvent]
            A list of AddRiskEvent
        """
        add_risk_events = []
        for k in range(len(broker_risks)):
            risk = broker_risks[k]
            add_risk_event = AddRiskEvent(risk.get("risk_id"), risk.get("broker_id"), risk.get("risk_start_time"), risk.get("risk_start_time")+1, risk.get("risk_factor"), risk.get("risk_category"), risk.get("risk_value"))
            add_risk_events.append(add_risk_event)

        return add_risk_events

    def generate_premium_events(self, sim_args):
        """
        Generate a set of AddPremiumEvent for an insurance market. 

        Returns
        ----------
        List[AddPremiumEvent]
            A list of AddPremiumEvents
        """
        add_premium_events = []
        for time in range(sim_args.get("max_time")+2):
            add_premium_event = AddPremiumEvent(time, time)
            add_premium_events.append(add_premium_event)

        return add_premium_events

    def generate_claim_events(self, sim_args):
        """
        Generate a set of AddClaimEvent for an insurance market.

        Returns
        ----------
        List[AddClaimEvent]
            A list of AddClaimEvents
        """
        add_claim_events = []
        for time in range(sim_args.get("max_time")+2):
            add_claim_event = AddClaimEvent(time, time)
            add_claim_events.append(add_claim_event)

        return add_claim_events