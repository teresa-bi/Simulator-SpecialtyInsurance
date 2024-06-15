"""
Contains all the simulation parameters
"""

def get_arguments():

    sim_args = {"max_time": 10, # Simualtion time step daily, 80 years 80*12
        "num_run_per_setting": 400, # Number of replication runs per simulation settings
        "mean_contract_runtime": 36, # Time scale five yeears
        "contract_runtime_halfspread": 2,
        "default_contract_payment_period": 3,
        "simulation_reinsurance_type": 'non-proportional',
        "margin_increase": 0,
        "market_permanency_off": False
        }

    manager_args = {"lead_top_k": 2, # The number of lead syndicates a broker reaches out to
        "follow_top_k": 5, # The number of follow syndicates a broker reaches out to
        "topology_broker_syndicate": 10 #####TODO Consider how to set the network topology
        }

    broker_args = {"num_brokers": 30, # Number of brokers in simulation, 5 times of num syndicates
        "lambda_risks_daily": 4,  # 0.06Lambda value for the Poisson distribution used by the broker process to generate new risks
        "decuctible": 0.2 # Percentage of risk value 
        }

    syndicate_args = {"num_syndicates": 6, # Number of syndicates in simulation
        "initial_capital": 80000, # Initial capital of each syndicate
        "lead_line_size": 0.5, # Default lead quote line size
        "follow_line_size": 0.25, # Default follow quote line size
        "ambiguity_level": 0, # From 0 to 1
        "cost_of_capital": 0.1, # Cost of capital 
        "actuarial_pricing_internal_weight": 0.5,  # Whether acturial pricing based on syndicate history or industry histor
        "loss_experiency_weight": 0.2, # Whether actuarial pricing weighs the past losses more than recent losses 
        "volatility_weight": 0, # How much actuarial pricing considers the standard deviation of losses
        "underwriter_markup_recency_weight": 0.2, # Whether the underwriter markup weighs past mark-ups more than recent ones
        "upper_premium_limit": 1.2, # Upper premium limit factor
        "lower_premium_limit": 0.85, # Lower premium limit factor
        "premium_reserve_ratio": 0.5, # Premium to capital reserve ratio
        "minimum_capital_reserve_ratio": 1, # Reserved capital to working capital ratio
        "maximum_scaling_factor": 1, # Minimum scaling factor applied to premium
        "market_entry_probability": 0.1, # Default probability of entering the market
        "interest_rate": 0.001, # Interest rate for the capital monthly
        "exit_capital_threshold": 0.6, # Capital employment threshold for insurance firm exit
        "cash_permanency_limit": 100, # This parameter enforces the limit under which the firms leave the market because they cannot underwrite anything
        "exit_time_limit": 24, # Time limit for insurance firm exit
        "premium_sensitivity": 5, # Syndicate premium sensitivity parameter 1.29e-9?
        "initial_acceptance_threshold": 0.5,
        "initial_acceptance_threshold_friction": 0.9,
        "reinsurance_limit": 0.1,
        "capacity_target_decrement_threshold": 1.8,
        "capacity_target_increment_threshold": 1.2,
        "capacity_target_decrement_factor": 24/25.,
        "capacity_target_increment_factor": 25/24.,
        "dividend_share_of_profits": 0.4,
        "default_non_proportional_reinsurance_deductible": 0.3,
        "default_non-proportional_reinsurance_excess": 1.0,
        "default_non-proportional_reinsurance_premium_share": 0.3,
        "insurers_balance_ratio": 0.1, # This ratio represents how low we want to keep the standard deviation of the cash reserved below the mean for insurers. Lower means more balanced
        "insurers_recursion_limit": 50, # Intensity of the recursion algorithm to balance the portfolio of risks for insurers
        "insurance_permanency_contracts_limit": 4,
        "insurance_permanency_ratio_limit": 0.6,
        "insurance_permanency_time_constraint": 24, # Defines the period that the insurers wait if they have few capital or few contract before leaving the market
        "insurance_retention": 0.85,
        "premium_sensitivity": 5, # This parameter represents how sensitive is the variation of the insurance premium with respect of the capital of the market
        "insurers_recursion_limit": 50 # Intensity of the recursion algorithm to balance the portfolio of risks for insurers 
        }

    reinsurancefirm_args = {"num_reinsurancefirms": 4, # Number of reinsurance firms in simulation
        "initial_capital": 20000000, # Initial capital of each reinsurance firm
        "deductible": 0.25, # Uniform distribution between [25%,30%] of the total risk held per peril region by the insurer
        "market_entry_probability": 0.05, # Default probability of entering the market
        "exit_capital_threshold": 0.4, # Capital employment threshold for reinsurance firm exit
        "exit_time_limit": 48, # Time limit for reinsurance firm exit
        "sensitivity_premium": 1.55e-9, # Reinsurance firms premium sensitivity parameter
        "initial_acceptance_threshold": 0.5,
        "acceptance_threshold_friction": 0.9,
        "reinsurance_limit": 0.1,
        "default_non_proportional_reinsurance_deductible": 0.3,
        "capacity_target_decrement_threshold": 1.8,
        "capacity_target_increment_threshold": 1.2,
        "capacity_target_decrement_factor": 24/25.,
        "capacity_target_increment_factor": 25/24.,
        "interest_rate": 0.001 # Interest rate for the capital monthly
        }

    shareholder_args = {"num_shareholders": 1 # Number of shareholders in simulation
        }

    risk_args = {"num_risks": 10000, # Number of risks
        "num_categories": 4, # Number of peril regions for the catastrophe
        "num_riskmodels": 4, # Number of risk models
        "min_cat_prob_distortion": 0.1, # Assume the difference between the closet loss probability model to the actual distribution 
        "max_cat_prob_distortion": 1, # Assume the difference between the farest loss probability model to the actual distribution 
        "risk_limit": 10000000, # The maximum value of the risk
        "riskmodel_inaccuracy": 2,
        "riskmodel_margin_of_safety": 2,
        "margin_increase": 0, # This parameter modifies the margin of safety depending on the number of risk models available in the market, when is 0 all risk models have the same margin of safety
        "value_at_risk_tail_probability": 0.005,
        "norm_profit_markup": 0.15,
        "dividend_share_of_profits": 0.4,
        "catastrophe_time_mean_separation": 100/3.,
        "lambda_attritional_loss": 0.1, # Lambda value for the Poisson distribution for the number of attritional claims generated per year
        "cov_attritional_loss": 1, # Coefficient of variation for the gamma distribution which generates the severity of attritional claim event
        "mu_attritional_loss": 3000000, # Mean of the gamma distribution which generates the severity of attritional claim events
        "lambda_catastrophe": 0.05, # Lambda value for the Poisson distribution for the number of catastrophe claims generated per year
        "pareto_shape": 5, # Shape parameter of the Pareto distribution which generates the severity of catastrophe claim events
        "minimum_catastrophe_damage": 0.25, # Minimum value for an event to be considered a catastrophe, fraction of the risk limit
        "var_em_exceedance_probability": 0.05, # The tail probability  used in the VaR calculations
        "var_em_safety_factor": 1, # Scaling safety factor applied to the VaR value, larger values employ more conservative exposure management
        "risk_factor_lower_bound": 0.4,
        "risk_factor_upper_bound": 0.6,
        "expire_immediately": False,
        "money_supply": 2000000000
        }
    
    seed = 123123

    return sim_args, manager_args, broker_args, syndicate_args, reinsurancefirm_args,  shareholder_args, risk_args, seed



