"""
Contains all the simulation parameters
"""

def get_arguments():

    sim_args = {"max_time": 30000, # Simualtion time step daily
        "num_run_per_setting": 400 # Number of replication runs per simulation settings
        }

    manager_args = {"lead_top_k": 2, # The number of lead syndicates a broker reaches out to
        "follow_top_k": 5, # The number of follow syndicates a broker reaches out to
        "topology_broker_syndicate": 10 # Consider how to set the network topology
        }

    broker_args = {"num_brokers": 100, # Number of brokers in simulation
        "lambda_risks_daily": 0.06  # Lambda value for the Poisson distribution used by the broker process to generate new risks 
        }

    syndicate_args = {"num_syndicates": 20, # Number of syndicates in simulation
        "initial_capital": 10000000, # Initial capital of each syndicate
        "leader": False, # Default boolean value
        "lead_line_size": 0.5, # Default lead quote line size
        "follower": False, # Default boolean value
        "follow_line_size": 0.1, # Default follow quote line size
        "actuarial_pricing_internal_weight": 0.5,  # Whether acturial pricing based on syndicate history or industry histor
        "loss_experiency_weight": 0.2, # Whether actuarial pricing weighs the past losses more than recent losses 
        "volatility_weight": 0, # How much actuarial pricing considers the standard deviation of losses
        "underwriter_markup_recency_weight": 0.2, # Whether the underwriter markup weighs past mark-ups more than recent ones
        "upper_premium_limit": 1.2, # Upper premium limit factor
        "lower_premium_limit": 0.85, # Lower premium limit factor
        "premium_reserve_ratio": 0.5, # Premium to capital reserve ratio
        "minimum_capital_reserve_ratio": 1, # Reserved capital to working capital ratio
        "maximum_scaling_factor": 1, # Minimum scaling factor applied to premium
        "market_entry_probability": 0.3, # Default probability of entering the market
        "interest_rate_monthly": 0.001, # Interest rate for the capital monthly
        "dividends_of_profit": 0.4, # Dividends share of profit
        "exit_capital_threshold": 0.6, # Capital employment threshold for insurance firm exit
        "exit_time_limit": 24, # Time limit for insurance firm exit
        "sensitivity_premium": 1.29e-9, # Syndicate premium sensitivity parameter
        "initial_acceptance_threshold": 0.5,
        "acceptance_threshold_friction": 0.9
        }

    reinsurancefirm_args = {"num_reinsurancefirms": 4, # Number of reinsurance firms in simulation
        "initial_capital": 20000000, # Initial capital of each reinsurance firm
        "deductible": 0.25, # Uniform distribution between [25%,30%] of the total risk held per peril region by the insurer
        "market_entry_probability": 0.05, # Default probability of entering the market
        "exit_capital_threshold": 0.4, # Capital employment threshold for reinsurance firm exit
        "exit_time_limit": 48, # Time limit for reinsurance firm exit
        "sensitivity_premium": 1.55e-9 # Reinsurance firms premium sensitivity parameter
        }

    shareholder_args = {"num_shareholders": 1 # Number of shareholders in simulation
        }

    risk_args = {"num_riskmodels": 4, # Number of risk models in simulation
        "num_risks": 20000, # Number of risks
        "num_categories": 10, # Number of peril regions for the catastrophe
        "risk_limit": 10000000, # The maximum value of the risk
        "inaccuracy_riskmodels": 2,
        "riskmodel_margin_of_safety": 2,
        "value_at_risk_tail_probability": 0.005,
        "norm_profit_markup": 0.15,
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
        "mean_contract_runtime": 12,
        "money_supply": 2000000000

        }

    return sim_args, manager_args, broker_args, syndicate_args, reinsurance_args,  shareholder_args, risk_args



