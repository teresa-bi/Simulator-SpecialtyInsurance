# Simulator-SpecialtyInsurance
An Agent-based Simulator for the Specialty Insurance Market
Time Line 
<img width="640" alt="TimeLine" src="https://github.com/teresa-bi/Simulator-SpecialtyInsurance/assets/97514447/b0dd40aa-b7bb-4a21-b4a8-fb733f9c779f">

Agents Interactions
<img width="840" alt="AgentsInteraction" src="https://github.com/teresa-bi/Simulator-SpecialtyInsurance/assets/97514447/139c014e-1af7-4e41-ad09-c26c38fad3ce">

# Quickstart
## Prerequisties
- Python 3.8
- Jupyter notebook 5.7.4
## Basic Usage

# Overview
## Modules
1. agents: market participants including broker, syndicate, reinsurancefirm, and shareholder
2. environment: contains environment, scenario generator and risk generator
   - env: SpecialtyInsuranceMarketEnv is a gym like environment contains reset, step, termination, etc
   - scenario_generator: different scenarios based on different types of participants and risk models
   - risk: generate catastrophe loss and attritional loss
3. manager: 
   - event_handler: handle events generated by agents and risks
   - environment_manager: evolves the environment and log status
   - ai_model: includes network settings and runs the training and testing process
   - game_model: calculates the payoff based on strategies
4. logger:
   - arguments: default arguments for agents and risks
   - logger: agents' status recording
6. visualisation
  
## Interfaces
- Broker
- Syndicate
- ReinsuranceFirm
- Shareholder
- CatastropheGenerator
- AttritionalLossGenerator
- Visuale
- Evaluate

## Simulation Management
- EGTA Equlibrium Calculation
