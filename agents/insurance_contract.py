import numpy as np
import sys, pdb

class InsuranceContract:
    """
    Insurance contract between broker and syndicate
    """
    def __init__(self, insurer, properties, time, premium, runtime, payment_period, expire_immediately, initial_VaR=0., insurancetype = "proportional", deductible_fraction = None, excess_fraction = None, reinsurance = 0):
        """
        Contract Instance

        Parameters
        ----------
        insurer: Syndicate
        properties: dict
        time: int, current time
        premium: float
        runtime: int
        payment_period: int
        expire_immediately: boolean
            True if the contract expires with the first risk event, False, if multiple risk events are covered
        initial_VaR: float
        optional:
            insurancetype: string
                The type of this contract, especially "proportional" vs "excess_of_loss"
            deductible: float
            excess: float
            reinsurance: float
                The value that is being reinsured
        Returns
        -------
        Insurance contract
        """
        self.insurer = insurer
        self.risk_factor = properties["risk_factor"]
        self.category = properties["category"]
        self.property_holder = properties["owner"]
        self.value = properties["value"]
        self.contract = properties.get("contract")
        self.insurancetype = properties.get("insurancetype")
        self.runtime = runtime
        self.starttime = time
        self.expiration = runtime + time
        self.expire_immediately = expire_immediately
        self.terminating = False
        self.current_claim = 0
        self.initial_VaR = initial_VaR

        # Set deductible from argument, risk property or default value, whichever first is not None
        defalut_deductible_fraction = 0.0
        deductible_fraction_generator = (item for item in [deductible_fraction, properties.get("deductible_fraction"), default_deductible_fraction] if item is not None)
        self.deductible_fraction = next(deductible_fraction_generator)
        self.deductible = self.deductible_fraction * self.value

        # Set excess from argument, risk property or default value, whichever first is not None
        default_excess_fraction = 1.0
        excess_fraction_generator = (item for item in [excess_fraction, properties.get("excess_fraction"), default_excess_fraction] if item is not None)
        self.excess_fraction = next(excess_fraction_generator)
        self.excess = self.excess_fraction * self.value

        self.reinsurance = reinsurance
        self.reinsurer = None
        self.reincontract = None
        self.reinsurance_share = None

        # Setup payment schedule
        total_premium = premium * self.value
        self.periodized_premium = total_premium / self.runtime
        self.payment_times = [time + i for i in range(runtime) if i % payment_period == 0]
        self.payment_values = total_premium * (np.ones(len(self.payment_times)) / len(self.payment_times))

        # Embed contract in reinsurance network if applicable
        if self.contract is not None:
            self.contract.reinsure(reinsurer=self.insurer,reinsurance_share=propeerties["reinsurance_share"], reincontract=self)

        # This flag is set to 1, when the contract is about to expire and there is an attempt to roll it over
        self.roll_over_flag = 0

    def check_payment_due(self, time):
        if len(self.payment_times) > 0 and time >= self.payment_times[0]:
            self.property_holder.receive_obligation(self.payment_values[0], self.insurer, time, 'premium')

            # Remove current payment from payment schedule
            self.payment_times = self.payment_times[1:]
            self.payment_values = self.payment_values[1:]

    def get_and_reset_current_claim(self):
        current_claim = self.current_claim
        self.current_claim = 0
        return self.category, current_claim, (self.insurancetype == "proportional")

    def terminate_reinsurance(self, time):
        """
        Terminate reinsurance method
        Parameters
        ----------
        time: int
            current time
        """
        if self.reincontract is not None:
            self.reincontract.dissolve(time)

    def dissolve(self, time):
        """
        Marks the contract as terminating, to avoid new reinsurance contracts for this contract
        """
        self.expiration = time

    def reinsure(self, reinsurer, reinsurance_share, reincontract):
        """
        Add parameters for reinsurance of the current contract
        Parameters
        ----------
        reinsurer: reinsurance firm
        reinsurance_share: float
            share of the value that is proportionally reinsured
        reincontract: reinsurance contract
        """
        self.reinsurer = reinsurer
        self.reinsurance = self.value * reinsurance_share
        self.reinsurance_share = reinsurance_share
        self.reincontract = reincontract
        assert self.reinsurance_share in [None,0.0,1.0]

    def unreinsure(self):
        """
        Remove parameters for reinsurance of the current contract, to be 
        """
        self.reinsurer = None
        self.reincontract = None
        self.reinsurance = 0
        self.reinsurance_share = None

