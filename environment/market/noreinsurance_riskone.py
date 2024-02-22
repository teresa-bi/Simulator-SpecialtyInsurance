import numpy as np

class NoReinsurance_RiskOne:
    """
    Basic environment including brokers, syndicates, sharholders, and one risk model
    """
    def __init__(self, time, sim_maxstep, manager_args, brokers, syndicates, reinsurancefirms, shareholders, risks, risk_model_configs):
        self.time = time
        self.sim_maxstep = sim_maxstep
        self.manager_args = manager_args
        self.brokers = brokers
        self.syndicates = syndicates
        self.reinsurancefirms = reinsurancefirms
        self.shareholders = shareholders
        self.risks = risks
        self.risk_model_configs = risk_model_configs

        market.catastrophe_event
        market.broker_bring_risk
        market.broker_pay_premium
        market.broker_bring_claim

    from __future__ import annotations

import json
import time
import typing
from collections import defaultdict

import numpy as np
from starling.core.aircraft import Aircraft
from starling.core.airspace import Airspace
from starling.utility import convert, geometry


class Environment:
    """
    Environment holding all situational data.
    """

    def __init__(
        self,
        time: float,
        airspace: Airspace,
        aircraft: typing.Dict[str, Aircraft],
        finished_aircraft: typing.Optional[typing.Dict[str, Aircraft]] = None,
    ):
        """
        Construct a new instance.

        Parameters
        ----------
        time: float
            Current time in the Environment.
        airspace: Airspace
            An Airspace the Aircraft are flying through.
        aircraft: dict[str, Aircraft]
            A dictionary of {callsign, Aircraft} currently in the Environment.
        finished_aircraft: dict[str, Aircraft], optional
            A dictionary of {callsign, Aircraft} that were previously in the Environment.
        """

        if time < 0.0:
            raise ValueError("Time must be non-negative.")

        self.time = time
        self.airspace = airspace
        self.aircraft = aircraft
        if finished_aircraft is None:
            self.finished_aircraft = {}
        else:
            self.finished_aircraft = finished_aircraft
        # By default all Aircraft are in Route following mode until told otherwise
        self.on_route = defaultdict(lambda: True)

    @staticmethod
    def from_json(s: str) -> Environment:
        """
        Construct a new instance from JSON representation.

        Parameters
        ----------
        s: str
            A string representation of an Environment in a JSON/dictionary structure.

        Returns
        ----------
        Environment
        """

        data = json.loads(s)

        time = convert.time_str(data["time"])

        airspace = Airspace.from_json(json.dumps(data["airspace"]))

        aircraft = {}
        for callsign, route in data["aircraft"].items():
            aircraft[callsign] = Aircraft.from_json(json.dumps(route))

        finished_aircraft = {}
        if "finished_aircraft" in data:
            for callsign, route in data["finished_aircraft"].items():
                finished_aircraft[callsign] = Aircraft.from_json(json.dumps(route))

        return Environment(time, airspace, aircraft, finished_aircraft)

    @staticmethod
    def load(filename: str) -> Environment:
        """
        Construct a new instance from a file.

        Parameters
        ----------
        filename: str
            Path to a JSON file with an Environment definition in dictionary format.

        Returns
        ----------
        Environment
        """

        with open(filename) as file:
            return Environment.from_json(file.read())

    def data(self) -> typing.Dict[str, typing.Any]:
        """
        Get the data as a serialisable dictionary.

        Returns
        ----------
        dict
        """

        timestamp = time.gmtime(self.time)
        time_str = time.strftime("%H:%M:%S", timestamp)

        return {
            "time": time_str,
            "airspace": self.airspace.data(),
            "aircraft": {callsign: aircraft.data() for (callsign, aircraft) in self.aircraft.items()},
            "finished_aircraft": {callsign: aircraft.data() for (callsign, aircraft) in self.finished_aircraft.items()},
        }

    def to_json(self) -> str:
        """
        Serialise the instance to JSON.

        Returns
        ----------
        str
        """

        return json.dumps(self.data(), indent=4)

    def save(self, filename: str):
        """
        Write the instance to a file.

        Parameters
        ----------
        filename: str
            Path to file.
        """

        with open(filename, "w") as file:
            file.write(self.to_json())

    def is_active(self) -> typing.Dict[str, bool]:
        """
        Determine which Aircraft are active (i.e., are within the Airspace).

        Returns
        ----------
        dict[str, bool]
            {Aircraft callsign: whether Aircraft is active}
        """

        active = {}

        for callsign, aircraft in self.aircraft.items():
            active[callsign] = self.airspace.contains(aircraft.pos3d())

        return active

    def aircraft_lateral_distance(self, aircraft_list) -> np.ndarray:
        """
        Compute distance matrix using lats/lons of Aircraft.

        Returns
        ----------
        np.ndarray
        """

        if len(aircraft_list) < 2:
            raise ValueError("len(aircraft_list) must be >= 2")

        # Collect latitudes/longitudes of active aircraft
        lats = []
        lons = []
        for ac in aircraft_list:
            lats.append(self.aircraft[ac].lat * convert.DEG_TO_NMI)
            lons.append(self.aircraft[ac].lon * convert.DEG_TO_NMI)

        dist_matrix = geometry.compute_distance_matrix(np.vstack([np.array(lats), np.array(lons)]).T)
        return dist_matrix

    def active_aircraft_lateral_distance(self) -> typing.Tuple[typing.List[str], typing.Optional[np.ndarray]]:
        """
        A convenience function to determine which Aircraft are active within the Airspace
            and then compute distance matrix using lats/lons.

        Returns
        ----------
        tuple[list, np.ndarray]
            list of active Aircraft callsigns, a distance matrix (or None if there are no active Aircraft)
        """

        # List of active aircraft
        aircraft = self.is_active()

        def _by(x):
            return aircraft[x]

        aircraft_list = list(filter(_by, aircraft))

        if len(aircraft_list) > 1:
            # Compute distance matrix
            dist_matrix = self.aircraft_lateral_distance(aircraft_list)
        else:
            dist_matrix = None

        return aircraft_list, dist_matrix

    def extract_exterior_coords(self) -> typing.List[typing.Tuple[typing.List, typing.List]]:
        """
        Extract coordinates of the Airspace exterior boundary.

        Returns
        ----------
        tuple[list, list]
            The [latitude,longitude] coordinates of the Airspace exterior boundary.

        Examples
        ----------
        x, y = environment.extract_exterior_coords()
        """

        points = []
        for sector in self.airspace.sectors.values():
            for volume in sector.volumes:
                for coord in volume.area.boundary.exterior.coords:
                    points.append([coord[0], coord[1]])

        return list(zip(*points))

    def extract_fixes(self) -> typing.Tuple[typing.List[float], typing.List[float], typing.List[float]]:
        """
        Extract fixes as lists of longitudes, latitudes, names.

        Returns
        ----------
        tuple[list, list, list]
            The fix longitudes, latitudes and names.
        """

        x_f = []
        y_f = []
        n_f = []
        for name, location in self.airspace.fixes.places.items():
            x_f.append(location.lon)
            y_f.append(location.lat)
            n_f.append(name)

        return x_f, y_f, n_f

    def distances_to_exit(self) -> typing.Dict[str, float]:
        """
        Return distance [nmi] to exit for all active Aircraft.

        Returns
        ----------
        dict[str, float]
            {Aircraft callsign, distance to exit}
        """

        distances = {}
        aircraft = self.is_active()
        for callsign, active in aircraft.items():
            if active:
                a = self.aircraft[callsign]
                distances[callsign] = a.distance_to_exit(self.airspace)

        return distances

    def controllable_aircraft(self) -> typing.List[str]:
        """
        Return a list of callsigns for Aircraft that respond to Agent-issued Actions.
        Non-controllable Aircraft are, for example, military. The Agent has to plan around them.

        Returns
        ----------
        list[str]
        """

        return [callsign for callsign, aircraft in self.aircraft.items() if aircraft.controllable]

    def find_closest_fix(self, callsign: str) -> str:
        """
        Find closest on Route Fix in direction of the exit Fix given Aircraft current position.
        Returns the exit Fix if Aircraft has passed the exit already.

        Parameters
        ----------
        callsign: str
            Aircraft callsign.

        Returns
        ----------
        str
            Name of the closest Fix on Route in direction of exit (this is the exit Fix if already passed).
        """

        aircraft = self.aircraft[callsign]
        ac_loc = aircraft.pos2d()

        exit_loc = self.airspace.route_exit_fix(aircraft.flight_plan.route)

        closest_fix = aircraft.flight_plan.route.names[0]
        closest_fix_loc = self.airspace.fixes.places[closest_fix]

        for fix in aircraft.flight_plan.route.names[1:]:
            fix_loc = self.airspace.fixes.places[fix]

            if (
                closest_fix_loc.distance(exit_loc) > ac_loc.distance(exit_loc)
                and fix_loc.distance(exit_loc) < closest_fix_loc.distance(exit_loc)
            ) or (fix_loc.distance(ac_loc) < closest_fix_loc.distance(ac_loc)
            ) or (closest_fix_loc.distance(ac_loc) < 15):
                closest_fix = fix
                closest_fix_loc = fix_loc

            if fix_loc == exit_loc:
                break

        return closest_fix

