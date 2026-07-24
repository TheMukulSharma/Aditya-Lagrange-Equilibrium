"""
Locate the Sun-Earth L1 point via direct force-balance root-finding.

Physics equation:
    G·M_sun/(D-r)² - G·M_earth/r² = ω²·(D-r)

Rearranged for root-finding:
    f(r) = G·M_sun/(D-r)² - G·M_earth/r² - ω²·(D-r) = 0
"""

import numpy as np
from scipy.optimize import brentq
from astropy.constants import (
    M_earth,
    M_sun,
    G,
    au,
)


class CelestialBody:
    """A minimal container for a gravitating body's physical properties."""

    def __init__(self, name: str, mass: float, position: np.ndarray):
        self.name = name
        self.mass = mass  # kg
        self.position = np.array(position, dtype=float)  # meters, rotating frame

    def __repr__(self):
        return f"CelestialBody(name={self.name!r}, mass={self.mass:.3e} kg)"


# Physical constants ( in SI unit)
G = G.value  # gravitational constant, m^3 kg^-1 s^-2
AU = au.value  # Sun-Earth distance, meters
YEAR = 365.25 * 24 * 3600  # seconds


# mass of celestial objects (in kg)
m_sun_kg = M_sun.value  # 1.98840987... x 10^30 kg
m_earth_kg = M_earth.value  # 5.97216787... x 10^24 kg

sun = CelestialBody("Sun", mass=m_sun_kg, position=[0.0, 0.0])
earth = CelestialBody("Earth", mass=M_earth.value, position=[AU, 0.0])


omega = 2 * np.pi / YEAR  # Earth's orbital angular velocity, rad/s


def l1_force_balance(r: float, sun: CelestialBody, earth: CelestialBody, omega: float):
    """
    L1 force-balance equation. Zero when r is the true L1
    distance from Earth (meters), measured along the Sun-Earth line.
    """
    D = earth.position[0] - sun.position[0]
    sun_pull = G * sun.mass / (D - r) ** 2
    earth_pull = G * earth.mass / r**2
    centripetal_required = omega**2 * (D - r)
    return sun_pull - earth_pull - centripetal_required


def find_l1(sun: CelestialBody, earth: CelestialBody, omega: float):
    """Root-find the L1 distance from Earth using Brent's method for finding roots"""

    D = earth.position[0] - sun.position[0]
    # L1 lies strictly between the two bodies; keep the bracket off the
    # singularities at r=0 and r=D.
    return brentq(l1_force_balance, 1e6, D - 1e6, args=(sun, earth, omega))


if __name__ == "__main__":
    r_L1 = find_l1(sun, earth, omega)
    print(f"L1 distance from Earth: {r_L1:,.0f} m  ({r_L1/1000:,.1f} km)")
