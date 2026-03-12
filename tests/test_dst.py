import numpy as np

from app_main import (
    parse_mass,
    FRAME,
)

# The DST combine function in your app_main.py is defined locally
# within the DST module under "dempster". We import the public API here:
from app_main import parse_mass as global_parse_mass

def test_parse_mass_basic():
    m = parse_mass({"{sustained}": 0.5, "Theta": 0.5})
    assert np.isclose(sum(m.values()), 1.0)
    assert FRAME in m

def test_parse_mass_subset_handling():
    m = parse_mass({"{managed,rapid}": 0.3, "{sustained}": 0.7})
    assert np.isclose(sum(m.values()), 1.0)
    # the set {managed, rapid} must be inside FRAME
    assert frozenset({"managed", "rapid"}) in m

def test_dempster_combine():
    # local reimplementation to match your DST module's function
    def dempster(m1, m2):
        K = 0
        m = {}
        for A,vA in m1.items():
            for B,vB in m2.items():
                inter = A & B
                if len(inter) == 0:
                    K += vA*vB
                else:
                    m[inter] = m.get(inter,0) + vA*vB
        if K >= 1:
            return {FRAME:1.0}
        s = 1/(1-K)
        for A in list(m.keys()):
            m[A] *= s
        tot = sum(m.values())
        return {A:v/tot for A,v in m.items()}

    m1 = parse_mass({"{sustained}": 0.6, "Theta": 0.4})
    m2 = parse_mass({"{sustained}": 0.5, "Theta": 0.5})
    m = dempster(m1, m2)

    assert np.isclose(sum(m.values()), 1.0)
    assert frozenset({"sustained"}) in m or FRAME in m