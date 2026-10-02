"""
The ground force law, as a pure function.

Kept separate from the environment so it can be tested in isolation against
closed-form results -- this is the "verified against closed-form" claim in
the proposal, and it is the part most likely to hide a sign error.
"""
import numpy as np

D_REF   = 0.02      # m, reference depth for the stiffening term
V_EPS   = 1e-3      # m/s, friction regularisation width


def normal_force(depth, sink_rate, k0, c, alpha, f_yield):
    """Vertical ground force at one contact point.

    depth      : penetration below the (possibly dented) surface, m. <=0 -> no contact
    sink_rate  : rate of penetration, m/s (positive = sinking)
    returns    : (force_N, new_dent_offset or None)

    The surface stiffens with depth and saturates at f_yield, past which it
    yields plastically -- the force cannot exceed f_yield and the surface is
    permanently displaced.
    """
    if depth <= 0.0:
        return 0.0, None

    k = k0 * (1.0 + alpha * depth / D_REF)
    f = k * depth + c * sink_rate

    f = max(f, 0.0)                       # ground pushes, never pulls

    if f > f_yield:                       # plastic yield
        elastic_depth = f_yield / k       # depth the surface can still support
        return f_yield, depth - elastic_depth   # how much further to dent
    return f, None


def friction_force(normal, tangential_velocity, mu):
    """Velocity-regularised Coulomb friction.

    tanh instead of a hard sign() avoids the stick-slip discontinuity that
    causes jitter with an explicit integrator. Note the minus sign: friction
    OPPOSES motion.
    """
    return -mu * normal * np.tanh(tangential_velocity / V_EPS)
