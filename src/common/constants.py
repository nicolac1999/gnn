import tensorflow as tf


def get_g():
    """
    Returns Gravitational acceleration for standard SI units [m], [s], [kg],
    """
    return tf.constant(9.81)


def get_g_cm():
    """
    Returns Gravitational acceleration compatible with units [cm], [s] and [g]
    """
    return tf.constant(980.67)


def get_kapitza_conductance():
    """
    Returns Kapitza conductance.

    Kapitza conductance is a function of the third power of T (temperature), or better an exponent of 3.4
    The  known range of Kapitza conductance is [860-1200] W/(m^2 * K^4)
    Thesis 2000 --> 860
    Benjamin CRYO_SIMU -->1100
    Thesis 2007 --> 1200

    Even if Kapitza conductance is a function of T, for now we are not modelling it so we are assuming it constant,
    Future work: model Kapitza conductance
    """

    return tf.constant(860.)


def get_kapitza_conductance_cm():
    """
    Returns Kapitza conductance in W/(cm^2 * K^4)

    The function calls "get_kapitza_conductance" and applies the transformation from
    m^2 to cm^2

    """
    kapitza_m = get_kapitza_conductance()

    kapitza_cm = kapitza_m * tf.constant(1e-4)

    return kapitza_cm

def get_helium_conductivity():
    """
    Returns Superfluid Helium Thermal Conductivity

    Superfluid Helium Thermal Conductivity is a function of T (temperature) and P(pressure).
    An example of this function can be found in the section docs\images_helium_physical_properties

    For now, we are assuming it constant and equal to 1500 (W^3/(cm^5 * K)
    Note: in the figure the thermal capacity is in the unit of W^3/(m^5 * K), but we are converting it in cm^5
    for numerical stability
    Future work: model Superfluid Helium Thermal Conductivity
    """

    return tf.constant(1500.)

def get_specific_heat_capacity():
    """
    Returns the Specific Heat Capacity of Helium II

    As the thermal conductivity also the specific heat capacity is a function of T (temperature) and P(pressure)
    An example of this function can be found in the section docs\images_helium_physical_properties

    For now we are assuming it constant and equal to 4000 J/(Kg * K)
    Future work: model Specific Heat Capacity
    """
    return tf.constant(4000.)

def get_latent_heat_vaporization():
    """
    Returns the Latent heat of vaporization of superfluid helium

    The latent heat of vaporization is defined as the amount of energy must be supplied to a liquid substance
    at its boiling point to transform it into a gas. The Latent heat of vaporization (or enthalpy of vaporization) is a function
    of the pressure at which the transformation takes place

    Unit: J/kg or J/g
    """
    return tf.constant(23.)

def get_superfluid_liquid_density():
    """
    Returns the helium superfluid liquid density

    Unit: g/cm^3
    """

    #return tf.constant(0.1456) slide 69 thesis 2000
    return tf.constant(0.145)



