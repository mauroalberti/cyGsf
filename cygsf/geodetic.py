# -*- coding: utf-8 -*-

from typing import Tuple

from math import sqrt, sin, cos, radians


# Earth WGS84 parameters

WGS84 = {'semi-major axis': 6378137.0,
         'first eccentricity squared': 6.69437999014e-3}

epsg_4326_str = "EPSG:4326"
epsg_4978_str = "EPSG:4978"


def n_phi(phi_rad: float) -> float:
    """
    It return the N(phi) parameter.
    See: https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates
    N(phi) =  a / sqrt( 1 - e^2 * sin(phi)^2
    where phi is the latitude (in radians) and e^2 is the first eccentricity squared.

    :param phi_rad: the latitude expressed in radians.
    :type phi_rad: float.
    :return: the N(phi) value.
    :rtype: float.
    """

    a = WGS84['semi-major axis']
    e_squared = WGS84['first eccentricity squared']
    return a / sqrt(1.0 - e_squared * sin(phi_rad) ** 2)


def geodetic2ecef(lat: float, lon: float, height: float) -> Tuple[float, float, float]:
    """
    Converts from geodetic (lat-long-height) to Cartesian ECEF reference system.
    See: https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates

    :param lat: latitude.
    :type lat: float.
    :param lon: longitude.
    ;:type lon: float.
    :param height: height.
    :type height: float.
    :return: x, y and z coordinates.
    :rtype: tuple of three float values.
    """

    e_squared = WGS84['first eccentricity squared']

    lat_rad, lon_rad = radians(lat), radians(lon)

    nphi = n_phi(lat_rad)

    x = (nphi + height) * cos(lat_rad) * cos(lon_rad)
    y = (nphi + height) * cos(lat_rad) * sin(lon_rad)
    z = (nphi * (1 - e_squared) + height) * sin(lat_rad)

    return x, y, z

