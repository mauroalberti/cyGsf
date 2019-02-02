
from typing import Optional, List, Tuple

from math import sqrt

import numpy as np


from .constants import MIN_SEPARATION_THRESHOLD
from .geodetic import epsg_4326_str, epsg_4978_str, geodetic2ecef


cdef class Point:
    """
    Cartesian point. Dimensions: 4D (space-time).
    The crs string has structure like "EPSG:4326" or "EPSG:32632" or is empty.
    """
    cdef:
        readonly double x, y, z, t
        readonly char* crs

    def __init__(self, x, y, z = 0.0, t = 0.0, crs = ""):
        """
        Construct a Point instance.

        :param x: point x coordinate.
        :type x: double.
        :param y: point y coordinate.
        :type y: double.
        :param z: point z coordinate.
        :type z: double.
        :param t: point time coordinate.
        :type t: double.
        :param crs: CRS code.
        :type crs: string.
        """

        self.x = x
        self.y = y
        self.z = z
        self.t = t
        self.crs = crs

    def __repr__(self) -> str:

        return "Point({:.4f}, {:.4f}, {:.4f}, {:.4f}, '{}')".format(self.x, self.y, self.z, self.t, self.crs)

    cpdef bool __eq__(self, Point another):
        """
        Return True if objects are equal.

        Example:
          >>> Point(1., 1., 1.) == Point(1, 1, 1)
          True
          >>> Point(1., 1., 1., crs="EPSG:4326") == Point(1, 1, 1)
          False
          >>> Point(1., 1., 1.) == Point(1, 1, -1)
          False
        """

        return all([
            self.x == another.x,
            self.y == another.y,
            self.z == another.z,
            self.t == another.t,
            self.crs == another.crs])

    cpdef bool __ne__(self, Point another):
        """
        Return False if objects are equal.

        Example:
          >>> Point(1., 1., 1.) != Point(0., 0., 0.)
          True
          >>> Point(1., 1., 1., crs="EPSG:4326") != Point(1, 1, 1)
          True
        """

        return not (self == another)

    @property
    def vals(self) -> Tuple[float, float, float, float, str]:
        """
        Return the individual values of the point.

        :return: double array of x, y, z values

        Examples:
          >>> Point(4, 3, 7, crs="EPSG:4326").vals
          (4.0, 3.0, 7.0, 0.0, 'EPSG:4326')
        """

        return self.x, self.y, self.z, self.t, self.crs

    cpdef Point clone(self):
        """
        Clone a point.

        :return: a new point.
        :rtype: Point.
        """

        return Point(*self.vals)

    def toXYZ(self) -> Tuple[float, float, float]:
        """
        Returns the spatial components as a tuple of three values.

        :return: the spatial components (x, y, z).
        :rtype: a tuple of three floats.

        Examples:
          >>> Point(1, 0, 3).toXYZ()
          (1.0, 0.0, 3.0)
        """

        return self.x, self.y, self.z

    def toXYZT(self) -> Tuple[float, float, float, float]:
        """
        Returns the spatial and time components as a tuple of four values.

        :return: the spatial components (x, y, z) and the time component.
        :rtype: a tuple of four floats.

        Examples:
          >>> Point(1, 0, 3).toXYZT()
          (1.0, 0.0, 3.0, 0.0)
        """

        return self.x, self.y, self.z, self.t

    def toArray(self) -> 'array':
        """
        Return a Numpy array representing the point values (without the crs code).

        :return: Numpy array

        Examples:
          >>> Point(1, 2, 3).toArray()
          array([1., 2., 3., 0.])
        """

        return np.asarray(self.toXYZT())

    cpdef Point pXY(self):
        """
        Projection on the x-y plane

        :return: projected object instance

        Examples:
          >>> Point(2, 3, 4).pXY()
          Point(2.0000, 3.0000, 0.0000, 0.0000, '')
        """

        return Point(self.x, self.y, 0.0, self.t, self.crs)

    cpdef Point pXZ(self):
        """
        Projection on the x-z plane

        :return: projected object instance

        Examples:
          >>> Point(2, 3, 4).pXZ()
          Point(2.0000, 0.0000, 4.0000, 0.0000, '')
        """

        return Point(self.x, 0.0, self.z, self.t, self.crs)

    cpdef Point pYZ(self):
        """
        Projection on the y-z plane

        :return: projected object instance

        Examples:
          >>> Point(2, 3, 4).pYZ()
          Point(0.0000, 3.0000, 4.0000, 0.0000, '')
        """

        return Point(0.0, self.y, self.z, self.t, self.crs)

    def deltaX(self, another: 'Point') -> Optional[float]:
        """
        Delta between x components of two Point Instances.

        :return: x coordinates difference value.
        :rtype: optional float.

        Examples:
          >>> Point(1, 2, 3).deltaX(Point(4, 7, 1))
          3.0
          >>> Point(1, 2, 3, crs="EPSG:4326").deltaX(Point(4, 7, 1)) is None
          True
        """

        if self.crs != another.crs:
            return None
        else:
            return another.x - self.x

    def deltaY(self, another: 'Point') -> Optional[float]:
        """
        Delta between y components of two Point Instances.

        :return: y coordinates difference value.
        :rtype: optional float.

        Examples:
          >>> Point(1, 2, 3).deltaY(Point(4, 7, 1))
          5.0
          >>> Point(1, 2, 3, crs="EPSG:4326").deltaY(Point(4, 7, 1)) is None
          True
        """

        if self.crs != another.crs:
            return None
        else:
            return another.y - self.y

    def deltaZ(self, another: 'Point') -> Optional[float]:
        """
        Delta between z components of two Point Instances.

        :return: z coordinates difference value.
        :rtype: optional float.

        Examples:
          >>> Point(1, 2, 3).deltaZ(Point(4, 7, 1))
          -2.0
          >>> Point(1, 2, 3, crs="EPSG:4326").deltaZ(Point(4, 7, 1)) is None
          True
        """

        if self.crs != another.crs:
            return None
        else:
            return another.z - self.z

    cpdef double deltaT(self, Point another):
        """
        Delta between t components of two Point Instances.

        :return: difference value
        :rtype: float

        Examples:
          >>> Point(1, 2, 3, 17.3).deltaT(Point(4, 7, 1, 42.9))
          25.599999999999998
        """

        return another.t - self.t

    def dist3DWith(self, another: 'Point') -> Optional[float]:
        """
        Calculate Euclidean spatial distance between two points.

        :param another: another Point instance.
        :type another: Point.
        :return: the optional distance (when the two points have the same CRS).
        :rtype: optional float.

        Examples:
          >>> Point(1., 1., 1.).dist3DWith(Point(4., 5., 1,))
          5.0
          >>> Point(1, 1, 1, crs="EPSG:32632").dist3DWith(Point(4, 5, 1, crs="EPSG:32632"))
          5.0
          >>> Point(1, 1, 1).dist3DWith(Point(4, 5, 1))
          5.0
        """

        if self.crs != another.crs:
            return None
        else:
            return sqrt((self.x - another.x) ** 2 + (self.y - another.y) ** 2 + (self.z - another.z) ** 2)

    def dist2DWith(self, another: 'Point') -> Optional[float]:
        """
        Calculate horizontal (2D) distance between two points.

        :param another: another Point instance.
        :type another: Point.
        :return: the optional 2D distance (when the two points have the same CRS).
        :rtype: optional float.

        Examples:
          >>> Point(1., 1., 1.).dist2DWith(Point(4., 5., 7.))
          5.0
          >>> Point(1., 1., 1., crs="EPSG:32632").dist2DWith(Point(4., 5., 7.)) is None
          True
        """

        if self.crs != another.crs:
            return None
        else:
            return sqrt((self.x - another.x) ** 2 + (self.y - another.y) ** 2)

    cpdef Point scale(self, double scale_factor):
        """
        Create a scaled object.
        Note: it does not make sense for polar coordinates.
        TODO: manage polar coordinates cases OR deprecate and remove - after dependency check.

        Example;
          >>> Point(1, 0, 1).scale(2.5)
          Point(2.5000, 0.0000, 2.5000, 0.0000, '')
          >>> Point(1, 0, 1).scale(2.5)
          Point(2.5000, 0.0000, 2.5000, 0.0000, '')
        """

        x, y, z = self.x * scale_factor, self.y * scale_factor, self.z * scale_factor
        return Point(x, y, z, self.t, self.crs)

    cpdef Point invert(self):
        """
        Create a new object with inverted direction.
        Note: it depends on scale method, that could be deprecated/removed.

        Examples:
          >>> Point(1, 1, 1).invert()
          Point(-1.0000, -1.0000, -1.0000, 0.0000, '')
          >>> Point(2, -1, 4).invert()
          Point(-2.0000, 1.0000, -4.0000, 0.0000, '')
        """

        return self.scale(-1)

    def isCoinc(self, another: 'Point', tolerance: float = MIN_SEPARATION_THRESHOLD) -> Optional[bool]:
        """
        Check spatial coincidence of two points

        Example:
          >>> Point(1., 0., -1.).isCoinc(Point(1., 1.5, -1.))
          False
          >>> Point(1., 0., 0.).isCoinc(Point(1., 0., 0.))
          True
          >>> Point(1.2, 7.4, 1.4).isCoinc(Point(1.2, 7.4, 1.4))
          True
          >>> Point(1.2, 7.4, 1.4, crs="EPSG:4326").isCoinc(Point(1.2, 7.4, 1.4)) is None
          True
        """

        if self.crs != another.crs:
            return None
        else:
            return self.dist3DWith(another) <= tolerance

    def already_present(self, pt_list: List['Point'], tolerance: [int, float] = MIN_SEPARATION_THRESHOLD) -> Optional[bool]:
        """
        Determines if a point is already in a given point list, using an optional distance separation,

        :param pt_list: list of points. May be empty.
        :type pt_list: List of Points.
        :param tolerance: optional maximum distance between near-coincident point pair.
        :type tolerance: numeric (int, float).
        :return: True if already present, False otherwise.
        :rtype: optional boolean.
        """

        for pt in pt_list:
            if self.isCoinc(pt, tolerance=tolerance):
                return True
        return False

    def shift(self, sx: float, sy: float, sz: float) -> Optional['Point']:
        """
        Create a new object shifted by given amount from the self instance.

        Example:
          >>> Point(1, 1, 1).shift(0.5, 1., 1.5)
          Point(1.5000, 2.0000, 2.5000, 0.0000, '')
          >>> Point(1, 2, -1).shift(0.5, 1., 1.5)
          Point(1.5000, 3.0000, 0.5000, 0.0000, '')
       """

        return Point(self.x + sx, self.y + sy, self.z + sz, self.t, self.crs)

    def shiftByVect(self, v: Vect) -> 'Point':
        """
        Create a new object shifted from the self instance by given vector.

        :param v: the shift vector.
        :type v: Vect.
        :return: the shifted point.
        :rtype: Point.

        Example:
          >>> Point(1, 1, 1).shiftByVect(Vect(0.5, 1., 1.5))
          Point(1.5000, 2.0000, 2.5000, 0.0000, '')
          >>> Point(1, 2, -1).shiftByVect(Vect(0.5, 1., 1.5))
          Point(1.5000, 3.0000, 0.5000, 0.0000, '')
       """

        sx, sy, sz = v.toXYZ()

        return Point(self.x + sx, self.y + sy, self.z + sz, self.t, self.crs)

    def asVect(self) -> 'Vect':
        """
        Create a vector based on the point coordinates

        Example:
          >>> Point(1, 1, 0).asVect()
          Vect(1.0000, 1.0000, 0.0000)
          >>> Point(0.2, 1, 6).asVect()
          Vect(0.2000, 1.0000, 6.0000)
        """

        return Vect(self.x, self.y, self.z)

    def wgs842ecef(self) -> Optional['Point']:
        """
        Converts from WGS84 to ECEF reference system, provided its CRS is EPSG:4326.

        :return: the point with ECEF coordinates (EPSG:4978).
        :rtype: optional Point.
        """

        if self.crs != epsg_4326_str:
            return None

        x, y, z = geodetic2ecef(
            lat=self.y,
            lon=self.x,
            height=self.z)

        return Point(
            x=x,
            y=y,
            z=z,
            t=self.t,
            crs=epsg_4978_str)

