
import numpy as np
from scipy import interpolate

from common.constants import get_superfluid_liquid_density
import numbers
from typing import Union

class BayonetGeometryEstimator:

    def __init__(self, D, d):

        self.D = D
        self.d = d

        NUM_STEPS = 500
        xs_height = np.linspace(0., self.D, NUM_STEPS)
        ys_cross_area = self.cross_area_from_height(xs_height)

        self._lut_height_to_cross_area = {'x': ys_cross_area, 'y': xs_height}
        self._height_from_cross_area_interpolator = interpolate.interp1d(**self._lut_height_to_cross_area, bounds_error=False, fill_value=(0, self.D) )
        # self._height_from_cross_area_interpolator = interpolate.interp1d(x=ys,y=xs)


    def get_height(self):
        """
        Returns dimension that correspond to the height. In this implementation, it is the big diameter (self.D)
        :return:
        """
        return self.D


    def max_cross_section_area(self):
        """
        Returns the maximal cross-section area, when the pipe is fully filled.
        """
        result = self.cross_area_from_height(self.get_height())
        return result



    def height_from_fraction(self, fraction: Union[float, np.ndarray]):
        """
        Returns the liquid height in cm for a given fraction of wetted perimeter

        :param fraction: fraction of wetted perimeter
        :return h: liquid height
        """

        alpha = fraction * np.pi
        h = self.D / 2 * (1 - np.cos(alpha))

        return h


    def cross_area_from_fraction(self, fraction: Union[float, np.ndarray]):
        """
        Returns the cross-section in cm^2 for a given fraction of wetted perimeter


        :param fraction: fraction of wetted perimeter
        :return: cross-section area
        """
        liquid_height = self.height_from_fraction(fraction)
        result = self.cross_area_from_height(liquid_height=liquid_height)

        return result


    def volume_from_height(self, liquid_height: Union[float, np.ndarray], length: Union[float, np.ndarray]):
        """
        Returns the volume in cm^3 for a given liquid height and a given length

        :param liquid_height: liquid level
        :param length: length of the cell (in cm)
        :return:
        """

        cross_area = self.cross_area_from_height(liquid_height=liquid_height)
        volume = cross_area * length

        return volume


    def volume_from_fraction(self, fraction: Union[float, np.ndarray], length: Union[float, np.ndarray]):

        """
        Returns the volume in cm^3 for a given fraction of wetted perimeter and a given length


        :param fraction: fraction of wetted perimeter
        :param length: length of the cell (in cm)
        :return:
        """

        liquid_height = self.height_from_fraction(fraction=fraction)
        cross_area = self.cross_area_from_height(liquid_height=liquid_height)
        volume = cross_area * length

        return volume


    def cross_area_from_height(self, liquid_height: Union[float, np.ndarray, list]):
        """
        Returns the cross-section in cm^2 given the liquid_height


        :param liquid_height: liquid height
        :return: cross-section area
        """
        if isinstance(liquid_height, list):
            liquid_height = np.array(liquid_height)

        alpha = np.arccos(1 - 2 * liquid_height / self.D)
        A = (np.power(self.D, 2) / 4) * (alpha - np.sin(2 * alpha) / 2)

        if self.d > 0.:
            small_pipe_covered_h_fraction = np.minimum(liquid_height / self.d, 1.0)
        else:
            small_pipe_covered_h_fraction = 0.0

        beta = np.arccos(1 - 2*small_pipe_covered_h_fraction)
        a = (np.power(self.d, 2) / 4) * (beta - np.sin(2 * beta) / 2)

        result = A - a
        return result


    def fraction_from_cross_area(self, cross_area: Union[float, np.ndarray]):
        """
        Returns the fraction of wetted perimeter for a given cross-section (in cm^2)

        :param cross_area: cross-area
        :return: fraction of wetted perimeter
        """

        liquid_height = self.height_from_cross_area(cross_areas=cross_area)
        fraction = self.fraction_from_height(liquid_height=liquid_height)

        return fraction


    def fraction_from_height(self, liquid_height: Union[float, np.ndarray]):
        """
        Returns the fraction of wetted perimeter for a given liquid height

        :param liquid_height: liquid height (in cm)
        :return:
        """

        height_fraction = np.minimum(liquid_height / self.D, 1.0)
        alpha = np.arccos(1 - 2 * height_fraction)
        perimeter = alpha * self.D
        result = perimeter / (np.pi * self.D)

        return result


    def height_from_cross_area(self, cross_areas: Union[float, np.ndarray]):
        """
        Returns liquid height for a given cross_are

        ------ DETAILS ------

        This function returns the height of the liquid given the cross-section area. The forward pass is
        easy can be easily computed, the inverse function instead is not easy to find analytically because
        it is a linear combination of a variable and its trigonometric value. To get the most precise solution
        an interpolator is used.
        ----------------------

        :param cross_areas: cross-section area
        :return: height of the liquid
        """

        result = self._height_from_cross_area_interpolator(cross_areas)

        return result


    def cross_section_from_volume(self, volume: Union[float, np.ndarray], length: Union[float, np.ndarray]):
        """
        Returns the cross-section for a given volume


        :param volume: volumes (in cm^3)
        :param length: length of the cell (in cm)
        :return: cross-section (in cm^2)
        """
        cross_sections = volume / length

        return cross_sections


    def height_from_volume(self, volume: Union[float, np.ndarray], length: Union[float, np.ndarray]):
        """
        Returns the height for a given volume

        :param volume: volume (in cm^3)
        :param length: length of the cell (in cm)
        :return: liquid height (in cm)
        """

        cross_sections = self.cross_section_from_volume(volume=volume, length=length)
        liquid_level = self.height_from_cross_area(cross_sections)

        return liquid_level


    def liquid_touching_perimeter_from_cross_area(self, cross_areas: Union[float, np.ndarray], weld_thickness=None):

        heights = self.height_from_cross_area(cross_areas=cross_areas)
        result = self.liquid_touching_perimeter_from_height(heights)

        return result


    def liquid_touching_perimeter_from_height(self, liquid_heights: Union[float, np.ndarray], weld_thickness=None):
        """
        Will compute "touching perimeter" of liquid with walls of big and (possibly) small tube.
        This "touching perimeter" is used in viscosity effects computations.

        If weld_thickness is specified, then the outer and the inner tube are considered to be welded
        on the bottom, meaning that a piece of "area" can't be filled by the liquid, and the contact perimeter
        is bigger than the normal case

        :param liquid_heights:
        :param weld_thickness:
        :return:
        """

        f_w_perimeter = self.fraction_from_height(liquid_height=liquid_heights)
        alpha = f_w_perimeter * np.pi
        P = alpha * self.D


        # ---------------> ADAPT CODE ALSO FOR WELD_THICKNESS
        # if weld_thickness:
        #     y_t, p_t, P_t, A_t = self.weld_bottom_measurements(weld_thickness)


        if self.d > 0.:
            small_pipe_covered_h_fraction = np.minimum(liquid_heights / self.d, 1.0)
        else:
            small_pipe_covered_h_fraction = 0.0

        beta = np.arccos(1 - 2*small_pipe_covered_h_fraction)
        p = beta * self.d
        result = P + p
        return result



    def weld_bottom_measurements(self, weld_thickness: np.ndarray):
        """

        :param weld_thickness: threshold for which the tubes are considered overlapped
        :return: depth of the welded part,
                 the perimeter of the inner tube to subtract for the computation of the hydraulic diameter,
                 the perimeter of the outer tube to subtract for the computation of the hydraulic diameter,
                 welded area to subtract for the computation of the hydraulic diameter
        """

        R = self.D / 2
        r = self.d / 2

        num = np.power(R - r, 2) + np.power(r + weld_thickness, 2) - np.power(R, 2)
        den = 2 * (R - r) * (r + weld_thickness)
        alpha_first = np.arccos(num / den)

        p_h_hidden = 2 * r * (np.pi - alpha_first)

        num = np.power(R - r, 2) + np.power(R, 2) - np.power(r + weld_thickness, 2)
        den = 2 * (R - r) * R
        beta_first = np.arccos(num / den)

        P_h_hidden = 2 * R * beta_first

        depth = r - (r + weld_thickness) * np.cos(np.pi - alpha_first)

        term1 = (np.pi - alpha_first) * np.power(r + weld_thickness, 2) / 2
        term2 = (np.pi - alpha_first) * np.power(r, 2) / 2
        A_h_hidden = term1 - term2

        return depth, p_h_hidden, P_h_hidden, A_h_hidden
