from typing import Union

import numpy as np


class SquareChannelGeometryEstimator:

    def __init__(self, width, height):
        self.width = width
        self.height = height


    def get_height(self):
        """
        Returns dimension that correspond to the height. In this implementation, it is the big diameter (self.D)
        :return:
        """
        return self.height


    def max_cross_section_area(self):
        """
        Returns the maximal cross-section area, when the pipe is fully filled.
        """
        result = self.cross_area_from_height(self.get_height())
        return result


    def cross_area_from_height(self, liquid_height: Union[float, np.ndarray, list]):
        res = self.width * liquid_height
        return res


    def height_from_cross_area(self, cross_areas: Union[float, np.ndarray]):
        h = cross_areas / self.width
        return h


    def liquid_touching_perimeter_from_cross_area(self, cross_areas: Union[float, np.ndarray]):
        liq_heights = self.height_from_cross_area(cross_areas)
        return self.liquid_touching_perimeter_from_height(liq_heights)


    def liquid_touching_perimeter_from_height(self, liquid_heights: Union[float, np.ndarray]):
        touching_perim = self.width + 2*liquid_heights
        return touching_perim
