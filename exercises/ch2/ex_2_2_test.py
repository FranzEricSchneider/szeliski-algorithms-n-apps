import numpy
import pytest


from ex_2_2 import (
    closest,
    Rectangle,
)


class TestClosest:

    def test_closest_nothing(self):
        """If no rectangles are given, you get Nones back"""
        assert (None, None, None) == closest(numpy.array([1, 1]), [])
        assert (None, None, None) == closest(numpy.array([1, 1]), None)

    @pytest.mark.parametrize("point,index,match_index", (
        # First rectangle
        (numpy.array([51, 50]),   0, 0),
        (numpy.array([149, 52]),  0, 1),
        (numpy.array([150, 150]), 0, 2),
        # Second rectangle
        (numpy.array([96, 100]),  1, 0),
        (numpy.array([99, 103]),  1, 3),
        # Third rectangle
        (numpy.array([24, 78]),   2, 0),
        (numpy.array([66, 130]),  2, 2),
        (numpy.array([18, 139]),  2, 3),
    ))
    def test_closest_selection(self, point, index, match_index):
        rectangles = [
            Rectangle(50, 50, 100, 100),
            Rectangle(100, 100, 2, 2),
            Rectangle(20, 80, 40, 55),
        ]
        _, rectangle, point_index = closest(point, rectangles)
        assert rectangle is rectangles[index]
        assert point_index == match_index

    def test_closest_dist(self):
        """Test distance measurement."""

        # Hypotenuse of (1, 1) is sqrt(2)
        dist, _, _ = closest(numpy.array([11, 21]),
                             [Rectangle(10, 20, 100, 100)])
        assert isinstance(dist, float)
        assert numpy.isclose(dist, numpy.sqrt(2))

        # Hypotenuse of (3, 4) is 5
        dist, _, _ = closest(numpy.array([23, 14]),
                             [Rectangle(20, 10, 100, 100)])
        assert isinstance(dist, float)
        assert numpy.isclose(dist, 5)


    def test_closest_tie(self):
        """Test when there's a tie."""
        rectangles = [
            Rectangle(0, 0, 100, 90),
            Rectangle(102, 92, 2, 2),
        ]

        # If you're in the middle of multiple corners on one rectangle, choose
        # the first corner
        dist, rectangle, point_index = closest(numpy.array([103, 93]), rectangles)
        assert numpy.isclose(dist, numpy.sqrt(2))
        assert rectangle is rectangles[1]
        assert point_index == 0

        # If you're in between two rectangles, choose the first one
        dist, rectangle, point_index = closest(numpy.array([101, 91]), rectangles)
        assert numpy.isclose(dist, numpy.sqrt(2))
        assert rectangle is rectangles[0]
        assert point_index == 2
