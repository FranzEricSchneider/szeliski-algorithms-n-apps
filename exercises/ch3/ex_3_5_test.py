import numpy
import pytest

from ex_3_5 import construct_equations, phi


class TestPhi:
    @pytest.mark.parametrize("len_s, p, N", (
        (10, 2, 9),
        (10, 9, 2),
        (5, 3, 3),
    ))
    def test_length_assertion_error(self, len_s, p, N):
        """When given s of insufficient length, raise."""
        s = [1] * len_s
        with pytest.raises(AssertionError):
            phi(s=s, p=p, N=N, i=0, k=0)

    @pytest.mark.parametrize("i, k", (
        ( 4,  0),  # invalid i
        (-1,  0),  # invalid i
        ( 0,  5),  # invalid k
        ( 1, -1),  # invalid k
        ( 4, -1),  # invalid both
    ))
    def test_value_valid_assertion_error(self, i, k):
        """Both i and k values need to be bounded by [0, p]"""
        s = [1] * 20
        with pytest.raises(AssertionError):
            phi(s=s, p=3, N=4, i=i, k=k)

    @pytest.mark.parametrize("s, i, k, expected", (
        ([1, 9, 5, 3, 2, 6, 1, 2], 0, 0, 54),  # 3*3 + 2*2 + 6*6 + 1 + 2*2
        ([1, 9, 5, 3, 2, 6, 1, 2], 2, 0, 69),  # 9*3 + 5*2 + 3*6 + 2 + 6*2
        ([1, 9, 5, 3, 2, 6, 1, 2], 0, 2, 69),  # 3*9 + 2*5 + 6*3 + 2 + 2*6
        ([1, 9, 5, 3, 2, 6, 1, 2], 3, 1, 62),  # 1*5 + 9*3 + 5*2 + 3*6 + 2
    ))
    def test_simple_situations(self, s, i, k, expected):
        """Test some hand-verifiable situations with fixed p/N."""
        assert phi(s=s, p=3, N=5, i=i, k=k) == expected

    def test_correct_type(self):
        """Whether passed int or uint8 values, output should still be int."""

        def check(s):
            output = phi(s=s, p=3, N=5, i=0, k=0)
            assert output == 54
            assert isinstance(output, int) or isinstance(output, numpy.int64)

        # Known working set
        s = [1, 9, 5, 3, 2, 6, 1, 2]
        check(s)

        # Try it again with s as an array of uint8
        check(numpy.array(s, dtype=numpy.uint8))

    def test_overlong_s(self):
        """
        When s is longer than necessary we want to use the FIRST p+N values,
        because we have constructed phi such that new values should appear at
        the beginning of the array.
        """

        # Expected = 1*3 + 2*4 + 3*5 + 4*6 = 50, which doesn't use the full s
        assert phi(s=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], p=2, N=4, i=2, k=0) == 50


class TestConstructEquations:
    def test_simple_scenario(self):
        """Run a simple version of eq. (18) by hand and check it."""
        matrix, vector = construct_equations(
            s=numpy.array([1, 7, 4, 2, 6, 3, 1, 7, 4, 3], dtype=numpy.uint8),
            p=3,
            N=4,
        )
        assert isinstance(matrix, numpy.ndarray)
        assert isinstance(vector, numpy.ndarray)
        assert matrix.dtype == numpy.int64
        assert vector.dtype == numpy.int64

        assert numpy.allclose(
            matrix,
            numpy.array([
                [16 + 4 + 36 + 9 , 28 + 8 + 12 + 18, 4 + 14 + 24 + 6],
                [28 + 8 + 12 + 18, 49 + 16 + 4 + 36, 7 + 28 + 8 + 12],
                [4 + 14 + 24 + 6 ,  7 + 28 + 8 + 12, 1 + 49 + 16 + 4]
            ])
        )
        assert numpy.allclose(
            vector,
            numpy.array([-(4*2 + 2*6 + 6*3 + 3*1),
                         -(7*2 + 4*6 + 2*3 + 6*1),
                         -(1*2 + 7*6 + 4*3 + 2*1)])
        )