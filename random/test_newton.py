import numpy
import pytest

from newton import CONVERGE_THRESHOLD, initialize, run, step


@pytest.fixture
def grid():
    return numpy.array([
        [-1 + 1j, 0 + 1j, 1 + 1j],
        [-1,      0,      1     ],
        [-1 - 1j, 0 - 1j, 1 - 1j],
    ])


@pytest.fixture
def line():
    return {
        "f": lambda x: x,
        "fprime": lambda x: 1,
    }


@pytest.fixture
def parabola():
    return {
        "f": lambda x: x**2,
        "fprime": lambda x: 2*x,
    }


class TestStep:
    def test_line(self, grid, line):
        output = step(grid, **line)
        assert (output == numpy.zeros(grid.shape)).all()

    def test_parabola(self, grid, parabola):
        output = step(grid, **parabola)
        # Note that in a simple step [1, 1] would become nan, since it started
        # off at a root. It was automatically caught and set to the previous
        # value.
        expected = numpy.array([
            [-0.5 + 0.5j,  0.5j, 0.5 + 0.5j],
            [-0.5,         0,    0.5       ],
            [-0.5 - 0.5j, -0.5j, 0.5 - 0.5j],
        ])
        assert (output == expected).all()


@pytest.fixture
def init_grid():
    return numpy.array([
        [-1 - 1j, -1, -1 + 1j],
        [ 0 - 1j,  0,  0 + 1j],
        [ 1 - 1j,  1,  1 + 1j],
    ])


class TestInitialize:

    @pytest.mark.parametrize("side", (-5, -2, -1, 0, 1, 2, 4, 6, 10))
    def test_raise(self, side):
        with pytest.raises(AssertionError):
            initialize(side_len=side)

    def test_size3(self, init_grid):
        assert (initialize(side_len=3) == init_grid).all()

    @pytest.mark.parametrize("step", (-1, 2, 1, 0.5, 1.5, 10))
    def test_step(self, init_grid, step):
        assert numpy.allclose(initialize(side_len=3, step=step),
                              step * init_grid)


class TestRun:

    def test_line(self, line):
        array, converge_count = run(side_len=5, function=line)
        assert array.shape == (5, 5)
        assert numpy.allclose(array, numpy.zeros(array.shape))
        assert numpy.allclose(converge_count, numpy.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]))

    def test_parabola(self, parabola):
        array, converge_count = run(side_len=5, function=parabola)
        # This is a little sketchy, but I'm basically saying that we've
        # converged when the step size gets small enough, which doesn't
        # necessarily mean that the goal is that close. I fudged it a little to
        # say that we're within 10x of the threshold from the goal.
        assert numpy.all(numpy.abs(array) < (10 * CONVERGE_THRESHOLD))
        # Experimentally determined
        assert numpy.allclose(converge_count, numpy.array([
            [11, 11, 10, 11, 11],
            [11, 10,  9, 10, 11],
            [10,  9,  0,  9, 10],
            [11, 10,  9, 10, 11],
            [11, 11, 10, 11, 11],
        ]))