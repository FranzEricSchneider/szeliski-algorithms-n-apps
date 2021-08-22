"""
The purpose of the Newton-Raphson method is to iteratively solve for the roots
of polynomials. I was curious to get a feel for this (and produce interesting
graphs) so I wanted to give it a shot.

Inspired by these pages:
    https://www.chiark.greenend.org.uk/~sgtatham/newton/
    https://mathworld.wolfram.com/NewtonsMethod.html
"""


import numpy


# Totally arbitrary
CONVERGE_THRESHOLD = 1e-3

POW4_MINUS1 = {
    "f": lambda x: x**4 - 1,
    "fprime": lambda x: 4 * x**3,
}


def step(array, f, fprime):
    """
    Take an array of (x, i) positions and step forward in time using the
        e_t = - f(X_t) / f'(X_t)
        X_t+1 = X_t + e_t
    """

    e = -f(array) / fprime(array)
    stepped = array + e

    # TODO
    nans = numpy.isnan(stepped)
    if nans.any():
        stepped[nans] = array[nans]

    return stepped


def initialize(side_len, step=1):

    # We need an odd-numbered side length of 3 or higher in order to have x
    # and y = 0 axes in the center
    assert side_len > 2
    assert side_len % 2 == 1

    extent = side_len // 2
    array = numpy.zeros((side_len, side_len), dtype=complex)
    for i, xvalue in enumerate(range(-extent, extent + 1)):
        for j, ivalue in enumerate(range(-extent, extent + 1)):
            array[i, j] = step * (xvalue + ivalue * 1j)
    return array


def run(side_len, function):
    array = initialize(side_len)
    not_done = numpy.ones(array.shape, dtype=bool)
    converge_count = numpy.zeros(array.shape, dtype=int)

    stepped = array.copy()
    while numpy.any(not_done):
        stepped[not_done] = step(array[not_done], **function)
        not_done[not_done] = numpy.abs(
            stepped[not_done] - array[not_done]
        ) > CONVERGE_THRESHOLD
        array[not_done] = stepped[not_done]
        converge_count[not_done] += 1

    return array, converge_count


if __name__ == "__main__":
    function = {
        "f": lambda x: x**2,
        "fprime": lambda x: 2*x,
    }
    array, converge_count = run(side_len=5, function=POW4_MINUS1)
    print(f"array: {array}")
    print(f"converge_count: {converge_count}")
    import ipdb; ipdb.set_trace()
