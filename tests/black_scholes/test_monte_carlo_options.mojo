"""Tests for the Black–Scholes Monte Carlo Mojo implementation.

These tests use a very small number of simulations so they run quickly
while still exercising the main pricing functions.
"""

from std.testing import assert_almost_equal, assert_true, TestSuite

from monte_carlo_options import (
    monte_carlo_option_price_mojo,
    monte_carlo_option_price_parallel,
)


def test_single_threaded_price_reasonable() raises:
    """Single-threaded price is finite and positive for at-the-money call."""
    var spot: Float64 = 100.0
    var strike: Float64 = 100.0
    var r: Float64 = 0.05
    var vol: Float64 = 0.20
    var t: Float64 = 1.0
    var n: Int = 10_000

    var price, stderr = monte_carlo_option_price_mojo(spot, strike, r, vol, t, n)

    assert_true(price > 0.0)
    assert_true(stderr > 0.0)


def test_parallel_matches_single_threaded_closely() raises:
    """Parallel price is close to single-threaded for the same seed/params."""
    var spot: Float64 = 100.0
    var strike: Float64 = 100.0
    var r: Float64 = 0.05
    var vol: Float64 = 0.20
    var t: Float64 = 1.0
    var n: Int = 10_000

    # Note: We rely on the underlying RNG and parameters being stable enough
    # that the two implementations agree within a small tolerance.
    var price1, _ = monte_carlo_option_price_mojo(spot, strike, r, vol, t, n)
    var price2, _ = monte_carlo_option_price_parallel(spot, strike, r, vol, t, n)

    assert_almost_equal(price1, price2, rtol=0.05)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
