"""Tests for the Advent of Code Day 1 Mojo implementation.

These are intentionally lightweight and focus on the pure helper
functions rather than the I/O-heavy `main()`.
"""

from std.testing import assert_equal, TestSuite

from day1 import load_rotations, update_position, N_POSITION


def test_update_position_wraps_forward() raises:
    """Positive rotation wraps around the dial correctly."""
    # Starting near the end of the dial and rotating forward should wrap.
    start: Int = 95
    rotation: Int = 10
    result = update_position(start, rotation, N_POSITION)
    # (95 + 10) % 100 == 5
    assert_equal(result, 5)


def test_update_position_wraps_backward() raises:
    """Negative rotation wraps backwards correctly."""
    start: Int = 3
    rotation: Int = -10
    result = update_position(start, rotation, N_POSITION)
    # (3 - 10) % 100 == 93
    assert_equal(result, 93)


# Note: We avoid testing `load_rotations` against real files here to keep the
# tests hermetic. If desired, we could add dedicated test input files under a
# tests/ tree and point load_rotations at those.


def main() raises:
    # Discover and run all test_* functions in this module.
    TestSuite.discover_tests[__functions_in_module()]().run()
