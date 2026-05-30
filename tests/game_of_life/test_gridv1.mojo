"""Basic tests for the Grid[rows, cols] implementation in gridv1.mojo.

These focus on small, deterministic patterns (still life and oscillator)
so they run quickly and don't depend on any external files.
"""

from std.testing import assert_equal, TestSuite

from gridv1 import Grid


def test_block_still_life() raises:
    """A 2x2 block should remain unchanged after one generation."""
    var g = Grid[4, 4]()

    # 2x2 block at (1,1), (1,2), (2,1), (2,2)
    g[1, 1] = 1
    g[1, 2] = 1
    g[2, 1] = 1
    g[2, 2] = 1

    var g_next = g.evolve()

    # The block should be preserved exactly.
    assert_equal(g_next[1, 1], 1)
    assert_equal(g_next[1, 2], 1)
    assert_equal(g_next[2, 1], 1)
    assert_equal(g_next[2, 2], 1)


def test_blinker_oscillator_period_2() raises:
    """A simple blinker should alternate between vertical and horizontal."""
    var g = Grid[5, 5]()

    # Vertical line of three cells in the centre column.
    centre_col: Int = 2
    g[1, centre_col] = 1
    g[2, centre_col] = 1
    g[3, centre_col] = 1

    var g_next = g.evolve()

    # After one evolution, it should become a horizontal line.
    assert_equal(g_next[2, 1], 1)
    assert_equal(g_next[2, 2], 1)
    assert_equal(g_next[2, 3], 1)

    # And the original vertical cells should now be dead.
    assert_equal(g_next[1, centre_col], 0)
    assert_equal(g_next[3, centre_col], 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
