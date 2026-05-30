"""Tests for Monte Carlo Pi estimation.

Tests the core logic of point-in-circle detection and pi estimation.
"""

from std.testing import assert_true, assert_equal, TestSuite
from math import pi


fn is_inside_circle(x: Float64, y: Float64) -> Bool:
    """Check if a point (x, y) is inside the unit circle."""
    return x * x + y * y <= 1.0


def test_origin_inside_circle() raises:
    """Test that the origin is inside the unit circle."""
    assert_true(is_inside_circle(0.0, 0.0), "Origin should be inside circle")


def test_unit_point_on_circle() raises:
    """Test that (1, 0) is on the circle boundary."""
    assert_true(is_inside_circle(1.0, 0.0), "Point (1,0) should be on/inside circle")


def test_point_outside_circle() raises:
    """Test that points far from origin are outside the circle."""
    assert_true(not is_inside_circle(2.0, 2.0), "Point (2,2) should be outside circle")


def test_diagonal_point_outside() raises:
    """Test that (1, 1) is outside the unit circle."""
    # sqrt(1^2 + 1^2) = sqrt(2) > 1
    assert_true(not is_inside_circle(1.0, 1.0), "Point (1,1) should be outside circle")


def test_pi_estimation_bounds() raises:
    """Test that pi estimation formula gives reasonable bounds."""
    # If 75% of points are inside, estimate should be 3.0
    var inside_count: Int = 75
    var total_count: Int = 100
    var estimate = 4.0 * Float64(inside_count) / Float64(total_count)
    
    assert_equal(estimate, 3.0)


def test_perfect_circle_coverage() raises:
    """Test the theoretical maximum: all points inside."""
    var inside_count: Int = 100
    var total_count: Int = 100
    var estimate = 4.0 * Float64(inside_count) / Float64(total_count)
    
    # This would give 4.0 (but is unrealistic for random points)
    assert_equal(estimate, 4.0)


def test_quarter_circle_area() raises:
    """Test that quarter circle area is pi/4."""
    # Area of quarter circle with radius 1 is pi/4 ≈ 0.7854
    var theoretical_ratio = pi / 4.0
    
    # Should be between 0.7 and 0.8
    assert_true(theoretical_ratio > 0.7 and theoretical_ratio < 0.8,
                "Quarter circle ratio should be approximately 0.7854")


def test_circle_points_at_quadrants() raises:
    """Test points at different quadrants of the unit circle."""
    # Points just inside the circle in each quadrant
    var radius: Float64 = 0.7  # Well inside circle
    
    assert_true(is_inside_circle(radius, 0.0), "Point on positive x-axis")
    assert_true(is_inside_circle(0.0, radius), "Point on positive y-axis")
    assert_true(is_inside_circle(-radius, 0.0), "Point on negative x-axis")
    assert_true(is_inside_circle(0.0, -radius), "Point on negative y-axis")


def main() raises:
    print("Running mcpi tests...")
    test_origin_inside_circle()
    print("✓ test_origin_inside_circle passed")
    test_unit_point_on_circle()
    print("✓ test_unit_point_on_circle passed")
    test_point_outside_circle()
    print("✓ test_point_outside_circle passed")
    test_diagonal_point_outside()
    print("✓ test_diagonal_point_outside passed")
    test_pi_estimation_bounds()
    print("✓ test_pi_estimation_bounds passed")
    test_perfect_circle_coverage()
    print("✓ test_perfect_circle_coverage passed")
    test_quarter_circle_area()
    print("✓ test_quarter_circle_area passed")
    test_circle_points_at_quadrants()
    print("✓ test_circle_points_at_quadrants passed")
    print("All 8 tests passed!")
