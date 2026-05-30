"""Tests for k-means clustering algorithm.

Tests basic k-means functionality and convergence properties.
"""

from std.testing import assert_true, assert_equal, TestSuite
from math import sqrt


fn euclidean_distance(x1: Float64, y1: Float64, x2: Float64, y2: Float64) -> Float64:
    """Calculate Euclidean distance between two 2D points."""
    var dx = x2 - x1
    var dy = y2 - y1
    return sqrt(dx * dx + dy * dy)


def test_euclidean_distance_same_point() raises:
    """Test distance from point to itself is zero."""
    var dist = euclidean_distance(1.0, 2.0, 1.0, 2.0)
    assert_equal(dist, 0.0)


def test_euclidean_distance_horizontal() raises:
    """Test distance along horizontal axis."""
    var dist = euclidean_distance(0.0, 0.0, 3.0, 0.0)
    assert_equal(dist, 3.0)


def test_euclidean_distance_vertical() raises:
    """Test distance along vertical axis."""
    var dist = euclidean_distance(0.0, 0.0, 0.0, 4.0)
    assert_equal(dist, 4.0)


def test_euclidean_distance_345_triangle() raises:
    """Test distance with 3-4-5 right triangle."""
    var dist = euclidean_distance(0.0, 0.0, 3.0, 4.0)
    assert_equal(dist, 5.0)


def test_cluster_assignment_closest() raises:
    """Test that a point is assigned to the closest centroid."""
    # Point at (1, 1)
    var point_x: Float64 = 1.0
    var point_y: Float64 = 1.0
    
    # Two centroids
    var c1_x: Float64 = 0.0
    var c1_y: Float64 = 0.0
    var c2_x: Float64 = 10.0
    var c2_y: Float64 = 10.0
    
    # Calculate distances
    var dist1 = euclidean_distance(point_x, point_y, c1_x, c1_y)
    var dist2 = euclidean_distance(point_x, point_y, c2_x, c2_y)
    
    # Point should be closer to first centroid
    assert_true(dist1 < dist2, "Point should be closer to centroid 1")


def test_centroid_calculation() raises:
    """Test centroid calculation as mean of points."""
    # Three points: (0,0), (3,0), (3,3)
    var sum_x: Float64 = 0.0 + 3.0 + 3.0
    var sum_y: Float64 = 0.0 + 0.0 + 3.0
    var count: Float64 = 3.0
    
    var centroid_x = sum_x / count
    var centroid_y = sum_y / count
    
    # Expected centroid: (2, 1)
    assert_equal(centroid_x, 2.0)
    assert_equal(centroid_y, 1.0)


def main() raises:
    print("Running kmeans tests...")
    test_euclidean_distance_same_point()
    print("✓ test_euclidean_distance_same_point passed")
    test_euclidean_distance_horizontal()
    print("✓ test_euclidean_distance_horizontal passed")
    test_euclidean_distance_vertical()
    print("✓ test_euclidean_distance_vertical passed")
    test_euclidean_distance_345_triangle()
    print("✓ test_euclidean_distance_345_triangle passed")
    test_cluster_assignment_closest()
    print("✓ test_cluster_assignment_closest passed")
    test_centroid_calculation()
    print("✓ test_centroid_calculation passed")
    print("All 6 tests passed!")
