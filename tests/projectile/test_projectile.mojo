"""Tests for the projectile simulation.

Tests the Vector2D struct and basic physics operations.
"""

from std.testing import assert_equal, assert_true, TestSuite
from math import sqrt


# Define Vector2D locally for testing (mirrors the src implementation)
@fieldwise_init
struct Vector2D(ImplicitlyCopyable, Movable):
    var x: Float64
    var y: Float64
    
    fn __add__(self, other: Self) -> Self:
        return Vector2D(self.x + other.x, self.y + other.y)
    
    fn __mul__(self, scalar: Float64) -> Self:
        return Vector2D(self.x * scalar, self.y * scalar)
    
    fn magnitude(self) -> Float64:
        return sqrt(self.x * self.x + self.y * self.y)


def test_vector_addition() raises:
    """Test vector addition operator."""
    var v1 = Vector2D(1.0, 2.0)
    var v2 = Vector2D(3.0, 4.0)
    var result = v1 + v2
    
    assert_equal(result.x, 4.0)
    assert_equal(result.y, 6.0)


def test_vector_scalar_multiplication() raises:
    """Test vector scalar multiplication."""
    var v = Vector2D(2.0, 3.0)
    var result = v * 2.0
    
    assert_equal(result.x, 4.0)
    assert_equal(result.y, 6.0)


def test_vector_magnitude() raises:
    """Test vector magnitude calculation."""
    var v = Vector2D(3.0, 4.0)
    var mag = v.magnitude()
    
    # 3-4-5 triangle
    assert_equal(mag, 5.0)


def test_zero_vector_magnitude() raises:
    """Test magnitude of zero vector."""
    var v = Vector2D(0.0, 0.0)
    var mag = v.magnitude()
    
    assert_equal(mag, 0.0)


def test_gravity_effect() raises:
    """Test that gravity acts downward."""
    var velocity = Vector2D(5.0, 10.0)
    var gravity = Vector2D(0.0, -9.81)
    var dt: Float64 = 0.1
    
    var new_velocity = velocity + gravity * dt
    
    # y-velocity should decrease
    assert_true(new_velocity.y < velocity.y, "Gravity should reduce upward velocity")
    # x-velocity should remain unchanged
    assert_equal(new_velocity.x, velocity.x)


def main() raises:
    print("Running projectile tests...")
    test_vector_addition()
    print("✓ test_vector_addition passed")
    test_vector_scalar_multiplication()
    print("✓ test_vector_scalar_multiplication passed")
    test_vector_magnitude()
    print("✓ test_vector_magnitude passed")
    test_zero_vector_magnitude()
    print("✓ test_zero_vector_magnitude passed")
    test_gravity_effect()
    print("✓ test_gravity_effect passed")
    print("All 5 tests passed!")
