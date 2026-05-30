"""Tests for the system information utility.

Verifies that system information can be queried without error.
"""

from std.testing import assert_true, TestSuite
from sys.info import (
    is_64bit,
    num_logical_cores,
    num_physical_cores,
)


def test_is_64bit() raises:
    """Test that 64-bit detection returns a valid boolean."""
    # On modern systems this should always be true
    var result = is_64bit()
    assert_true(result, "Expected 64-bit system")


def test_core_counts() raises:
    """Test that core count queries return positive values."""
    var logical = num_logical_cores()
    var physical = num_physical_cores()
    
    assert_true(logical > 0, "Logical cores must be positive")
    assert_true(physical > 0, "Physical cores must be positive")
    assert_true(logical >= physical, "Logical cores should be >= physical cores")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
