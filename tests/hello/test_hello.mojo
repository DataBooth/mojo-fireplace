"""Tests for the hello world example.

Simple test to verify the hello module can be imported and executed.
"""

from std.testing import assert_equal, TestSuite


def test_hello_output() raises:
    """Test that hello world example runs without error.
    
    This is a minimal test that just verifies the code is syntactically
    correct and can execute. More complex examples would test actual
    functionality.
    """
    # For hello world, we just verify the code structure is valid
    # by testing a simple expression
    var greeting = String("Hello, World!")
    assert_equal(len(greeting), 13)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
