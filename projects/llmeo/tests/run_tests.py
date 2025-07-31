#!/usr/bin/env python3
"""
Test runner for LLMEO project

This script runs all tests and provides options for different test types.
"""

import sys
import os
import subprocess
import argparse

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
sys.path.insert(0, project_root)


def run_tests(test_type="all", verbose=False, coverage=False):
    """Run tests with specified options"""
    
    # Base command
    cmd = [sys.executable, "-m"]
    
    # Choose test runner
    try:
        import pytest
        cmd.extend(["pytest"])
        
        # Add test directory
        test_dir = os.path.dirname(os.path.abspath(__file__))
        cmd.append(test_dir)
        
        # Add verbose flag
        if verbose:
            cmd.append("-v")
        
        # Add coverage
        if coverage:
            cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
        
        # Filter by test type
        if test_type == "unit":
            cmd.extend(["-m", "unit"])
        elif test_type == "integration":
            cmd.extend(["-m", "integration"])
        elif test_type == "slow":
            cmd.extend(["-m", "slow"])
        elif test_type != "all":
            print(f"Unknown test type: {test_type}")
            return 1
            
    except ImportError:
        # Fall back to unittest
        cmd.extend(["unittest"])
        
        if test_type == "all":
            cmd.extend(["discover", "-s", "tests", "-p", "test_*.py"])
        else:
            print("Test filtering only available with pytest")
            return 1
        
        if verbose:
            cmd.append("-v")
    
    # Run the tests
    print(f"Running {'all' if test_type == 'all' else test_type} tests...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Run LLMEO tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test types:
  all          Run all tests (default)
  unit         Run only unit tests
  integration  Run only integration tests
  slow         Run only slow tests

Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py -t unit -v         # Run unit tests with verbose output
  python run_tests.py -t integration -c  # Run integration tests with coverage
        """
    )
    
    parser.add_argument(
        "-t", "--type",
        choices=["all", "unit", "integration", "slow"],
        default="all",
        help="Type of tests to run (default: all)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Run with coverage (requires pytest-cov)"
    )
    
    args = parser.parse_args()
    
    # Run tests
    exit_code = run_tests(
        test_type=args.type,
        verbose=args.verbose,
        coverage=args.coverage
    )
    
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()