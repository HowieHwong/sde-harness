#!/usr/bin/env python3
"""
Test runner for SDE-Harness framework

This script runs all tests and provides options for different test types.
"""

import sys
import os
import subprocess
import argparse

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def run_tests(test_type="all", verbose=False, coverage=False, pattern=None):
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
            cmd.extend(["--cov=sde_harness", "--cov-report=html", "--cov-report=term"])
        
        # Add pattern matching
        if pattern:
            cmd.extend(["-k", pattern])
        
        # Filter by test type
        if test_type == "unit":
            cmd.extend(["-m", "unit"])
        elif test_type == "integration":
            cmd.extend(["-m", "integration"])
        elif test_type == "slow":
            cmd.extend(["-m", "slow"])
        elif test_type == "requires_api":
            cmd.extend(["-m", "requires_api"])
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
    if pattern:
        print(f"Pattern filter: {pattern}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def run_specific_module(module_name, verbose=False):
    """Run tests for a specific module"""
    
    # Map module names to test files
    module_map = {
        'generation': 'test_core_generation.py',
        'oracle': 'test_core_oracle.py', 
        'prompt': 'test_core_prompt.py',
        'workflow': 'test_core_workflow.py',
        'base': 'test_base_classes.py',
        'core': 'test_core_*.py'
    }
    
    if module_name not in module_map:
        print(f"Unknown module: {module_name}")
        print(f"Available modules: {', '.join(module_map.keys())}")
        return 1
    
    test_file = module_map[module_name]
    test_path = os.path.join(os.path.dirname(__file__), test_file)
    
    if '*' in test_file:
        # Use pytest for pattern matching
        try:
            import pytest
            cmd = [sys.executable, "-m", "pytest", test_path.replace('*', '*')]
            if verbose:
                cmd.append("-v")
        except ImportError:
            print("pytest required for pattern matching")
            return 1
    else:
        # Use unittest for specific files
        if os.path.exists(test_path):
            cmd = [sys.executable, "-m", "unittest", f"tests.{test_file[:-3]}"]
            if verbose:
                cmd.append("-v")
        else:
            print(f"Test file not found: {test_path}")
            return 1
    
    print(f"Running tests for module: {module_name}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def check_dependencies():
    """Check test dependencies"""
    print("Checking test dependencies...")
    
    required_packages = [
        'pytest',
        'unittest',
        'yaml'
    ]
    
    optional_packages = [
        'pytest-cov',
        'pytest-timeout',
        'pytest-mock'
    ]
    
    missing_required = []
    missing_optional = []
    
    # Check required packages
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_required.append(package)
            print(f"❌ {package} (required)")
    
    # Check optional packages
    for package in optional_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_optional.append(package)
            print(f"⚠️  {package} (optional)")
    
    if missing_required:
        print(f"\nMissing required packages: {', '.join(missing_required)}")
        print("Install with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"\nMissing optional packages: {', '.join(missing_optional)}")
        print("Install with: pip install " + " ".join(missing_optional))
    
    print("\n✅ All required dependencies available")
    return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Run SDE-Harness tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test types:
  all          Run all tests (default)
  unit         Run only unit tests
  integration  Run only integration tests
  slow         Run only slow tests
  requires_api Run only tests that require API keys

Modules:
  generation   Run tests for Generation class
  oracle       Run tests for Oracle class
  prompt       Run tests for Prompt class
  workflow     Run tests for Workflow class
  base         Run tests for base classes
  core         Run all core module tests

Examples:
  python run_tests.py                          # Run all tests
  python run_tests.py -t unit -v               # Run unit tests with verbose output
  python run_tests.py -t integration -c        # Run integration tests with coverage
  python run_tests.py -m generation            # Run only Generation tests
  python run_tests.py -p "test_oracle"         # Run tests matching pattern
  python run_tests.py --check-deps             # Check test dependencies
        """
    )
    
    parser.add_argument(
        "-t", "--type",
        choices=["all", "unit", "integration", "slow", "requires_api"],
        default="all",
        help="Type of tests to run (default: all)"
    )
    
    parser.add_argument(
        "-m", "--module",
        choices=["generation", "oracle", "prompt", "workflow", "base", "core"],
        help="Run tests for specific module"
    )
    
    parser.add_argument(
        "-p", "--pattern",
        help="Run tests matching pattern"
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
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check test dependencies"
    )
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        deps_ok = check_dependencies()
        sys.exit(0 if deps_ok else 1)
    
    # Run specific module tests
    if args.module:
        exit_code = run_specific_module(args.module, args.verbose)
    else:
        # Run tests with filters
        exit_code = run_tests(
            test_type=args.type,
            verbose=args.verbose,
            coverage=args.coverage,
            pattern=args.pattern
        )
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()