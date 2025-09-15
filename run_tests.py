#!/usr/bin/env python3
"""
Test Runner for CAPSTONE-LAZARUS
===============================

Comprehensive test runner with coverage reporting and parallel execution.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import unittest
import time
from typing import List, Optional


def setup_test_environment():
    """Set up the test environment"""
    # Add src to Python path
    project_root = Path(__file__).parent.parent
    src_path = project_root / 'src'
    sys.path.insert(0, str(src_path))
    
    # Set environment variables for testing
    os.environ['TESTING'] = 'true'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU for tests
    os.environ['PYTHONPATH'] = str(src_path)
    
    print("🔧 Test environment set up")


def install_test_dependencies():
    """Install test dependencies if needed"""
    test_deps = [
        'pytest',
        'pytest-cov',
        'pytest-xdist',
        'coverage',
        'unittest-xml-reporting'
    ]
    
    print("📦 Checking test dependencies...")
    
    missing_deps = []
    for dep in test_deps:
        try:
            __import__(dep.replace('-', '_'))
        except ImportError:
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"Installing missing dependencies: {', '.join(missing_deps)}")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', *missing_deps
        ], check=True)
    else:
        print("✅ All test dependencies are available")


def run_unittest_suite(test_dir: Path, pattern: str = 'test*.py', verbosity: int = 2) -> bool:
    """Run unittest suite"""
    print(f"🧪 Running unittest suite from {test_dir}")
    
    # Discover tests
    loader = unittest.TestLoader()
    
    try:
        suite = loader.discover(str(test_dir), pattern=pattern)
        
        # Run tests
        runner = unittest.TextTestRunner(
            verbosity=verbosity,
            buffer=True,
            stream=sys.stdout
        )
        
        result = runner.run(suite)
        
        # Report results
        print(f"\n📊 Test Results:")
        print(f"   Tests run: {result.testsRun}")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
        print(f"   Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
        
        if result.failures:
            print(f"\n❌ Failures:")
            for test, traceback in result.failures:
                print(f"   {test}: {traceback.split('AssertionError:')[-1].strip()}")
        
        if result.errors:
            print(f"\n💥 Errors:")
            for test, traceback in result.errors:
                print(f"   {test}: {traceback.split('Exception:')[-1].strip()}")
        
        return len(result.failures) == 0 and len(result.errors) == 0
        
    except Exception as e:
        print(f"❌ Error running unittest suite: {e}")
        return False


def run_pytest_suite(test_dir: Path, coverage: bool = True, parallel: bool = False) -> bool:
    """Run pytest suite with optional coverage and parallel execution"""
    print(f"🧪 Running pytest suite from {test_dir}")
    
    cmd = [sys.executable, '-m', 'pytest', str(test_dir), '-v']
    
    # Add coverage
    if coverage:
        cmd.extend(['--cov=src', '--cov-report=term-missing', '--cov-report=html:htmlcov'])
    
    # Add parallel execution
    if parallel:
        import multiprocessing
        num_workers = max(1, multiprocessing.cpu_count() // 2)
        cmd.extend(['-n', str(num_workers)])
    
    # Add XML output for CI
    cmd.extend(['--junitxml=test-results.xml'])
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        success = result.returncode == 0
        
        if success:
            print("✅ All pytest tests passed!")
        else:
            print("❌ Some pytest tests failed")
        
        return success
        
    except FileNotFoundError:
        print("⚠️  pytest not available, falling back to unittest")
        return False
    except Exception as e:
        print(f"❌ Error running pytest: {e}")
        return False


def run_linting_checks() -> bool:
    """Run code linting checks"""
    print("🔍 Running linting checks...")
    
    checks = []
    
    # Flake8
    try:
        result = subprocess.run([
            sys.executable, '-m', 'flake8', 'src', 'tests',
            '--max-line-length=100',
            '--extend-ignore=E203,W503'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  ✅ flake8 passed")
            checks.append(True)
        else:
            print(f"  ❌ flake8 failed:\n{result.stdout}")
            checks.append(False)
            
    except FileNotFoundError:
        print("  ⚠️  flake8 not available")
        checks.append(True)
    
    # Black (formatting check)
    try:
        result = subprocess.run([
            sys.executable, '-m', 'black', '--check', 'src', 'tests'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  ✅ black formatting check passed")
            checks.append(True)
        else:
            print(f"  ❌ black formatting issues found")
            checks.append(False)
            
    except FileNotFoundError:
        print("  ⚠️  black not available")
        checks.append(True)
    
    # isort (import sorting check)
    try:
        result = subprocess.run([
            sys.executable, '-m', 'isort', '--check-only', 'src', 'tests'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  ✅ isort check passed")
            checks.append(True)
        else:
            print(f"  ❌ isort issues found")
            checks.append(False)
            
    except FileNotFoundError:
        print("  ⚠️  isort not available")
        checks.append(True)
    
    return all(checks)


def run_security_checks() -> bool:
    """Run security checks"""
    print("🔒 Running security checks...")
    
    # Bandit security linter
    try:
        result = subprocess.run([
            sys.executable, '-m', 'bandit', '-r', 'src',
            '-f', 'json', '-o', 'security-report.json'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  ✅ bandit security check passed")
            return True
        else:
            print(f"  ❌ bandit found security issues")
            return False
            
    except FileNotFoundError:
        print("  ⚠️  bandit not available")
        return True
    except Exception as e:
        print(f"  ❌ Error running security checks: {e}")
        return False


def run_type_checks() -> bool:
    """Run type checking with mypy"""
    print("🔍 Running type checks...")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'mypy', 'src',
            '--ignore-missing-imports',
            '--no-strict-optional'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  ✅ mypy type checking passed")
            return True
        else:
            print(f"  ❌ mypy found type issues:\n{result.stdout}")
            return False
            
    except FileNotFoundError:
        print("  ⚠️  mypy not available")
        return True
    except Exception as e:
        print(f"  ❌ Error running type checks: {e}")
        return False


def generate_coverage_report():
    """Generate coverage report"""
    print("📊 Generating coverage report...")
    
    try:
        # Generate terminal report
        subprocess.run([sys.executable, '-m', 'coverage', 'report'], check=True)
        
        # Generate HTML report
        subprocess.run([sys.executable, '-m', 'coverage', 'html'], check=True)
        print("  ✅ Coverage reports generated (terminal + htmlcov/)")
        
    except FileNotFoundError:
        print("  ⚠️  coverage not available")
    except Exception as e:
        print(f"  ❌ Error generating coverage report: {e}")


def cleanup_test_artifacts():
    """Clean up test artifacts"""
    artifacts = [
        '.coverage',
        'test-results.xml',
        'security-report.json',
        'htmlcov',
        '__pycache__',
        '.pytest_cache'
    ]
    
    project_root = Path(__file__).parent.parent
    
    for artifact in artifacts:
        artifact_path = project_root / artifact
        if artifact_path.exists():
            if artifact_path.is_file():
                artifact_path.unlink()
            else:
                import shutil
                shutil.rmtree(artifact_path, ignore_errors=True)


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description='CAPSTONE-LAZARUS Test Runner')
    parser.add_argument('--unit', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration', action='store_true', help='Run only integration tests')
    parser.add_argument('--coverage', action='store_true', default=True, help='Run with coverage')
    parser.add_argument('--parallel', action='store_true', help='Run tests in parallel')
    parser.add_argument('--lint', action='store_true', help='Run linting checks')
    parser.add_argument('--security', action='store_true', help='Run security checks')
    parser.add_argument('--type-check', action='store_true', help='Run type checking')
    parser.add_argument('--cleanup', action='store_true', help='Clean up test artifacts')
    parser.add_argument('--verbose', '-v', action='count', default=1, help='Increase verbosity')
    parser.add_argument('--use-pytest', action='store_true', help='Use pytest instead of unittest')
    
    args = parser.parse_args()
    
    if args.cleanup:
        cleanup_test_artifacts()
        print("🧹 Test artifacts cleaned up")
        return
    
    print("🚀 Starting CAPSTONE-LAZARUS Test Suite")
    print("=" * 50)
    
    # Setup
    setup_test_environment()
    
    if args.use_pytest:
        install_test_dependencies()
    
    project_root = Path(__file__).parent.parent
    tests_dir = project_root / 'tests'
    
    # Determine which tests to run
    test_suites = []
    
    if args.unit or (not args.unit and not args.integration):
        test_suites.append(('Unit Tests', tests_dir / 'unit'))
    
    if args.integration or (not args.unit and not args.integration):
        test_suites.append(('Integration Tests', tests_dir / 'integration'))
    
    # Run tests
    all_passed = True
    start_time = time.time()
    
    for suite_name, suite_dir in test_suites:
        if not suite_dir.exists():
            print(f"⚠️  {suite_name} directory not found: {suite_dir}")
            continue
        
        print(f"\n{'='*20} {suite_name} {'='*20}")
        
        if args.use_pytest:
            passed = run_pytest_suite(suite_dir, coverage=args.coverage, parallel=args.parallel)
        else:
            passed = run_unittest_suite(suite_dir, verbosity=args.verbose)
        
        all_passed = all_passed and passed
    
    # Additional checks
    if args.lint:
        print(f"\n{'='*20} Linting Checks {'='*20}")
        lint_passed = run_linting_checks()
        all_passed = all_passed and lint_passed
    
    if args.security:
        print(f"\n{'='*20} Security Checks {'='*20}")
        security_passed = run_security_checks()
        all_passed = all_passed and security_passed
    
    if args.type_check:
        print(f"\n{'='*20} Type Checks {'='*20}")
        type_passed = run_type_checks()
        all_passed = all_passed and type_passed
    
    # Generate coverage report
    if args.coverage and not args.use_pytest:
        generate_coverage_report()
    
    # Final results
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{'='*50}")
    print(f"🏁 Test Suite Complete (Duration: {duration:.2f}s)")
    
    if all_passed:
        print("✅ All tests passed successfully!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == '__main__':
    exit(main())