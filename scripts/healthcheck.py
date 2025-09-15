#!/usr/bin/env python3
"""
Health check script for CAPSTONE-LAZARUS Docker containers
"""

import sys
import requests
import os
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_streamlit():
    """Check Streamlit health"""
    try:
        response = requests.get(
            "http://localhost:8501/_stcore/health",
            timeout=5
        )
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Streamlit health check failed: {e}")
        return False


def check_fastapi():
    """Check FastAPI health"""
    try:
        response = requests.get(
            "http://localhost:8000/health",
            timeout=5
        )
        return response.status_code == 200
    except Exception as e:
        logger.error(f"FastAPI health check failed: {e}")
        return False


def check_file_system():
    """Check required directories and permissions"""
    try:
        required_dirs = [
            "/app/data",
            "/app/models", 
            "/app/logs",
            "/app/experiments"
        ]
        
        for dir_path in required_dirs:
            path = Path(dir_path)
            if not path.exists():
                logger.error(f"Required directory missing: {dir_path}")
                return False
            
            if not path.is_dir():
                logger.error(f"Path is not a directory: {dir_path}")
                return False
            
            # Check if we can write to the directory
            try:
                test_file = path / ".health_check"
                test_file.write_text("health_check")
                test_file.unlink()
            except Exception as e:
                logger.error(f"Cannot write to directory {dir_path}: {e}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"File system check failed: {e}")
        return False


def check_python_imports():
    """Check critical Python imports"""
    try:
        import tensorflow as tf
        import streamlit as st
        import numpy as np
        import pandas as pd
        
        logger.info(f"TensorFlow version: {tf.__version__}")
        logger.info(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
        
        return True
        
    except ImportError as e:
        logger.error(f"Critical import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Python import check failed: {e}")
        return False


def main():
    """Main health check function"""
    
    logger.info("Starting health check...")
    
    # Determine service type from environment or process
    service_type = os.environ.get('SERVICE_TYPE', 'unknown')
    
    # If no service type, try to detect from running processes
    if service_type == 'unknown':
        try:
            import psutil
            processes = [p.name() for p in psutil.process_iter()]
            
            if any('streamlit' in p for p in processes):
                service_type = 'streamlit'
            elif any('uvicorn' in p for p in processes):
                service_type = 'fastapi'
            elif any('gunicorn' in p for p in processes):
                service_type = 'fastapi'
            elif any('jupyter' in p for p in processes):
                service_type = 'jupyter'
            
        except ImportError:
            pass
    
    # Run basic checks
    checks = []
    
    # File system check
    logger.info("Checking file system...")
    checks.append(("File System", check_file_system()))
    
    # Python imports check
    logger.info("Checking Python imports...")
    checks.append(("Python Imports", check_python_imports()))
    
    # Service-specific checks
    if service_type == 'streamlit':
        logger.info("Checking Streamlit service...")
        # Give Streamlit time to start up
        time.sleep(2)
        checks.append(("Streamlit", check_streamlit()))
        
    elif service_type == 'fastapi':
        logger.info("Checking FastAPI service...")
        # Give FastAPI time to start up
        time.sleep(1)
        checks.append(("FastAPI", check_fastapi()))
    
    # Report results
    logger.info("Health check results:")
    all_passed = True
    
    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {check_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        logger.info("🎉 All health checks passed!")
        sys.exit(0)
    else:
        logger.error("💥 Some health checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()