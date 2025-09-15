#!/usr/bin/env python3
"""
Configuration Validation Script for CAPSTONE-LAZARUS
===================================================

This script validates configuration files to ensure they are properly
formatted and contain required fields.
"""

import sys
import os
from pathlib import Path
import json
import yaml
from typing import Dict, Any, List, Optional

# Required configuration sections and fields
REQUIRED_CONFIG_SECTIONS = {
    "model": ["name", "input_shape"],
    "training": ["batch_size", "epochs", "learning_rate"],
    "data": ["train_path", "val_path"]
}

OPTIONAL_CONFIG_SECTIONS = {
    "model": ["num_classes", "dropout_rate", "activation"],
    "training": ["optimizer", "loss", "metrics", "callbacks"],
    "data": ["test_path", "augmentation", "preprocessing"],
    "deployment": ["model_name", "version", "endpoint"],
    "monitoring": ["metrics", "alerts", "logging"]
}

def load_config_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load configuration file (JSON or YAML)"""
    try:
        with open(file_path, 'r') as f:
            if file_path.suffix.lower() == '.json':
                return json.load(f)
            elif file_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            else:
                print(f"⚠️  Warning: Unknown config format: {file_path}")
                return None
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON in {file_path}: {e}")
        return None
    except yaml.YAMLError as e:
        print(f"❌ ERROR: Invalid YAML in {file_path}: {e}")
        return None
    except Exception as e:
        print(f"❌ ERROR: Cannot load {file_path}: {e}")
        return None

def validate_config_structure(config: Dict[str, Any], file_path: Path) -> List[str]:
    """Validate configuration structure"""
    errors = []
    warnings = []
    
    # Check required sections
    for section, required_fields in REQUIRED_CONFIG_SECTIONS.items():
        if section not in config:
            errors.append(f"Missing required section '{section}' in {file_path}")
            continue
            
        section_config = config[section]
        if not isinstance(section_config, dict):
            errors.append(f"Section '{section}' must be a dictionary in {file_path}")
            continue
        
        # Check required fields in section
        for field in required_fields:
            if field not in section_config:
                errors.append(f"Missing required field '{section}.{field}' in {file_path}")
    
    # Check for deprecated or unknown sections
    known_sections = set(REQUIRED_CONFIG_SECTIONS.keys()) | set(OPTIONAL_CONFIG_SECTIONS.keys())
    for section in config.keys():
        if section not in known_sections and not section.startswith('_'):  # Allow private sections
            warnings.append(f"Unknown section '{section}' in {file_path}")
    
    return errors

def validate_config_values(config: Dict[str, Any], file_path: Path) -> List[str]:
    """Validate configuration values"""
    errors = []
    
    # Model validation
    if "model" in config:
        model_config = config["model"]
        
        # Check input_shape format
        if "input_shape" in model_config:
            input_shape = model_config["input_shape"]
            if not isinstance(input_shape, (list, tuple)) or len(input_shape) < 2:
                errors.append(f"Invalid input_shape format in {file_path}")
        
        # Check num_classes
        if "num_classes" in model_config:
            num_classes = model_config["num_classes"]
            if not isinstance(num_classes, int) or num_classes < 1:
                errors.append(f"Invalid num_classes value in {file_path}")
    
    # Training validation
    if "training" in config:
        training_config = config["training"]
        
        # Check batch_size
        if "batch_size" in training_config:
            batch_size = training_config["batch_size"]
            if not isinstance(batch_size, int) or batch_size < 1:
                errors.append(f"Invalid batch_size value in {file_path}")
        
        # Check epochs
        if "epochs" in training_config:
            epochs = training_config["epochs"]
            if not isinstance(epochs, int) or epochs < 1:
                errors.append(f"Invalid epochs value in {file_path}")
        
        # Check learning_rate
        if "learning_rate" in training_config:
            lr = training_config["learning_rate"]
            if not isinstance(lr, (int, float)) or lr <= 0:
                errors.append(f"Invalid learning_rate value in {file_path}")
    
    # Data validation
    if "data" in config:
        data_config = config["data"]
        
        # Check paths exist if they're not placeholders
        for path_key in ["train_path", "val_path", "test_path"]:
            if path_key in data_config:
                path_value = data_config[path_key]
                if isinstance(path_value, str) and not path_value.startswith(("$", "{{", "data/")):
                    # Check if absolute path exists
                    if os.path.isabs(path_value) and not Path(path_value).exists():
                        errors.append(f"Path does not exist: {path_key} = {path_value} in {file_path}")
    
    return errors

def validate_environment_config(config: Dict[str, Any], file_path: Path) -> List[str]:
    """Validate environment-specific configurations"""
    errors = []
    
    # Check for environment placeholders
    config_str = json.dumps(config)
    
    # Common environment variable patterns
    env_patterns = ["${", "{{", "$ENV{"]
    has_env_vars = any(pattern in config_str for pattern in env_patterns)
    
    if has_env_vars:
        # This is good - using environment variables
        pass
    else:
        # Check for hardcoded sensitive values
        sensitive_keys = ["password", "secret", "key", "token", "api_key"]
        
        def check_sensitive_recursive(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if any(sensitive in key.lower() for sensitive in sensitive_keys):
                        if isinstance(value, str) and len(value) > 10 and not value.startswith(("$", "{{")):
                            errors.append(f"Potential hardcoded sensitive value at {current_path} in {file_path}")
                    check_sensitive_recursive(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_sensitive_recursive(item, f"{path}[{i}]")
        
        check_sensitive_recursive(config)
    
    return errors

def validate_docker_config(file_path: Path) -> List[str]:
    """Validate Docker configuration files"""
    errors = []
    
    if file_path.name == "docker-compose.yml":
        try:
            with open(file_path, 'r') as f:
                compose_config = yaml.safe_load(f)
            
            # Check required structure
            if "services" not in compose_config:
                errors.append(f"Missing 'services' section in {file_path}")
                return errors
            
            # Validate each service
            for service_name, service_config in compose_config["services"].items():
                if not isinstance(service_config, dict):
                    errors.append(f"Service '{service_name}' configuration must be a dictionary")
                    continue
                
                # Check for common issues
                if "image" not in service_config and "build" not in service_config:
                    errors.append(f"Service '{service_name}' missing 'image' or 'build' configuration")
                
                # Check port configurations
                if "ports" in service_config:
                    ports = service_config["ports"]
                    if isinstance(ports, list):
                        for port in ports:
                            if isinstance(port, str) and ":" not in port:
                                errors.append(f"Invalid port format '{port}' in service '{service_name}'")
        
        except Exception as e:
            errors.append(f"Error parsing Docker Compose file {file_path}: {e}")
    
    elif file_path.name == "Dockerfile":
        try:
            with open(file_path, 'r') as f:
                dockerfile_content = f.read()
            
            lines = dockerfile_content.strip().split('\n')
            if not lines or not lines[0].strip().upper().startswith('FROM'):
                errors.append(f"Dockerfile must start with FROM instruction in {file_path}")
            
            # Check for common issues
            if "COPY . ." in dockerfile_content:
                errors.append(f"Avoid 'COPY . .' in Dockerfile - use specific paths in {file_path}")
            
            if "RUN apt-get update" in dockerfile_content and "apt-get clean" not in dockerfile_content:
                errors.append(f"Missing cleanup after apt-get update in {file_path}")
        
        except Exception as e:
            errors.append(f"Error reading Dockerfile {file_path}: {e}")
    
    return errors

def main():
    """Main configuration validation function"""
    print("⚙️  Running configuration validation...")
    
    # Find configuration files
    config_files = []
    
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'diff', '--cached', '--name-only'],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            staged_files = result.stdout.strip().split('\n')
            config_files = [
                Path(f) for f in staged_files 
                if Path(f).exists() and is_config_file(Path(f))
            ]
        else:
            # Fallback: find all config files
            config_files = find_all_config_files()
    
    except Exception:
        # Fallback: find all config files
        config_files = find_all_config_files()
    
    if not config_files:
        print("✅ No configuration files to validate")
        return 0
    
    print(f"🔍 Validating {len(config_files)} configuration files...")
    
    all_errors = []
    all_warnings = []
    
    for file_path in config_files:
        print(f"  📄 {file_path}")
        
        # Docker files need special handling
        if file_path.name in ["Dockerfile", "docker-compose.yml", "docker-compose.yaml"]:
            errors = validate_docker_config(file_path)
            all_errors.extend(errors)
            continue
        
        # Load and validate regular config files
        config = load_config_file(file_path)
        if config is None:
            continue  # Error already reported
        
        # Validate structure
        errors = validate_config_structure(config, file_path)
        all_errors.extend(errors)
        
        # Validate values
        errors = validate_config_values(config, file_path)
        all_errors.extend(errors)
        
        # Validate environment configuration
        errors = validate_environment_config(config, file_path)
        all_errors.extend(errors)
    
    # Report results
    if all_warnings:
        print("\n⚠️  WARNINGS:")
        for warning in all_warnings:
            print(f"  {warning}")
    
    if all_errors:
        print("\n❌ CONFIGURATION ERRORS:")
        for error in all_errors:
            print(f"  {error}")
        
        print("\n💡 SUGGESTIONS:")
        print("  • Check configuration file syntax")
        print("  • Ensure all required fields are present")
        print("  • Use environment variables for sensitive data")
        print("  • Validate paths and references")
        print("  • Follow configuration schema documentation")
        
        return 1
    
    print("✅ All configuration files are valid!")
    return 0

def is_config_file(file_path: Path) -> bool:
    """Check if file is a configuration file"""
    config_patterns = [
        'config', 'settings', 'docker-compose', 'Dockerfile',
        '.env', '.yaml', '.yml', '.json', '.toml', '.ini'
    ]
    
    file_name = file_path.name.lower()
    return any(pattern in file_name for pattern in config_patterns)

def find_all_config_files() -> List[Path]:
    """Find all configuration files in the project"""
    config_files = []
    
    # Common config locations
    config_dirs = ['.', 'config', 'configs', 'src/config', 'app', 'infra']
    
    for config_dir in config_dirs:
        dir_path = Path(config_dir)
        if dir_path.exists():
            # Find config files
            for pattern in ['*.json', '*.yaml', '*.yml', '*.toml', '*.ini']:
                config_files.extend(dir_path.glob(pattern))
            
            # Find Docker files
            for docker_file in ['Dockerfile', 'docker-compose.yml', 'docker-compose.yaml']:
                docker_path = dir_path / docker_file
                if docker_path.exists():
                    config_files.append(docker_path)
    
    return config_files

if __name__ == "__main__":
    sys.exit(main())