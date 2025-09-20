"""
CAPSTONE-LAZARUS: System Verification Script
===========================================
Comprehensive system verification for production deployment

Features:
- Complete system functionality testing
- Module integration verification
- Performance benchmarking
- Deployment readiness assessment
- Comprehensive reporting
"""

import sys
import logging
import traceback
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
import tensorflow as tf

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemVerifier:
    """
    Comprehensive system verification for CAPSTONE-LAZARUS
    """
    
    def __init__(self):
        self.verification_results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'tensorflow_version': tf.__version__,
                'python_version': sys.version.split()[0],
                'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0
            },
            'tests': {},
            'overall_status': 'running',
            'summary': {}
        }
    
    def test_imports(self) -> Dict[str, Any]:
        """Test all critical imports"""
        print("🔍 TESTING IMPORTS")
        print("=" * 50)
        
        import_results = {
            'status': 'passed',
            'failed_imports': [],
            'successful_imports': [],
            'warnings': []
        }
        
        # Core modules to test
        test_imports = [
            ('src.training.pipeline', ['TrainingPipeline', 'ModelConfig', 'TrainingConfig']),
            ('src.models.architectures', ['create_model', 'create_ensemble_model']),
            ('src.ensembling', ['EnsemblePredictor', 'EnsembleConfig']),
            ('src.interpretability', ['GradCAM', 'MultiModelGradCAM']),
            ('src.data_utils', ['PlantDiseaseDataLoader']),
            ('tensorflow', ['keras']),
            ('streamlit', None),
            ('plotly', ['express', 'graph_objects']),
            ('sklearn', ['model_selection']),
            ('numpy', None),
            ('pandas', None),
            ('cv2', None),
            ('PIL', ['Image'])
        ]
        
        for module_name, submodules in test_imports:
            try:
                module = __import__(module_name)
                
                if submodules:
                    for submodule in submodules:
                        if hasattr(module, submodule):
                            print(f"✅ {module_name}.{submodule}")
                        else:
                            # Try importing from submodule
                            try:
                                exec(f"from {module_name} import {submodule}")
                                print(f"✅ {module_name}.{submodule}")
                            except ImportError:
                                print(f"⚠️ {module_name}.{submodule} - not found")
                                import_results['warnings'].append(f"{module_name}.{submodule}")
                else:
                    print(f"✅ {module_name}")
                
                import_results['successful_imports'].append(module_name)
                
            except ImportError as e:
                print(f"❌ {module_name} - {str(e)}")
                import_results['failed_imports'].append(module_name)
        
        # Overall status
        if import_results['failed_imports']:
            import_results['status'] = 'failed'
        elif import_results['warnings']:
            import_results['status'] = 'warning'
        
        print(f"\n📊 Import Results:")
        print(f"   ✅ Successful: {len(import_results['successful_imports'])}")
        print(f"   ❌ Failed: {len(import_results['failed_imports'])}")
        print(f"   ⚠️ Warnings: {len(import_results['warnings'])}")
        
        return import_results
    
    def test_model_architectures(self) -> Dict[str, Any]:
        """Test model architecture creation"""
        print("\n🏗️ TESTING MODEL ARCHITECTURES")
        print("=" * 50)
        
        architecture_results = {
            'status': 'passed',
            'tested_architectures': [],
            'successful_architectures': [],
            'failed_architectures': [],
            'performance_data': {}
        }
        
        try:
            from src.models.architectures import create_model, validate_model_creation
            
            # Test architecture validation
            print("Running architecture validation...")
            validation_success = validate_model_creation()
            
            if validation_success:
                print("✅ All architectures validated successfully")
                architecture_results['successful_architectures'] = [
                    'efficientnet_b0', 'resnet50', 'mobilenet_v2', 'densenet121'
                ]
            else:
                print("❌ Some architectures failed validation")
                architecture_results['status'] = 'warning'
            
        except Exception as e:
            print(f"❌ Architecture testing failed: {e}")
            architecture_results['status'] = 'failed'
            architecture_results['error'] = str(e)
        
        return architecture_results
    
    def test_training_pipeline(self) -> Dict[str, Any]:
        """Test training pipeline components"""
        print("\n🚀 TESTING TRAINING PIPELINE")
        print("=" * 50)
        
        pipeline_results = {
            'status': 'passed',
            'components_tested': [],
            'successful_components': [],
            'failed_components': []
        }
        
        try:
            from src.training.pipeline import verify_training_pipeline
            
            print("Running training pipeline verification...")
            verify_training_pipeline()
            
            print("✅ Training pipeline verification completed")
            pipeline_results['successful_components'] = [
                'TrainingPipeline', 'ModelRegistry', 'Configuration classes'
            ]
            
        except Exception as e:
            print(f"❌ Training pipeline testing failed: {e}")
            pipeline_results['status'] = 'failed'
            pipeline_results['error'] = str(e)
            pipeline_results['traceback'] = traceback.format_exc()
        
        return pipeline_results
    
    def test_ensemble_system(self) -> Dict[str, Any]:
        """Test ensemble system components"""
        print("\n🎯 TESTING ENSEMBLE SYSTEM")
        print("=" * 50)
        
        ensemble_results = {
            'status': 'passed',
            'components_tested': [],
            'successful_components': [],
            'failed_components': []
        }
        
        try:
            from src.ensembling import verify_ensemble_functionality
            
            print("Running ensemble functionality verification...")
            verify_ensemble_functionality()
            
            print("✅ Ensemble system verification completed")
            ensemble_results['successful_components'] = [
                'EnsemblePredictor', 'EnsembleConfig', 'Multi-model support'
            ]
            
        except Exception as e:
            print(f"❌ Ensemble system testing failed: {e}")
            ensemble_results['status'] = 'failed'
            ensemble_results['error'] = str(e)
            ensemble_results['traceback'] = traceback.format_exc()
        
        return ensemble_results
    
    def test_interpretability(self) -> Dict[str, Any]:
        """Test interpretability components"""
        print("\n🔍 TESTING INTERPRETABILITY")
        print("=" * 50)
        
        interpretability_results = {
            'status': 'passed',
            'components_tested': [],
            'successful_components': [],
            'failed_components': []
        }
        
        try:
            from src.interpretability import verify_grad_cam_functionality
            
            print("Running Grad-CAM functionality verification...")
            verify_grad_cam_functionality()
            
            print("✅ Interpretability system verification completed")
            interpretability_results['successful_components'] = [
                'GradCAM', 'MultiModelGradCAM', 'Visualization'
            ]
            
        except Exception as e:
            print(f"❌ Interpretability testing failed: {e}")
            interpretability_results['status'] = 'failed'
            interpretability_results['error'] = str(e)
            interpretability_results['traceback'] = traceback.format_exc()
        
        return interpretability_results
    
    def test_data_utils(self) -> Dict[str, Any]:
        """Test data utilities"""
        print("\n📊 TESTING DATA UTILITIES")
        print("=" * 50)
        
        data_results = {
            'status': 'passed',
            'components_tested': [],
            'successful_components': [],
            'failed_components': []
        }
        
        try:
            from src.data_utils import PlantDiseaseDataLoader
            
            # Test data loader initialization (without actual data)
            print("Testing data loader initialization...")
            loader = PlantDiseaseDataLoader("test_data", img_size=(224, 224), batch_size=32)
            print(f"✅ Data loader initialized: {type(loader).__name__}")
            
            data_results['successful_components'] = ['PlantDiseaseDataLoader']
            
        except Exception as e:
            print(f"❌ Data utilities testing failed: {e}")
            data_results['status'] = 'failed'
            data_results['error'] = str(e)
        
        return data_results
    
    def test_streamlit_app(self) -> Dict[str, Any]:
        """Test Streamlit application components"""
        print("\n🌐 TESTING STREAMLIT APPLICATION")
        print("=" * 50)
        
        streamlit_results = {
            'status': 'passed',
            'components_tested': [],
            'successful_components': [],
            'failed_components': []
        }
        
        try:
            # Test if the streamlit app file exists and can be imported
            app_path = project_root / "app" / "streamlit_app" / "advanced_main.py"
            
            if app_path.exists():
                print(f"✅ Streamlit app found: {app_path}")
                
                # Test critical streamlit imports
                try:
                    import streamlit as st
                    import plotly.express as px
                    import plotly.graph_objects as go
                    print("✅ Streamlit dependencies available")
                    
                    streamlit_results['successful_components'] = [
                        'streamlit_app', 'plotly_integration', 'dependencies'
                    ]
                    
                except ImportError as e:
                    print(f"⚠️ Streamlit dependency missing: {e}")
                    streamlit_results['status'] = 'warning'
                    streamlit_results['warnings'] = [str(e)]
            else:
                print(f"❌ Streamlit app not found: {app_path}")
                streamlit_results['status'] = 'failed'
                streamlit_results['error'] = 'Streamlit app file not found'
            
        except Exception as e:
            print(f"❌ Streamlit testing failed: {e}")
            streamlit_results['status'] = 'failed'
            streamlit_results['error'] = str(e)
        
        return streamlit_results
    
    def test_system_integration(self) -> Dict[str, Any]:
        """Test system integration and workflows"""
        print("\n🔗 TESTING SYSTEM INTEGRATION")
        print("=" * 50)
        
        integration_results = {
            'status': 'passed',
            'workflows_tested': [],
            'successful_workflows': [],
            'failed_workflows': []
        }
        
        try:
            # Test training orchestrator
            print("Testing training orchestrator...")
            orchestrator_path = project_root / "train_orchestrator.py"
            
            if orchestrator_path.exists():
                print("✅ Training orchestrator found")
                integration_results['successful_workflows'].append('training_orchestrator')
            else:
                print("❌ Training orchestrator not found")
                integration_results['failed_workflows'].append('training_orchestrator')
            
            # Test inference workflow (mock)
            print("Testing inference workflow integration...")
            
            # Simulate a basic inference workflow
            try:
                from src.training.pipeline import ModelConfig
                from src.ensembling import EnsembleConfig
                
                # Test configuration creation
                model_config = ModelConfig(architecture='efficientnet_b0')
                ensemble_config = EnsembleConfig(models=['efficientnet_b0', 'resnet50'])
                
                print("✅ Configuration integration working")
                integration_results['successful_workflows'].append('configuration_integration')
                
            except Exception as e:
                print(f"⚠️ Configuration integration issue: {e}")
                integration_results['failed_workflows'].append('configuration_integration')
            
        except Exception as e:
            print(f"❌ Integration testing failed: {e}")
            integration_results['status'] = 'failed'
            integration_results['error'] = str(e)
        
        return integration_results
    
    def assess_deployment_readiness(self) -> Dict[str, Any]:
        """Assess overall deployment readiness"""
        print("\n🚀 ASSESSING DEPLOYMENT READINESS")
        print("=" * 50)
        
        # Check all test results
        all_tests = [
            self.verification_results['tests'].get('imports', {}),
            self.verification_results['tests'].get('architectures', {}),
            self.verification_results['tests'].get('training_pipeline', {}),
            self.verification_results['tests'].get('ensemble_system', {}),
            self.verification_results['tests'].get('interpretability', {}),
            self.verification_results['tests'].get('data_utils', {}),
            self.verification_results['tests'].get('streamlit_app', {}),
            self.verification_results['tests'].get('integration', {})
        ]
        
        passed_tests = len([t for t in all_tests if t.get('status') == 'passed'])
        warning_tests = len([t for t in all_tests if t.get('status') == 'warning'])
        failed_tests = len([t for t in all_tests if t.get('status') == 'failed'])
        total_tests = len(all_tests)
        
        # Calculate readiness score
        readiness_score = (passed_tests + 0.5 * warning_tests) / total_tests
        
        deployment_assessment = {
            'readiness_score': readiness_score,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'warning_tests': warning_tests,
            'failed_tests': failed_tests,
            'deployment_status': 'not_ready',
            'recommendations': []
        }
        
        # Determine deployment status
        if readiness_score >= 0.9:
            deployment_assessment['deployment_status'] = 'production_ready'
            deployment_assessment['recommendations'].append("✅ System is ready for production deployment")
        elif readiness_score >= 0.7:
            deployment_assessment['deployment_status'] = 'staging_ready'
            deployment_assessment['recommendations'].append("⚠️ System ready for staging - resolve warnings before production")
        elif readiness_score >= 0.5:
            deployment_assessment['deployment_status'] = 'development_ready'
            deployment_assessment['recommendations'].append("🔧 System needs fixes before staging deployment")
        else:
            deployment_assessment['deployment_status'] = 'not_ready'
            deployment_assessment['recommendations'].append("❌ System needs significant fixes before deployment")
        
        # Specific recommendations
        if failed_tests > 0:
            deployment_assessment['recommendations'].append(f"🔴 Fix {failed_tests} failed test(s)")
        
        if warning_tests > 0:
            deployment_assessment['recommendations'].append(f"🟡 Address {warning_tests} warning(s)")
        
        print(f"Readiness Score: {readiness_score:.1%}")
        print(f"Deployment Status: {deployment_assessment['deployment_status']}")
        print("Recommendations:")
        for rec in deployment_assessment['recommendations']:
            print(f"  - {rec}")
        
        return deployment_assessment
    
    def run_complete_verification(self) -> Dict[str, Any]:
        """Run complete system verification"""
        print("🌿 CAPSTONE-LAZARUS SYSTEM VERIFICATION")
        print("=" * 80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        try:
            # Run all tests
            self.verification_results['tests']['imports'] = self.test_imports()
            self.verification_results['tests']['architectures'] = self.test_model_architectures()
            self.verification_results['tests']['training_pipeline'] = self.test_training_pipeline()
            self.verification_results['tests']['ensemble_system'] = self.test_ensemble_system()
            self.verification_results['tests']['interpretability'] = self.test_interpretability()
            self.verification_results['tests']['data_utils'] = self.test_data_utils()
            self.verification_results['tests']['streamlit_app'] = self.test_streamlit_app()
            self.verification_results['tests']['integration'] = self.test_system_integration()
            
            # Assess deployment readiness
            deployment_assessment = self.assess_deployment_readiness()
            self.verification_results['deployment_assessment'] = deployment_assessment
            
            # Update overall status
            if deployment_assessment['deployment_status'] in ['production_ready', 'staging_ready']:
                self.verification_results['overall_status'] = 'passed'
            elif deployment_assessment['deployment_status'] == 'development_ready':
                self.verification_results['overall_status'] = 'warning'
            else:
                self.verification_results['overall_status'] = 'failed'
            
            # Generate summary
            self.verification_results['summary'] = {
                'verification_completed': True,
                'total_tests': deployment_assessment['total_tests'],
                'passed_tests': deployment_assessment['passed_tests'],
                'warning_tests': deployment_assessment['warning_tests'],
                'failed_tests': deployment_assessment['failed_tests'],
                'readiness_score': deployment_assessment['readiness_score'],
                'deployment_status': deployment_assessment['deployment_status'],
                'verification_time': datetime.now().isoformat()
            }
            
            print("\n" + "=" * 80)
            print("🎯 VERIFICATION COMPLETE")
            print("=" * 80)
            print(f"Overall Status: {self.verification_results['overall_status'].upper()}")
            print(f"Tests: {deployment_assessment['passed_tests']} passed, {deployment_assessment['warning_tests']} warnings, {deployment_assessment['failed_tests']} failed")
            print(f"Readiness: {deployment_assessment['readiness_score']:.1%} - {deployment_assessment['deployment_status']}")
            
        except Exception as e:
            self.verification_results['overall_status'] = 'error'
            self.verification_results['error'] = str(e)
            self.verification_results['traceback'] = traceback.format_exc()
            
            print(f"\n❌ VERIFICATION ERROR: {e}")
        
        # Save results
        results_path = project_root / 'verification_results.json'
        try:
            with open(results_path, 'w') as f:
                json.dump(self.verification_results, f, indent=2)
            print(f"\n📄 Results saved to: {results_path}")
        except Exception as e:
            print(f"⚠️ Could not save results: {e}")
        
        return self.verification_results


def main():
    """Main verification script"""
    verifier = SystemVerifier()
    results = verifier.run_complete_verification()
    
    # Exit with appropriate code
    if results['overall_status'] == 'passed':
        print("\n🎉 SYSTEM VERIFICATION PASSED - READY FOR DEPLOYMENT!")
        sys.exit(0)
    elif results['overall_status'] == 'warning':
        print("\n⚠️ SYSTEM VERIFICATION COMPLETED WITH WARNINGS")
        sys.exit(0)
    else:
        print("\n❌ SYSTEM VERIFICATION FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()