"""
Base Test Generator Module - Contains base class for test generators
"""
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
#from llm_provider import LLMProvider

class BaseTestGenerator:
    """Base class for all test generators"""

    def __init__(self, name: str):
        self.name = name
        self.id = f"{name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
        
        self.generated_tests = []

    def generate_tests(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tests based on input data - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement generate_tests method")

    def format_report(self) -> Dict[str, Any]:
        """Format test generation report with metadata"""
        return {
            "generator": self.name,
            "id": self.id,
            "timestamp": datetime.now().isoformat(),
            "test_count": len(self.generated_tests),
            "tests": self.generated_tests
        }

class TestGenerationManager:
    """Manages multiple test generators and combines their results"""

    def __init__(self, name: str = "Test Generation Manager"):
        self.name = name
        self.id = f"{name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
        self.generators = []
        self.results = {}

    def add_generator(self, generator: BaseTestGenerator) -> None:
        """Add a test generator to the manager"""
        self.generators.append(generator)

    def generate_all_tests(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run all generators and combine results"""
        all_results = {
            "manager": self.name,
            "id": self.id,
            "timestamp": datetime.now().isoformat(),
            "generator_count": len(self.generators),
            "generator_results": [],
            "total_test_count": 0,
            "all_tests": []
        }

        # Generate tests from each generator
        for generator in self.generators:
            try:
                generator.generated_tests = generator.generate_tests(input_data)
                generator_report = generator.format_report()
                all_results["generator_results"].append(generator_report)
                all_results["all_tests"].extend(generator.generated_tests)
                all_results["total_test_count"] += len(generator.generated_tests)
            except Exception as e:
                print(f"Error in generator {generator.name}: {e}")
                all_results["generator_results"].append({
                    "generator": generator.name,
                    "id": generator.id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "test_count": 0,
                    "tests": []
                })

        self.results = all_results
        return all_results

    def save_results(self, filename: str) -> None:
        """Save results to a JSON file"""
        import json
        import os

        # Create directory if it doesn't exist
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)