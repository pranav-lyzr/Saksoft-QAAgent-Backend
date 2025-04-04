#performance_test_generator.py
"""
Purpose: Creates tests to measure performance characteristics like response time, throughput, and scalability under various loads.
Key Method: generate_tests(input_data) takes performance requirements to generate load, stress, and endurance tests.
"""
"""
Performance Test Generator Module - Generates tests for performance characteristics
"""
import json
import re
import requests
from typing import Dict, List, Any
#from base_test_generator import BaseTestGenerator
#from llm_provider import LLMProvider

class PerformanceTestGenerator():
    """Generates tests for performance characteristics of the application"""

    def __init__(self):
        self.api_url = "https://agent-prod.studio.lyzr.ai/v3/inference/chat/"
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": "sk-default-yStV4gbpjadbQSw4i7QhoOLRwAs5dEcl"
        }
        self.user_id = "pranav@lyzr.ai"
      
    def _call_lyzr_api(self, agent_id: str, session_id: str, system_prompt: str, message: str) -> str:
        """Helper function to call Lyzr API"""
        
        messages = json.dumps([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ])


        payload = {
            "user_id": self.user_id,
            "agent_id": agent_id,
            "session_id": session_id,
            "message": messages
        }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            response(f"response: {response}")
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {str(e)}")
            return json.dumps({"error": str(e)})
        
    def parse_json_response(self, response: str, default_key: str = "test_cases") -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        # Try to extract JSON from code blocks
        json_match = re.search(r'```(?:json)?\n(.*?)\n```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from code block: {e}")
                print("Response content in code block:", json_match.group(1)[:100] + "...")

        # Try to parse the entire response as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from full response: {e}")
            print("First 100 chars of response:", response[:100] + "...")

            # If all parsing fails, return a default structure
            print(f"Returning default structure with key: {default_key}")
            return {default_key: []}
        

    def generate_tests(self, input_data: Dict[str, Any], agent_id:str) -> List[Dict[str, Any]]:
        """Generate performance tests based on input data"""
        app_type = input_data.get("app_type", "unknown")
        architecture = input_data.get("architecture", {})
        requirements = input_data.get("requirements", {})
        performance_reqs = requirements.get("performance", {})

        # Use LLM to generate performance tests if we have performance requirements
        if performance_reqs:
            prompt = self._create_performance_prompt(app_type, architecture, performance_reqs)
            #system_prompt = """You are a performance testing expert specializing in creating
            #comprehensive test cases to measure and validate system performance characteristics."""

            system_prompt = """You are a performance testing expert specializing in creating comprehensive test cases to measure and validate system performance characteristics. Your expertise includes:

            1. Designing load tests that accurately simulate real-world user behavior and traffic patterns
            2. Creating stress tests to identify system breaking points and failure modes
            3. Developing endurance tests to detect memory leaks and resource exhaustion
            4. Building spike tests to evaluate system recovery capabilities under sudden load changes
            5. Crafting targeted tests for database performance, network latency, and application response time
            6. Identifying critical performance metrics and appropriate thresholds for different system components
            7. Developing test scenarios that validate performance SLAs and NFRs (Non-Functional Requirements)
            8. Creating performance baselines and regression test suites for continuous monitoring
            9. Designing distributed load generation architectures for high-volume testing
            10. Developing custom performance monitoring instrumentation and telemetry
            11. Analyzing bottlenecks through profiling and resource utilization patterns
            12. Building test scenarios for horizontal and vertical scaling validation
            13. Creating reproducible test environments with controlled variables
            14. Developing specialized tests for caching effectiveness, connection pooling, and resource optimization

            You provide detailed analysis of test results, including statistical significance and confidence intervals. You translate technical performance data into business impact metrics, balancing theoretical performance models with practical, measurable outcomes. Your recommendations consider the cost-benefit relationship between performance improvements and implementation effort.
            """

            response = self._call_lyzr_api(
                agent_id=agent_id,
                session_id=agent_id,
                system_prompt=system_prompt,
                message=prompt
            )
            parsed_response = self.parse_json_response(response)

            if "test_cases" in parsed_response and parsed_response["test_cases"]:
                # Ensure all test cases have required fields and proper category
                test_cases = parsed_response["test_cases"]
                for test in test_cases:
                    if "id" not in test:
                        test["id"] = f"PERF-{len(test_cases):03d}"
                    test["category"] = "Performance"
                    if "priority" not in test:
                        test["priority"] = "Medium"

                return test_cases

        # Fall back to default tests based on app type
        return self._generate_default_tests(app_type, architecture)

    def _create_performance_prompt(self, app_type: str,
                                 architecture: Dict[str, Any],
                                 performance_reqs: Dict[str, Any]) -> str:
        """Create prompt for performance test generation"""
        return f"""
        Generate performance tests for a {app_type} application with the following architecture:
        {json.dumps(architecture, indent=2)}

        Performance requirements:
        {json.dumps(performance_reqs, indent=2)}

        Generate tests that verify:
        1. Response time under expected load
        2. System throughput and capacity
        3. Scalability characteristics
        4. Stability under sustained load
        5. Resource utilization (CPU, memory, network, etc.)
        6. Performance under stress conditions

        Include the following types of tests:
        - Load tests (normal operating conditions)
        - Stress tests (beyond normal operating conditions)
        - Endurance tests (sustained operation over time)
        - Spike tests (sudden increases in load)
        - Capacity tests (maximum operating capacity)

        For each test case, include:
        - A unique ID (PERF-XXX format)
        - A descriptive name
        - A detailed description of what is being tested
        - Steps to execute the test
        - Expected result
        - Priority (High/Medium/Low)
        - Metrics to measure

        Return the test cases in JSON format with the following structure:
        {{
            "test_cases": [
                {{
                    "id": "PERF-001",
                    "name": "Test Name",
                    "description": "Detailed description",
                    "steps": ["Step 1", "Step 2", ...],
                    "metrics": ["metric1", "metric2", ...],
                    "expected_result": "Expected outcome",
                    "priority": "High/Medium/Low"
                }},
                ...
            ]
        }}
        """

    def _generate_default_tests(self, app_type: str, architecture: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate default performance tests based on application type"""
        common_tests = [
            {
                "id": "PERF-001",
                "name": "Load Test - Expected User Load",
                "category": "Performance",
                "description": "Verify system performance under expected user load",
                "steps": [
                    "Set up test environment",
                    "Define expected user load scenario",
                    "Execute load test with gradual ramp-up",
                    "Measure response times and throughput",
                    "Monitor system resource utilization"
                ],
                "metrics": ["response_time", "throughput", "error_rate", "cpu_usage", "memory_usage"],
                "expected_result": "System maintains responsive performance under expected load",
                "priority": "High"
            },
            {
                "id": "PERF-002",
                "name": "Stress Test - Peak Load",
                "category": "Performance",
                "description": "Verify system behavior under peak load conditions",
                "steps": [
                    "Set up test environment",
                    "Define peak load scenario (2-3x normal load)",
                    "Execute stress test with rapid ramp-up",
                    "Measure response times and throughput",
                    "Monitor system resource utilization",
                    "Verify system stability"
                ],
                "metrics": ["response_time", "throughput", "error_rate", "cpu_usage", "memory_usage", "system_stability"],
                "expected_result": "System maintains acceptable performance under peak load without failure",
                "priority": "High"
            },
            {
                "id": "PERF-003",
                "name": "Endurance Test - Sustained Operation",
                "category": "Performance",
                "description": "Verify system stability during sustained operation",
                "steps": [
                    "Set up test environment",
                    "Define moderate load scenario",
                    "Execute test for an extended period (8+ hours)",
                    "Monitor response times and throughput over time",
                    "Monitor system resource utilization for trends",
                    "Check for memory leaks or degradation"
                ],
                "metrics": ["response_time_trend", "throughput_trend", "error_rate_trend", "memory_growth", "resource_utilization_trend"],
                "expected_result": "System maintains stable performance over extended periods without degradation",
                "priority": "Medium"
            },
            {
                "id": "PERF-004",
                "name": "Spike Test - Sudden Load Increase",
                "category": "Performance",
                "description": "Verify system handling of sudden load increases",
                "steps": [
                    "Set up test environment",
                    "Start with baseline load",
                    "Suddenly increase load to peak levels",
                    "Measure response to load spike",
                    "Verify system recovery after spike"
                ],
                "metrics": ["peak_response_time", "error_rate_during_spike", "recovery_time", "throughput_impact"],
                "expected_result": "System handles load spikes without failure and recovers quickly",
                "priority": "Medium"
            }
        ]

        # Add application type specific tests
        specific_tests = []

        if app_type.lower() in ["web", "webapp"]:
            specific_tests.extend([
                {
                    "id": f"PERF-{len(common_tests) + 1:03d}",
                    "name": "Page Load Performance Test",
                    "category": "Performance",
                    "description": "Verify page load times across key application pages",
                    "steps": [
                        "Identify key pages in the application",
                        "Measure page load metrics for each page",
                        "Verify metrics meet performance requirements",
                        "Test under various network conditions"
                    ],
                    "metrics": ["page_load_time", "time_to_first_byte", "time_to_interactive", "first_contentful_paint"],
                    "expected_result": "All pages load within acceptable time limits under various conditions",
                    "priority": "High"
                },
                {
                    "id": f"PERF-{len(common_tests) + 2:03d}",
                    "name": "CDN Performance Test",
                    "category": "Performance",
                    "description": "Verify performance of content delivery through CDN",
                    "steps": [
                        "Identify static resources served via CDN",
                        "Measure delivery performance from multiple geographic locations",
                        "Compare CDN performance to direct server delivery",
                        "Verify CDN caching behavior"
                    ],
                    "metrics": ["resource_load_time", "cache_hit_ratio", "geographic_variance"],
                    "expected_result": "CDN delivers static content efficiently across geographic regions",
                    "priority": "Medium"
                }
            ])

        elif app_type.lower() in ["mobile", "android", "ios"]:
            specific_tests.extend([
                {
                    "id": f"PERF-{len(common_tests) + 1:03d}",
                    "name": "App Launch Performance Test",
                    "category": "Performance",
                    "description": "Verify application startup performance",
                    "steps": [
                        "Measure cold start time on various device models",
                        "Measure warm start time on various device models",
                        "Verify startup times meet requirements",
                        "Identify resources loaded during startup"
                    ],
                    "metrics": ["cold_start_time", "warm_start_time", "memory_usage_at_launch", "cpu_usage_at_launch"],
                    "expected_result": "Application launches within acceptable time on all supported devices",
                    "priority": "High"
                },
                {
                    "id": f"PERF-{len(common_tests) + 2:03d}",
                    "name": "Network Performance Test",
                    "category": "Performance",
                    "description": "Verify application performance under various network conditions",
                    "steps": [
                        "Test application under strong network conditions",
                        "Test application under weak network conditions",
                        "Test application under intermittent connectivity",
                        "Verify appropriate handling of network transitions"
                    ],
                    "metrics": ["response_time_by_network", "data_usage", "offline_functionality"],
                    "expected_result": "Application performs acceptably under various network conditions",
                    "priority": "High"
                },
                {
                    "id": f"PERF-{len(common_tests) + 3:03d}",
                    "name": "Battery Consumption Test",
                    "category": "Performance",
                    "description": "Verify application impact on device battery life",
                    "steps": [
                        "Measure battery consumption during typical usage",
                        "Measure battery consumption during background operation",
                        "Compare battery usage to similar applications",
                        "Identify high battery consumption scenarios"
                    ],
                    "metrics": ["battery_drain_rate", "background_power_usage", "cpu_wakeups"],
                    "expected_result": "Application has minimal impact on device battery life",
                    "priority": "Medium"
                }
            ])

        # Database-specific performance tests if applicable
        if "database" in architecture:
            db_type = architecture["database"].get("type", "unknown")
            specific_tests.extend([
                {
                    "id": f"PERF-{len(common_tests) + len(specific_tests) + 1:03d}",
                    "name": f"{db_type} Database Performance Test",
                    "category": "Performance",
                    "description": "Verify database performance under load",
                    "steps": [
                        "Identify key database operations",
                        "Measure performance of read operations under load",
                        "Measure performance of write operations under load",
                        "Verify database scaling behavior"
                    ],
                    "metrics": ["query_response_time", "transactions_per_second", "connection_pool_utilization", "disk_io"],
                    "expected_result": "Database performs efficiently under expected load",
                    "priority": "High"
                }
            ])

        # Cloud/microservices specific tests if applicable
        if app_type.lower() in ["cloud", "microservices", "serverless"]:
            specific_tests.extend([
                {
                    "id": f"PERF-{len(common_tests) + len(specific_tests) + 1:03d}",
                    "name": "Auto-scaling Performance Test",
                    "category": "Performance",
                    "description": "Verify auto-scaling behavior under changing load",
                    "steps": [
                        "Start with minimal infrastructure",
                        "Gradually increase load to trigger scaling",
                        "Measure scaling response time",
                        "Verify performance during scaling operations",
                        "Reduce load and measure scale-down behavior"
                    ],
                    "metrics": ["scale_up_time", "scale_down_time", "performance_during_scaling", "resource_efficiency"],
                    "expected_result": "System scales efficiently in response to changing load",
                    "priority": "High"
                },
                {
                    "id": f"PERF-{len(common_tests) + len(specific_tests) + 2:03d}",
                    "name": "Service Dependency Performance Test",
                    "category": "Performance",
                    "description": "Verify performance impact of service dependencies",
                    "steps": [
                        "Identify service dependencies",
                        "Measure performance with all services functioning normally",
                        "Simulate degraded performance in dependent services",
                        "Measure impact on system performance",
                        "Verify circuit breaker and fallback mechanisms"
                    ],
                    "metrics": ["dependency_response_times", "fallback_activation_rate", "end_to_end_response_time"],
                    "expected_result": "System maintains acceptable performance when dependencies degrade",
                    "priority": "Medium"
                }
            ])

        # Combine common and specific tests
        return common_tests + specific_tests