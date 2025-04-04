import json
import re
import requests
from typing import Dict, List, Any

class StoryTestGenerator():
    def __init__(self):
        self.api_url = "https://agent-prod.studio.lyzr.ai/v3/inference/chat/"
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": "sk-default-yStV4gbpjadbQSw4i7QhoOLRwAs5dEcl"
        }
        self.user_id = "pranav@lyzr.ai"
      
    def _call_lyzr_api(self, agent_id: str, session_id: str, system_prompt: str, message: str) -> str:
        """Optimized helper function to call Lyzr API with simplified payload"""
        # Simplified payload structure
        payload = {
            "user_id": self.user_id,
            "agent_id": agent_id,
            "session_id": session_id,
            "message": json.dumps([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ])
        }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
            )
            
            # Print full response for debugging
            print(f"Response Status Code: {response.status_code}")
            print(f"Response Headers: {response.headers}")
            print(f"Response payload: {payload}")
            print(f"Response Content: {response.text}")

            # Raise an exception for bad status codes
            response.raise_for_status()

            # Parse the response JSON
            response_json = response.json()
            
            # Return the response content
            return response_json.get("response", response_json)
        
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {str(e)}")
            return json.dumps({"error": str(e)})
        except ValueError as e:
            print(f"JSON parsing error: {str(e)}")
            return json.dumps({"error": "Failed to parse response"})

    def generate_tests(self, input_data: Dict[str, Any], agent_id:str) -> List[Dict[str, Any]]:
        """Generate user story tests based on input data"""
        print("input data", input_data)
        user_stories = input_data.get("user_stories", [])
        features = input_data.get("features", [])
        requirements = input_data.get("requirements", {})
        app_type = input_data.get("app_type", "unknown")

        # Build context string from all user stories
        story_context = "\n".join(
            f"User Story {idx+1}:\n"
            f"- Title: {story.get('title', '')}\n"
            f"- Description: {story.get('description', '')}\n"
            f"- Category: {story.get('category', '')}\n"
            f"- Priority: {story.get('priority', '')}\n"
            f"- Features: {', '.join(story.get('features', []))}\n"
            f"- Topics: {', '.join(story.get('topics', []))}"
            for idx, story in enumerate(user_stories)
        )

        print("stories", story_context)
        print("features", features)
        print("requirements", requirements)
        print("app_type", app_type)

        # Use LLM to generate story tests if we have user stories or requirements
        if story_context or features or requirements:
            prompt = self._create_story_prompt(story_context, features, requirements, app_type)
            
            system_prompt = """You are a test engineer specializing in creating comprehensive test cases from user stories and requirements. Your expertise includes analyzing user stories, developing detailed test scenarios, creating test matrices, translating requirements into validation criteria, and designing comprehensive test cases that cover functional and non-functional aspects of the system."""

            response = self._call_lyzr_api(
                agent_id=agent_id,
                session_id=agent_id,
                system_prompt=system_prompt,
                message=prompt
            )
            
            # Add print for debugging the response
            print("API Response:", response)

            # Parse the response
            parsed_response = self.parse_json_response(response)

            if "test_cases" in parsed_response and parsed_response["test_cases"]:
                # Ensure all test cases have required fields and proper category
                test_cases = parsed_response["test_cases"]
                for test in test_cases:
                    if "id" not in test:
                        test["id"] = f"STORY-{len(test_cases):03d}"
                    test["category"] = "User Story"
                    if "priority" not in test:
                        test["priority"] = "Medium"

                return test_cases

        # Fall back to default tests based on app type and features
        return self._generate_default_tests(app_type, features)

    def _create_story_prompt(self, stories: str,
                       features: List[str],
                       requirements: List[Dict[str, Any]],
                       app_type: str) -> str:
        """Create a comprehensive and enriched prompt for test generation"""
        # Prepare user stories
        # formatted_stories = "\n".join([
        #     f"• {story}" for story in stories
        # ])
        # print("formatted_stories__________________",formatted_stories)
        # Prepare features with detailed context
        formatted_features = "\n".join([
            f"• {feature}" for feature in features
        ]) if features else "No specific features defined"

        # Prepare requirements with additional details
        formatted_requirements = "\n".join([
            f"• {req}" for req in requirements
        ]) if requirements else "No explicit functional requirements provided"

        # Context about the application type
        app_type_details = {
            "web": "Web-based network management application with browser-based interface",
            "mobile": "Mobile application for on-the-go network monitoring and management",
            "desktop": "Comprehensive desktop application for network operations",
            "unknown": "Telecom Network Management System with flexible deployment"
        }.get(app_type.lower(), f"Custom {app_type} Network Management Application")

        return f"""Comprehensive Test Case Generation for Telecom Network Management System

    System Context:
    - Application Type: {app_type_details}
    - Deployment Environment: Enterprise-grade telecom network management
    - Primary Objectives: Reliability, Performance, Compliance, and User Experience

    Detailed Input Analysis:

    1. User Stories Breakdown:
    {stories}

    2. System Features:
    {formatted_features}

    3. Functional Requirements:
    {formatted_requirements}

    Test Generation Comprehensive Guidelines:
    - Develop tests that cover each user story's explicit and implicit requirements
    - Design test scenarios addressing multiple dimensions:
    * Functional Validation
    * Performance Verification
    * Security Compliance
    * Error Handling
    * User Experience
    - Prioritize tests based on business impact and risk assessment
    - Ensure comprehensive coverage across different operational scenarios

    Test Case Composition Requirements:
    - Unique Test Identifier (TELECOM-XXX format)
    - Descriptive and Clear Test Name
    - Detailed Test Objective
    - Specific, Reproducible Test Steps
    - Precise Expected Outcomes
    - Priority Categorization
    - Potential Failure Mode Analysis
    - Traceability to Original User Stories

    Specific Focus Areas:
    1. Network Reliability
    2. Real-time Monitoring Capabilities
    3. Predictive Maintenance Mechanisms
    4. Compliance and Regulatory Adherence
    5. User Support and Issue Resolution Workflow
    6. Configuration Management Automation

    Output Structured Format:
    {{
        "test_cases": [
            {{
                "id": "TELECOM-XXX",
                "name": "Descriptive Test Name",
                "description": "Comprehensive test scenario description",
                "user_story_reference": "Original user story mapping",
                "steps": ["Detailed executable steps"],
                "expected_result": "Precise, measurable outcome",
                "priority": "High/Medium/Low",
                "failure_modes": ["Potential failure scenarios"],
                "risk_level": "Critical/High/Medium/Low"
            }}
        ]
    }}

    Critical Evaluation Criteria:
    - Thoroughness of Test Coverage
    - Alignment with User Story Intents
    - Practical Executability
    - Comprehensive Scenario Representation

    Generate test cases that not only validate functionality but also provide strategic insights into the telecom network management system's capabilities, limitations, and improvement opportunities."""

    # Rest of the class remains the same
    def parse_json_response(self, response: str, default_key: str = "test_cases") -> Dict[str, Any]:
        """Parse JSON from LLM response with improved error handling"""
        # If response is already a dictionary, return it
        if isinstance(response, dict):
            return response

        # Try to extract JSON from code blocks
        json_match = re.search(r'```(?:json)?\n(.*?)\n```', str(response), re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from code block: {e}")

        # Try to parse the entire response as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from full response: {e}")

            # If all parsing fails, return a default structure
            return {default_key: [
                {
                    "id": "TELECOM-DEFAULT-001",
                    "name": "Default Network Availability Test",
                    "description": "Verify basic network availability and responsiveness",
                    "steps": [
                        "Initiate network connectivity check",
                        "Verify system responds within expected timeframe",
                        "Confirm no critical errors or warnings"
                    ],
                    "expected_result": "Network management system is operational and responsive",
                    "priority": "High"
                }
            ]}

    def _generate_default_tests(self, app_type: str, features: List[str]) -> List[Dict[str, Any]]:
        """Generate default user story tests based on application type and features"""
        return [
            {
                "id": "TELECOM-001",
                "name": "Network Downtime Reduction Verification",
                "category": "User Story",
                "description": "Verify system capabilities for reducing network downtime",
                "steps": [
                    "Simulate network stress conditions",
                    "Monitor system response and recovery mechanisms",
                    "Measure total downtime during stress test",
                    "Compare with baseline performance metrics"
                ],
                "expected_result": "System demonstrates ability to minimize and quickly recover from network disruptions",
                "priority": "High"
            }
        ]