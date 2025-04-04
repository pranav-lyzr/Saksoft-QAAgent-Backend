#Architecture Test Generator Module
"""Purpose: Generates tests to validate the system’s architecture, such as layer separation, dependency flow, and architectural patterns (e.g., Factory, Singleton).
Key Method: generate_tests(input_data) takes architecture details, patterns, tools, and technical insights, then produces tests to ensure architectural integrity.
"""
"""
Architecture Test Generator Module - Generates tests that validate system architecture
"""
import json
import requests
import re
from typing import Dict, List, Any

class ArchitectureTestGenerator():
    """Generates tests that validate system architecture principles and patterns"""

    def __init__(self):
        self.api_url = "https://agent-prod.studio.lyzr.ai/v3/inference/chat/"
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": "sk-default-yStV4gbpjadbQSw4i7QhoOLRwAs5dEcl"
        }
        self.user_id = "pranav@lyzr.ai"
        

    def _call_lyzr_api(self, prompt: str, agent_id: str) -> str:
        """Helper function to call Lyzr API with proper message formatting"""
        payload = {
            "user_id": self.user_id,
            "agent_id": agent_id,
            "session_id": "arch-test-session",
            "message": prompt
        }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {str(e)}")
            return json.dumps({"error": str(e)})

    def generate_tests(self, input_data: Dict[str, Any], agent_id: str) -> List[Dict[str, Any]]:
        """Generate architecture validation tests with proper prompt structure"""
        print("input_data", input_data)
        architecture = input_data.get("architecture", {})
        patterns = input_data.get("patterns", [])
        tools = input_data.get("tools", {})
        technical_insights = input_data.get("technical_insights", [])
        
        print("architecture", architecture)
        print("patterns", patterns)
        print("tools", tools)
        print("technical_insights", technical_insights)

        # Format technical insights for the prompt
        insights_str = "No technical insights provided"
        if technical_insights:
            insights_str = "\n".join([
                f"- {insight['text']} (Analysis: {json.dumps(insight['analysis_result'], indent=2)})"
                for insight in technical_insights
            ])

        structured_prompt = f"""
        **Architecture Test Generation Request**
        
        System Context:
        {json.dumps(architecture, indent=2)}
        
        Design Patterns: {', '.join(patterns) or 'None specified'}
        
        Technology Stack:
        {json.dumps(tools, indent=2) if tools else 'No tools specified'}
        
        Technical Insights:
        {insights_str}
        
        **Test Requirements:**
        1. Validate architecture pattern implementation
        2. Verify technology integration points
        3. Check compliance with quality attributes
        4. Ensure proper component boundaries
        5. Validate cross-cutting concerns
        6. Address technical insights findings (e.g., strengths, weaknesses, improvements)
        
        **Output Format:**
        {{
            "test_cases": [
                {{
                    "id": "ARCH-001",
                    "name": "Pattern Validation - [Pattern Name]",
                    "scope": "Architecture",
                    "type": "Structural",
                    "priority": "High",
                    "steps": [
                        "Inspect component relationships",
                        "Verify pattern implementation",
                        "Check interface contracts"
                    ],
                    "expected_result": "System implements [Pattern] correctly",
                    "validation_method": "Code Analysis/Review",
                    "related_insight": "Optional: Reference to technical insight text"
                }}
            ]
        }}
        """

        response = self._call_lyzr_api(structured_prompt, agent_id)
        return self._parse_response(response)

    def _parse_response(self, response: str) -> List[Dict[str, Any]]:
        """Handle response parsing with robust error checking"""
        try:
            # Clean response and extract JSON
            clean_response = re.sub(r'[\x00-\x1F]+', '', response)
            json_str = re.search(r'\{.*\}', clean_response, re.DOTALL).group()
            parsed = json.loads(json_str)
            
            # Validate test case structure
            test_cases = parsed.get("test_cases", [])
            return [self._validate_test_case(tc) for tc in test_cases]
            
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Failed to parse response: {str(e)}")
            return self._generate_fallback_tests()

    def _validate_test_case(self, test_case: Dict) -> Dict:
        """Ensure test case contains required fields"""
        required_fields = ["id", "name", "steps", "expected_result"]
        optional_fields = ["related_insight"]
        
        for field in required_fields:
            if field not in test_case:
                test_case[field] = "Not Specified"
        
        # Ensure optional fields have defaults
        for field in optional_fields:
            if field not in test_case:
                test_case[field] = None
                
        return test_case

    def _generate_fallback_tests(self) -> List[Dict[str, Any]]:
        """Generate basic architecture tests when parsing fails"""
        return [
            {
                "id": "ARCH-001",
                "name": "Architectural Pattern Validation",
                "scope": "Architecture",
                "type": "Structural",
                "priority": "High",
                "steps": [
                    "Review architecture documentation",
                    "Inspect component relationships",
                    "Verify pattern implementation"
                ],
                "expected_result": "System implements declared architectural patterns correctly",
                "validation_method": "Design Review",
                "related_insight": None
            },
            {
                "id": "ARCH-002",
                "name": "Technical Insights Compliance",
                "scope": "Architecture",
                "type": "Compliance",
                "priority": "Medium",
                "steps": [
                    "Review technical insights",
                    "Verify implementation of suggested improvements",
                    "Check resolution of identified weaknesses"
                ],
                "expected_result": "System addresses key technical insights",
                "validation_method": "Code Review",
                "related_insight": "General technical insights review"
            }
        ]