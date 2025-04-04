#security_test_generator.py
"""
Purpose: Generates security tests, identifying vulnerabilities (e.g., SQL injection, XSS) and testing authentication, authorization, and data protection.
Key Method: generate_tests(input_data) analyzes code samples and app type to produce security-focused tests.
"""
"""
Security Test Generator Module - Generates security tests based on application components
"""
import json
import re
import requests
from typing import Dict, List, Any

class SecurityTestGenerator():
    """Generates security tests for application components"""

    def __init__(self):
        # Fixed URL typo: removed space in "inferen ce"
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
        print(f"message: {messages}")

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            print(f"response: {response}")
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {str(e)}")
            return json.dumps({"error": str(e)})
        
    def parse_json_response(self, response: str, default_key: str = "test_cases") -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        json_match = re.search(r'```(?:json)?\n(.*?)\n```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from code block: {e}")
                print("Response content in code block:", json_match.group(1)[:100] + "...")

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from full response: {e}")
            print("First 100 chars of response:", response[:100] + "...")
            print(f"Returning default structure with key: {default_key}")
            return {default_key: []}
        
    def generate_tests(self, input_data: Dict[str, Any], agent_id: str) -> List[Dict[str, Any]]:
        """Generate security tests based on input data"""
        print("input_data", input_data)
        app_type = input_data.get("app_type", "unknown")
        architecture = input_data.get("architecture", {})
        code_samples = input_data.get("code_samples", {})
        api_info = input_data.get("api", {})
        print("app_type", app_type)
        print("architecture", architecture)
        print("code_samples", code_samples)
        print("api", api_info)

        vulnerabilities = self._analyze_code_for_vulnerabilities(code_samples, agent_id)
        security_tests = []

        for i, vuln in enumerate(vulnerabilities):
            test = {
                "id": f"SEC-VULN-{i+1:03d}",
                "name": vuln.get("name", f"Test for {vuln['type']}"),
                "category": "Security",
                "description": vuln["description"],
                "steps": vuln.get("test_steps", ["Perform a manual security check"]),  # Use agent's steps or default
                "expected_result": vuln.get("expected_result", "The vulnerability should be mitigated"),  # Use agent's result or default
                "priority": vuln["severity"]
            }
            security_tests.append(test)

        # app_type_tests = self._generate_security_tests_by_app_type(app_type, architecture, api_info, agent_id)
        # security_tests.extend(app_type_tests)

        if not security_tests:
            return self._generate_default_tests()

        return security_tests

    def _analyze_code_for_vulnerabilities(self, code_samples: Dict[str, str], agent_id: str) -> List[Dict[str, Any]]:
        """Analyze all code samples at once for security vulnerabilities with a character limit"""
        vulnerabilities = []
        if not code_samples:
            return vulnerabilities

        MAX_CHAR_LIMIT = 5000  # Reduced from 10,000
        current_prompt_size = 0
        batched_code_samples = {}
        remaining_code_samples = dict(code_samples)

        system_prompt = """You are a security expert specializing in code analysis and vulnerability detection. Your expertise includes:
        1. Identifying OWASP Top 10 risks
        2. Static and dynamic code analysis
        3. Authentication and authorization analysis
        4. Cryptographic weakness detection
        5. API security and data validation"""

        while remaining_code_samples:
            batched_code_samples.clear()
            current_prompt_size = 0

            for filename, code in list(remaining_code_samples.items()):
                sample_size = len(filename) + len(code) + 100
                if current_prompt_size + sample_size <= MAX_CHAR_LIMIT:
                    batched_code_samples[filename] = code[:200] if len(code) > 200 else code  # Further reduced from 300
                    current_prompt_size += sample_size
                    del remaining_code_samples[filename]
                else:
                    break

            if not batched_code_samples:
                break

            prompt = self._create_vulnerability_analysis_prompt_for_batch(batched_code_samples)
            response = self._call_lyzr_api(
                agent_id=agent_id,
                session_id=agent_id,
                system_prompt=system_prompt,
                message=prompt
            )
            parsed_response = self.parse_json_response(response, "vulnerabilities")

            if "vulnerabilities" in parsed_response and parsed_response["vulnerabilities"]:
                for vuln in parsed_response["vulnerabilities"]:
                    if "id" not in vuln:
                        vuln["id"] = f"VULN-{len(vulnerabilities) + 1:03d}"
                    if "severity" not in vuln:
                        vuln["severity"] = "Medium"
                    vulnerabilities.append(vuln)

        return vulnerabilities

    def _create_vulnerability_analysis_prompt_for_batch(self, code_samples: Dict[str, str]) -> str:
        """Create a concise prompt for analyzing multiple code samples"""
        # Limit metadata to 100 chars per file and max 5 files
        MAX_FILES = 5
        MAX_CHARS_PER_FILE = 100

        code_metadata = ""
        for i, (filename, code) in enumerate(code_samples.items()):
            if i >= MAX_FILES:
                break
            truncated_code = code[:MAX_CHARS_PER_FILE] + ("..." if len(code) > MAX_CHARS_PER_FILE else "")
            code_metadata += f"File: {filename}\nMetadata: {truncated_code}\n"

        prompt = (
            "Analyze these code samples for vulnerabilities:\n"
            f"{code_metadata}"
            "For each vulnerability found, provide:\n"
            "- type: The category of the vulnerability (e.g., SQL Injection, XSS)\n"
            "- description: A detailed explanation of the issue\n"
            "- severity: Critical, High, Medium, or Low\n"
            "- file: The file where the issue was found\n"
            "- test_steps: A list of specific steps to test the vulnerability\n"
            "- expected_result: What should happen if the vulnerability is fixed\n"
            "Return the findings in JSON format:\n"
            "{\n"
                "\"vulnerabilities\": [{\n"
                    "\"type\": \"Category\", \"description\": \"Details\", \"severity\": \"Level\", "
                    "\"file\": \"Name\", \"test_steps\": [\"Step 1\", \"Step 2\"], \"expected_result\": \"Outcome\"\n"
                "}]\n"
            "}"
        )
        print(f"Prompt size: {len(prompt)} chars")  # Debug prompt size
        return prompt
    def _generate_security_tests_by_app_type(self, app_type: str, architecture: Dict[str, Any], api_info: Dict[str, Any], agent_id: str) -> List[Dict[str, Any]]:
        """Generate security tests based on application type"""
        security_tests = []
        test_id = 1

        security_tests.extend([
            {"id": f"SEC-{test_id:03d}", "name": "Input Validation Test", "category": "Security", "description": "Verify input validation", "steps": ["Identify inputs", "Test malicious inputs", "Verify sanitization"], "expected_result": "Inputs sanitized", "priority": "High"}
        ])
        test_id += 1

        security_tests.extend([
            {"id": f"SEC-{test_id:03d}", "name": "Authentication Security Test", "category": "Security", "description": "Verify authentication security", "steps": ["Test password policies", "Test lockout", "Test storage", "Test MFA"], "expected_result": "Secure authentication", "priority": "Critical"}
        ])
        test_id += 1

        security_tests.extend([
            {"id": f"SEC-{test_id:03d}", "name": "Session Management Test", "category": "Security", "description": "Verify session security", "steps": ["Test timeout", "Test token security", "Test logout", "Test concurrent sessions"], "expected_result": "Secure sessions", "priority": "High"}
        ])
        test_id += 1

        if app_type.lower() in ["web", "webapp"]:
            security_tests.extend([
                {"id": f"SEC-{test_id:03d}", "name": "XSS Test", "category": "Security", "description": "Verify XSS protection", "steps": ["Identify inputs/outputs", "Test XSS payloads", "Verify sanitization"], "expected_result": "No XSS", "priority": "Critical"}
            ])
            test_id += 1
            security_tests.extend([
                {"id": f"SEC-{test_id:03d}", "name": "CSRF Test", "category": "Security", "description": "Verify CSRF protection", "steps": ["Identify operations", "Forge requests", "Verify protection"], "expected_result": "No CSRF", "priority": "High"}
            ])
            test_id += 1

        elif app_type.lower() in ["mobile", "android", "ios"]:
            security_tests.extend([
                {"id": f"SEC-{test_id:03d}", "name": "Secure Data Storage Test", "category": "Security", "description": "Verify secure storage", "steps": ["Identify storage", "Verify encryption", "Check key management", "Test leakage"], "expected_result": "Secure storage", "priority": "Critical"}
            ])
            test_id += 1
            security_tests.extend([
                {"id": f"SEC-{test_id:03d}", "name": "Transport Security Test", "category": "Security", "description": "Verify secure transmission", "steps": ["Intercept traffic", "Verify HTTPS", "Test cert pinning", "Check plaintext"], "expected_result": "Secure transmission", "priority": "High"}
            ])
            test_id += 1

        if api_info:
            security_tests.extend([
                {"id": f"SEC-{test_id:03d}", "name": "API Authorization Test", "category": "Security", "description": "Verify API authorization", "steps": ["Identify endpoints", "Test privileges", "Verify checks"], "expected_result": "Proper authorization", "priority": "Critical"}
            ])
            test_id += 1

        return security_tests

    def _generate_default_tests(self) -> List[Dict[str, Any]]:
        """Generate default security tests"""
        return [
            {"id": "SEC-001", "name": "Authentication Security Test", "category": "Security", "description": "Verify authentication", "steps": ["Test policies", "Test lockout", "Test storage", "Test MFA"], "expected_result": "Secure auth", "priority": "Critical"},
            {"id": "SEC-002", "name": "Authorization Test", "category": "Security", "description": "Verify authorization", "steps": ["Identify resources", "Test roles", "Verify access"], "expected_result": "No unauthorized access", "priority": "Critical"},
            {"id": "SEC-003", "name": "Input Validation Test", "category": "Security", "description": "Verify input validation", "steps": ["Identify inputs", "Test malicious inputs", "Verify sanitization"], "expected_result": "Inputs sanitized", "priority": "High"},
            {"id": "SEC-004", "name": "Sensitive Data Exposure Test", "category": "Security", "description": "Verify data protection", "steps": ["Identify storage", "Verify encryption", "Check transmission", "Verify controls"], "expected_result": "Data protected", "priority": "High"},
            {"id": "SEC-005", "name": "Session Management Test", "category": "Security", "description": "Verify session security", "steps": ["Test timeout", "Test tokens", "Test logout", "Test concurrent"], "expected_result": "Secure sessions", "priority": "High"}
        ]