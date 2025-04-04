"""
Purpose: Creates tests for API endpoints, covering functionality, validation, authentication, error handling, and performance.
Key Method: generate_tests(input_data) uses API specs (endpoints, methods) to generate comprehensive API tests.
"""
import json
import re
import requests
from typing import Dict, List, Any

class APITestGenerator:
    """Generates vibrant, project-specific tests for API endpoints"""

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
        print("payload", messages)
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            print(f"response: {response.json()}")
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {str(e)}")
            return json.dumps({"error": str(e)})

    def parse_json_response(self, response: str, default_key: str = "test_cases") -> Dict[str, Any]:
        """Parse JSON from LLM response with fallback"""
        json_match = re.search(r'```(?:json)?\n(.*?)\n```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from code block: {e}")
                print("Response content:", json_match.group(1)[:100] + "...")

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from full response: {e}")
            print("First 100 chars:", response[:100] + "...")
            return {default_key: []}

    def generate_tests(self, input_data: Dict[str, Any], agent_id: str) -> List[Dict[str, Any]]:
        """Generate enriched API tests leveraging all available project data"""
        print("input_Data", input_data)
        api_info = input_data.get("api", {})
        endpoints = api_info.get("endpoints", [])
        auth_type = api_info.get("auth_type", "None")
        code_samples = input_data.get("code_samples", {})
        context = input_data.get("context", {})

        print("api_info", api_info)
        print("endpoints", endpoints)
        print("auth_type", auth_type)
        print("code_samples", code_samples)
        print("context", context)

        if not endpoints:
            return self._generate_default_tests()

        # Use LLM for tailored tests
        prompt = self._create_api_prompt(endpoints, auth_type, code_samples, context)
        system_prompt = """
        You are an API testing virtuoso, crafting dazzling, project-specific test cases for a bustling e-commerce backend. Your expertise shines in:

        1. Validating RESTful endpoints with real-world e-commerce scenarios (e.g., payments, product queries)
        2. Testing JWT authentication and role-based access (e.g., admin vs. user)
        3. Ensuring Stripe webhook and payment intent endpoints handle edge cases
        4. Stress-testing performance under high-traffic sales conditions
        5. Securing APIs against injection, auth bypass, and data leaks
        6. Verifying error handling with vivid, customer-facing examples
        7. Testing integration with MongoDB Atlas via Mongoose-powered routes
        8. Simulating multi-user concurrency for order processing
        9. Validating request/response payloads for Express middleware chains
        10. Ensuring CORS and rate-limiting protect the API fortress

        You blend technical precision with e-commerce flair, delivering test cases that are functional, secure, and engaging. Tailor your tests to a MERN stack backend with Express, Passport JWT, Stripe, and Nodemailer, reflecting its modular MVC structure and RESTful design.
        """
        response = self._call_lyzr_api(agent_id, agent_id, system_prompt, prompt)
        parsed_response = self.parse_json_response(response)

        if "test_cases" in parsed_response and parsed_response["test_cases"]:
            test_cases = parsed_response["test_cases"]
            for i, test in enumerate(test_cases):
                test["id"] = f"API-{i+1:03d}"
                test["category"] = "API"
                test["priority"] = test.get("priority", "Medium")
            return test_cases

        # Fallback to manual generation if LLM fails
        return self._generate_api_tests_manually(endpoints, auth_type)

    def _create_api_prompt(self, endpoints: List[Dict[str, Any]], auth_type: str,
                          code_samples: Dict[str, str], context: Dict[str, Any]) -> str:
        """Craft a rich, lively prompt for API test generation"""
        # Format endpoints with pizzazz
        endpoint_details = []
        for ep in endpoints:
            method = ep.get("method", "GET")
            route = ep.get("route", "/unknown")
            desc = ep.get("description", "Mystery endpoint")
            params = ep.get("parameters", [])
            security = ep.get("security", [])
            endpoint_details.append(f"- {method} {route}: {desc} (Params: {params}, Security: {security})")

        # Extract juicy context
        readme_context = context.get("README.md", {})
        pkg_context = context.get("package.json", {})
        api_components = readme_context.get("API Components", {})
        security_practices = readme_context.get("Security Practices", {})
        dependencies = pkg_context.get("Dependencies", {}).get("Libraries", [])
        project_desc = readme_context.get("Code Structure", "A MERN stack e-commerce backend")
        known_endpoints = api_components.get("Endpoints", "Unknown endpoints")

        # Build an electrifying prompt
        return f"""
        Step into the bustling marketplace of a MERN stack e-commerce empire! Your task: forge a suite of electrifying API tests for this Express-powered backend. Here’s the vibrant scene you’re working with:

        **API Endpoints**:
        {chr(10).join(endpoint_details) if endpoint_details else '- A few endpoints to start, but more lurk in the shadows!'}

        **Authentication**:
        - Stated: {auth_type}
        - Reality Check: {security_practices.get('Authentication', 'Passport JS with JWT')} (trust the context—it’s JWT-powered!)

        **Code Snippets**:
        {json.dumps(list(code_samples.keys()), indent=2) if code_samples else '- No snippets yet, but picture Express routes and Mongoose magic!'}

        **Project Spotlight**:
        - **Overview**: {project_desc}
        - **Dependencies**: {', '.join(dependencies) if dependencies else 'Express, Passport, Stripe, and more'}
        - **API Flavor**: {api_components.get('Request/Response', 'REST API using Express')}
        - **Known Routes**: {', '.join(known_endpoints) if isinstance(known_endpoints, list) else known_endpoints}
        - **Security**: {security_practices.get('Authentication', 'JWT')} with {security_practices.get('Authorization', 'role-based access')}
        - **Extras**: Stripe for payments, Nodemailer for emails, MongoDB Atlas backend

        **Your Mission**:
        Craft test cases that make these APIs sing! Focus on:
        1. Functionality: Does /webhook dance with Stripe events? Does /create-payment-intent spark a sale?
        2. Validation: Can invalid payloads crash the party?
        3. Auth: Does JWT guard the gates (despite ‘None’ in auth_type—context says otherwise)?
        4. Performance: Can it handle a Black Friday stampede?
        5. Security: Are we safe from injection bandits?
        6. Errors: Do failures tell a helpful tale?

        For each test, deliver:
        - **ID**: API-XXX (e.g., API-001)
        - **Name**: A snappy, memorable title
        - **Description**: A thrilling story of what’s tested
        - **Steps**: Clear, e-commerce-infused steps
        - **Expected Result**: A vivid outcome tied to the project
        - **Priority**: High/Medium/Low based on impact

        Return this in JSON format:
        {{
            "test_cases": [
                {{
                    "id": "API-001",
                    "name": "Snappy Test Name",
                    "description": "A thrilling test tale",
                    "steps": ["Step 1", "Step 2"],
                    "expected_result": "What happens",
                    "priority": "High"
                }}
            ]
        }}
        """

    def _generate_default_tests(self) -> List[Dict[str, Any]]:
        """Generate fallback tests with e-commerce flair"""
        return [
            {
                "id": "API-001",
                "name": "Payment Webhook Listener",
                "category": "API",
                "description": "Ensure the webhook catches Stripe’s payment whispers",
                "steps": [
                    "Simulate a Stripe payment event",
                    "Send POST to /webhook",
                    "Check response and database update",
                ],
                "expected_result": "200 OK, payment event logged",
                "priority": "High"
            },
            {
                "id": "API-002",
                "name": "Checkout Intent Kickoff",
                "category": "API",
                "description": "Verify payment intents spark to life",
                "steps": [
                    "POST to /create-payment-intent with cart data",
                    "Validate response has Stripe intent ID",
                ],
                "expected_result": "200 OK, intent ID returned",
                "priority": "High"
            },
            {
                "id": "API-003",
                "name": "Auth Gatekeeper",
                "category": "API",
                "description": "Test if JWT locks out uninvited guests",
                "steps": [
                    "Send request to protected endpoint without token",
                    "Check for 401 Unauthorized",
                ],
                "expected_result": "401 Unauthorized, access denied",
                "priority": "High"
            }
        ]

    def _generate_api_tests_manually(self, endpoints: List[Dict[str, Any]], auth_type: str) -> List[Dict[str, Any]]:
        """Generate manual API tests with project-specific tweaks"""
        test_cases = []
        test_id = 1

        for endpoint in endpoints:
            method = endpoint.get("method", "GET")
            route = endpoint.get("route", "/api/resource")
            desc = endpoint.get("description", "API Endpoint")

            test_cases.append({
                "id": f"API-{test_id:03d}",
                "name": f"{method} {route} Happy Path",
                "category": "API",
                "description": f"Verify {desc} works like a charm",
                "steps": [
                    f"Send {method} request to {route} with valid data",
                    "Check response status and payload",
                ],
                "expected_result": "200 OK, expected data returned",
                "priority": "High"
            })
            test_id += 1

            if method in ["POST", "PUT", "PATCH"]:
                test_cases.append({
                    "id": f"API-{test_id:03d}",
                    "name": f"{method} {route} Bad Input Bust",
                    "category": "API",
                    "description": f"Ensure {desc} rejects junk data",
                    "steps": [
                        f"Send {method} request to {route} with invalid payload",
                        "Verify error response",
                    ],
                    "expected_result": "400 Bad Request, validation error detailed",
                    "priority": "Medium"
                })
                test_id += 1

            # Override auth_type with JWT from context
            if "jwt" in str(endpoint).lower() or auth_type.lower() != "none":
                test_cases.append({
                    "id": f"API-{test_id:03d}",
                    "name": f"{method} {route} JWT Shield",
                    "category": "API",
                    "description": f"Confirm {desc} demands a valid JWT",
                    "steps": [
                        f"Send {method} to {route} without JWT",
                        "Check for 401",
                        f"Send with valid JWT",
                        "Check for 200",
                    ],
                    "expected_result": "401 without JWT, 200 with valid JWT",
                    "priority": "High"
                })
                test_id += 1

            test_cases.append({
                "id": f"API-{test_id:03d}",
                "name": f"{method} {route} Error Drama",
                "category": "API",
                "description": f"Test {desc}’s error theatrics",
                "steps": [
                    f"Trigger an error condition for {route}",
                    "Verify status and message",
                ],
                "expected_result": "Appropriate error code (e.g., 500), helpful message",
                "priority": "Medium"
            })
            test_id += 1

        return test_cases