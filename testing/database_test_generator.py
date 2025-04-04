"""Purpose: Produces tests for database components, verifying schema integrity, CRUD operations, transactions, and performance.
Key Method: generate_tests(input_data) uses database schema and type to generate relevant tests.
"""
import json
import re
import requests
from typing import Dict, List, Any

class DatabaseTestGenerator:
    """Generates vibrant, project-specific tests for database components and operations"""

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
      """Generate enriched database tests leveraging all available project data"""
      print("inputdata", input_data)
      architecture = input_data.get("architecture", {})
      
      # Handle case where architecture is a list instead of a dict
      if isinstance(architecture, dict):
          database_info = architecture.get("database", {})
      else:
          database_info = {}  # Default to empty dict if architecture is a list
      
      db_type = database_info.get("type", "unknown")
      schema_info = input_data.get("database_schema", [])
      code_samples = input_data.get("code_samples", {})
      context = input_data.get("context", {})

      print("architecture", architecture)
      print("database_info", database_info)
      print("db_type", db_type)
      print("schema_info", schema_info)
      print("code_samples", code_samples)
      print("context", context)

      # Rest of the method remains unchanged
      if database_info or schema_info or context:
          prompt = self._create_database_prompt(database_info, schema_info, code_samples, context)
          system_prompt = """
          You are a database testing expert specializing in creating comprehensive test cases for database operations and integrity. Your expertise includes:

            1. Designing test cases to validate ACID properties (Atomicity, Consistency, Isolation, Durability)
            2. Creating tests for data validation, constraints, and referential integrity
            3. Developing performance tests for complex queries, indexes, and query optimization
            4. Building test suites for stored procedures, triggers, and database functions
            5. Designing test scenarios for database concurrency and transaction handling
            6. Creating migration and upgrade test plans to validate schema changes
            7. Developing test cases for data replication, sharding, and distributed database architectures
            8. Building security test scenarios for access control, authentication, and data protection
            9. Creating test data generation strategies that maintain referential integrity
            10. Designing test cases for backup and recovery operations
            11. Developing test scenarios for database failover and high availability configurations
            12. Creating test cases for ETL processes and data integration workflows
            13. Building test frameworks for database connection pooling and resource management
            14. Designing test scenarios for various database technologies (SQL, NoSQL, NewSQL)
            15. Developing specialized tests for temporal data, spatial data, and other complex data types

            You balance theoretical database principles with practical testing approaches, providing concrete test scenarios and SQL examples when appropriate. You adapt your testing strategies based on the specific database technology, data volumes, and business requirements. You emphasize both functional correctness and non-functional aspects like performance, security, and reliability in your database testing approach.
          """
          response = self._call_lyzr_api(agent_id, agent_id, system_prompt, prompt)
          parsed_response = self.parse_json_response(response)

          if "test_cases" in parsed_response and parsed_response["test_cases"]:
              test_cases = parsed_response["test_cases"]
              for i, test in enumerate(test_cases):
                  test["id"] = f"DB-{i+1:03d}"
                  test["category"] = "Database"
                  test["priority"] = test.get("priority", "Medium")
              return test_cases

      return self._generate_default_tests(db_type)
    def _create_database_prompt(self, database_info: Dict[str, Any], schema_info: List[Dict[str, Any]],
                               code_samples: Dict[str, str], context: Dict[str, Any]) -> str:
        """Craft a rich, lively prompt for database test generation"""
        db_type = database_info.get("type", "unknown")

        # Extract schema details with flair
        # schema_details = []
        # for schema in schema_info:
        #     fields = schema.get("fields", "")
        #     if isinstance(fields, str):
        #         fields_desc = fields
        #     elif isinstance(fields, list):
        #         fields_desc = ", ".join([f"{f.get('name')} ({f.get('type', 'unknown')})" for f in fields])
        #     else:
        #         fields_desc = "Unknown fields"
        #     schema_details.append(f"- {schema['model_name']} (from {schema['file_path']}): {fields_desc}")

        # Highlight key context from README and package.json
        readme_context = context.get("README.md", {})
        pkg_context = context.get("package.json", {})
        db_interactions = readme_context.get("Database Interactions", {})
        dependencies = pkg_context.get("Dependencies", {}).get("Libraries", [])
        project_desc = readme_context.get("Code Structure", "")
        queries_info = db_interactions.get("Queries", "Unknown query patterns")
        schemas_list = db_interactions.get("Schemas", "Unknown schemas")

        # Build a vibrant prompt
        return f"""
        Imagine you're testing the beating heart of an e-commerce empire—a {db_type} database powering a MERN stack backend! Your mission: craft a suite of dazzling, project-specific database tests that ensure this system thrives under pressure. Here’s the treasure trove of info at your fingertips:

        **Database Setup**:
        {json.dumps(database_info, indent=2)}

        **Schema Spotlight**:
        {json.dumps(schema_info) if schema_info else '- No schemas provided'}

        **Code Snippets**:
        {json.dumps(list(code_samples.keys()), indent=2) if code_samples else '- No code samples yet'}

        **Project Context**:
        - **Overview**: {project_desc}
        - **Dependencies**: {', '.join(dependencies) if dependencies else 'Mongoose, Express, and friends'}
        - **Database Flavor**: {db_interactions.get('Database', '')}
        - **Query Magic**: {queries_info}
        - **Known Schemas**: {schemas_list}
         Generate tests that verify:
        1. Database schema integrity and constraints
        2. CRUD operations functionality
        3. Data integrity and validation
        4. Transaction management
        5. Performance of critical queries
        6. Security of database access

        
        For each test, deliver:
        - **ID**: DB-XXX (e.g., DB-001)
        - **Name**: A catchy, descriptive title
        - **Description**: A vivid story of what’s being tested
        - **Steps**: Clear, actionable steps with e-commerce flavor
        - **Expected Result**: A concrete outcome tied to the project
        - **Priority**: High/Medium/Low based on criticality

        Return this in JSON format:
        {{
            "test_cases": [
                {{
                    "id": "DB-001",
                    "name": "Catchy Test Name",
                    "description": "A thrilling test scenario",
                    "steps": ["Step 1", "Step 2"],
                    "expected_result": "What should happen",
                    "priority": "High"
                }}
            ]
        }}
        """

    def _generate_default_tests(self, db_type: str = "unknown") -> List[Dict[str, Any]]:
        """Generate fallback tests with a touch of e-commerce flavor"""
        return [
            {
                "id": "DB-001",
                "name": "Product Stock Integrity Check",
                "category": "Database",
                "description": "Ensure product stock levels remain consistent during high-traffic sales",
                "steps": [
                    "Insert a product with 100 units in stock",
                    "Simulate 50 concurrent order requests",
                    "Check stock level after transactions",
                ],
                "expected_result": "Stock reduces accurately to 50 units, no overselling",
                "priority": "High"
            },
            {
                "id": "DB-002",
                "name": "Order Creation Blitz",
                "category": "Database",
                "description": "Verify order creation handles multi-item carts smoothly",
                "steps": [
                    "Create an order with 3 products",
                    "Save to the Order collection",
                    "Retrieve and validate order details",
                ],
                "expected_result": "Order saved with all items, totals correct",
                "priority": "High"
            },
            {
                "id": "DB-003",
                "name": "Payment Transaction Safety",
                "category": "Database",
                "description": "Test transaction rollback on payment failure",
                "steps": [
                    "Start a transaction for a $100 order",
                    "Simulate Stripe payment failure",
                    "Rollback the transaction",
                ],
                "expected_result": "No order saved, database unchanged",
                "priority": "Critical"
            }
        ]