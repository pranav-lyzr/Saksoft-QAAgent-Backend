import os
import json
import logging
import requests
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestPlanGenerator:
    """
    Test Plan Generator using OpenAI's GPT model
    """

    def __init__(self):
        """
        Initialize the test plan generator

        Args:
            api_key (str): OpenAI API key
        """
        # self.client = OpenAI(api_key=api_key)
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
        logger.info(f"prompt: {message}")

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
            logger.info(f"response: {response}")
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {str(e)}")
            return json.dumps({"error": str(e)})

    def generate_test_plan(self, user_story, api_data, db_data, ui_data, technical_insights, rag_Data, agent_id, custom_change=None):
        """
        Generate a comprehensive test plan

        Args:
            user_story (str): The user story to generate a test plan for
            api_data (list): API insights
            db_data (list): Database insights
            ui_data (list): UI insights
            technical_insights (str): Technical considerations
            custom_change (str, optional): Additional custom change requirements

        Returns:
            str: Generated test plan in plain text
        """
        # Prepare context
        context = {
            "api_insights": api_data,
            "database_insights": db_data,
            "ui_insights": ui_data,
            "rag_Data": rag_Data,
            "technical_insights": technical_insights,
        }
        print("api_insights",api_data)
        print("db_data",db_data)
        print("ui_data",ui_data)
        print("technical_insights",technical_insights)
        print("user_story",user_story)
        print("agent_id",agent_id)
        # Construct prompt for test plan generation
        prompt = f"""
        Generate a comprehensive test plan for the following user story:

        User Story: {user_story}

        Custom Change Requirements: {custom_change or 'None'}

        Context:
        - API Insights: {json.dumps(context['api_insights'], indent=2)}
        - Database Insights: {json.dumps(context['database_insights'], indent=2)}
        - UI Insights: {json.dumps(context['ui_insights'], indent=2)}
        - Technical Insights: {context['technical_insights']}

        Please provide a detailed test plan in plain text format that includes:
        1. Test Plan Overview
        2. Scope of Testing
        3. Test Scenarios
        4. Test Cases
        5. Risk Assessment
        6. Test Environment Requirements
        7. Test Data Requirements
        8. Acceptance Criteria

        Ensure the test plan is practical, comprehensive, and directly addresses the user story.
                
        1. **Extract Requirements**
        - Analyze the user stories and technical insights to extract both **functional** and **non-functional requirements**.
        - Identify user expectations, system constraints, and any specific conditions that must be tested.

        2. **Generate Test Scenarios**
        - Develop a comprehensive list of test scenarios that cover:
            - All **functional requirements** (e.g., core features and workflows).
            - **Non-functional requirements** (e.g., performance, security, usability).
            - **Edge cases**, **boundary conditions**, and **error handling**.
            - **Integration points** and dependencies between components.

        3. **Trace to Codebase**
        - For each test scenario, identify and document the specific parts of the codebase (e.g., functions, classes, modules) that implement the corresponding functionality.
        - Ensure traceability to guarantee that all relevant code paths are tested and that the test plan reflects the current state of the codebase.

        4. **Define Detailed Test Cases**
        - For every test scenario, create clear and unambiguous test cases that include:
            - **Inputs** and **preconditions**.
            - **Expected outputs** and **postconditions**.
            - **Acceptance criteria** directly tied to the requirements.
            - Necessary **test data** or **environment setup instructions**.

        5. **Evaluate Automation Potential**
        - Identify test cases suitable for automation based on:
            - **Repeatability** and frequency of execution.
            - **Complexity** and time required for manual testing.
            - **Criticality** to the system’s functionality.
        - Provide recommendations on automation tools, frameworks, and strategies.
        - Suggest including **code coverage measurements** during automated testing to verify that the test suite adequately covers the codebase.

        6. **Prioritize Test Cases**
        - Assign priority levels (e.g., high, medium, low) to each test case based on:
            - **Risk of failure** and potential impact on the system.
            - **Frequency of use** in the application.
            - **Criticality** to business objectives.
            - **Dependencies** on other test cases or system components.

        7. **Adhere to Provided Information**
        - Base the entire test plan **exclusively** on the provided Requirements, Architecture, and Guidelines (RAG), user stories, technical insights, and the existing codebase.
        - Do not include tests for features or functionalities not explicitly mentioned in these sources.

        **Additional Guidelines**:
        - Ensure the test plan is **practical**, **actionable**, and tailored to the specific project context.
        - Use **clear, consistent terminology** and reference specific code elements (e.g., file names, function names) where applicable.
        - Consider the project’s development and testing environment when suggesting tools or frameworks.
        """

        try:
            # Call OpenAI API to generate test plan
            # response = self.client.chat.completions.create(
            #     model="gpt-4-turbo",
            #     messages=[
            #         {"role": "system", "content": "You are an expert test plan generator creating a comprehensive test strategy."},
            #         {"role": "user", "content": prompt}
            #     ],
            #     max_tokens=2000,
            #     temperature=0.7
            # )

            response = self._call_lyzr_api(
                agent_id= agent_id,
                session_id= agent_id,
                system_prompt="You are an expert test plan generator creating a comprehensive test strategy.",
                message=prompt
            )
            # Extract and return the test plan text
            test_plan_text = response;

            # Add metadata to the test plan text
            metadata = f"""
            TEST PLAN METADATA
            ------------------
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

            """

            return metadata + test_plan_text

        except Exception as e:
            return f"Error generating test plan: {str(e)}"