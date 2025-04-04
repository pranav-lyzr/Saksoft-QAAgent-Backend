"""
LLM Provider Module - Handles interactions with language models for test generation
Uses real LLM calls exclusively without simulation
"""
import os
import json
import re
from typing import Dict, Any, Optional

class LLMProvider:
    """Interface for LLM interactions used by test generators"""

    def __init__(self, api_key=None, model="gpt-3.5-turbo"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.client = self._initialize_api()

    def _initialize_api(self):
        """Initialize the LLM API connection"""
        try:
            import openai
            if self.api_key:
                client = openai.OpenAI(api_key=self.api_key)
                # Test the connection with a simple query
                test_response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                )
                print(f"LLM connection established successfully using model: {self.model}")
                return client
            else:
                print("No API key provided. Please set your OpenAI API key.")
                return None
        except Exception as e:
            print(f"Error initializing OpenAI API: {e}")
            print("Please check your API key and internet connection.")
            return None

    def query(self, prompt: str, system_prompt: str = None) -> str:
        """Send a query to the LLM and get a response"""
        if not self.client:
            raise ValueError("LLM client not initialized. Please provide a valid API key.")

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            print(f"Sending query to LLM... (prompt length: {len(prompt)} chars)")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,  # Using low temperature for more deterministic outputs
                max_tokens=4000
            )

            return response.choices[0].message.content
        except Exception as e:
            error_msg = f"Error querying LLM API: {e}"
            print(error_msg)
            raise RuntimeError(error_msg)

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