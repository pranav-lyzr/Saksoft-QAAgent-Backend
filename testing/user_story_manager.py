import os
import json
import logging
import requests
from typing import List, Dict, Any
from bson import ObjectId
from fastapi import HTTPException
import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UserStoryManager:
    """Manager for user stories that stores data in MongoDB."""

    def __init__(self, db=None):
        """Initialize the manager with a MongoDB database connection."""
        self.db = db
        self.analysis_results = []
        self.categories = []
        self.topics = []
        self.features = []
        self.api_url = "https://agent-prod.studio.lyzr.ai/v3/inference/chat/"
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": "sk-default-yStV4gbpjadbQSw4i7QhoOLRwAs5dEcl"
        }
        self.user_id = "pranav@lyzr.ai"
        self.prd_extraction_agent_id = "67e64c0a235e2e0d922cc41b"

    def set_db(self, db):
        """Set the MongoDB database connection."""
        self.db = db
        logger.info("MongoDB connection configured")

    def _call_lyzr_api(self, agent_id: str, session_id: str, text: str, prompt: str) -> str:
        """Helper function to call Lyzr API."""
        formatted_prompt = prompt.format(text=text)
        payload = {
            "user_id": self.user_id,
            "agent_id": agent_id,
            "session_id": session_id,
            "message": formatted_prompt
        }
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            logger.info(f"Response from Lyzr API: {response.json()}")
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return json.dumps({"error": str(e)})

    async def extract_from_prd(self, prd_text: str, project_id: str) -> Dict[str, Any]:
        """Extract user stories, features, and requirements from a PRD document and store in user_stories."""
        if self.db is None:
            raise HTTPException(status_code=500, detail="MongoDB connection not configured")

        # Define the extraction prompt
        extraction_prompt = """
        You’re a master extractor diving into a Product Requirements Document (PRD) for an e-commerce MERN stack project. Your task is to unearth user stories, features, and requirements from this PRD text:

        PRD Text: "{text}"

        Extract and structure the following:
        1. **User Stories**: Identify distinct user stories in the format "As a [user_role], I want [goal] so that [benefit]." List key features described or implied in the PRD also as user stories. Capture functional and non-functional requirements also as user stories.
        2. **Features**: List key features described or implied in the PRD (e.g., "Product filtering", "Order checkout").
        3. **Requirements**: Capture functional and non-functional requirements (e.g., "System must handle 1000 concurrent users").

        Return the result in JSON format:
        {{
            "user_stories": ["As a customer, I want to filter products by category so that I can find items quickly", ...],
            "features": ["Product filtering", "Order checkout", ...],
            "requirements": ["System must handle 1000 concurrent users", "Response time under 2 seconds", ...]
        }}

        Ensure the output is vivid, precise, and tied to e-commerce workflows. Return ONLY valid JSON, no extra text.
        """

        logger.info(f"Extracting from PRD for project {project_id}")
        try:
            # Call the PRD extraction agent
            result = self._call_lyzr_api(
                agent_id=self.prd_extraction_agent_id,
                session_id=f"prd_{project_id}",
                text=prd_text,
                prompt=extraction_prompt
            )
            logger.info(f"Raw result from API: {result}")

            # Handle case where result is an error JSON string
            if result.startswith('{"error":'):
                error_data = json.loads(result)
                raise HTTPException(status_code=500, detail=f"Lyzr API error: {error_data['error']}")

            # Clean up and parse the result
            cleaned_result = result.replace('```json', '').replace('```', '').strip()
            extracted_data = json.loads(cleaned_result)
            logger.info(f"Extracted data: {extracted_data}")

            # Extract user stories, features, and requirements
            user_stories = extracted_data.get("user_stories", [])
            features = extracted_data.get("features", [])
            requirements = extracted_data.get("requirements", [])
            if features:
                await self.db.projects.update_one(
                    {"_id": ObjectId(project_id)},
                    {"$addToSet": {"extracted_features": {"$each": features}}}
                )
            if requirements:
                await self.db.projects.update_one(
                    {"_id": ObjectId(project_id)},
                    {"$addToSet": {"extracted_requirements": {"$each": requirements}}}
                )
            # Store each user story as a separate document in user_stories
            inserted_ids = []
            if user_stories:
                for story_text in user_stories:
                    story_doc = {
                        "project_id": ObjectId(project_id),
                        "text": story_text,
                        "prd_text": prd_text,
                        # "extracted_features": features,
                        # "extracted_requirements": requirements,
                        "analyzed": False,
                        "analysis_result": None,
                        "created_at": datetime.datetime.utcnow().isoformat()
                    }
                    insert_result = await self.db.user_stories.insert_one(story_doc)
                    inserted_ids.append(str(insert_result.inserted_id))
            else:
                # If no user stories, still save a document to record the PRD
                story_doc = {
                    "project_id": ObjectId(project_id),
                    "text": None,  # No user story extracted
                    "prd_text": prd_text,
                    # "extracted_features": features,
                    # "extracted_requirements": requirements,
                    "analyzed": False,
                    "analysis_result": None,
                    "created_at": datetime.datetime.utcnow().isoformat(),
                    "no_user_stories": True  # Flag to indicate no user stories were extracted
                }
                insert_result = await self.db.user_stories.insert_one(story_doc)
                inserted_ids.append(str(insert_result.inserted_id))

            logger.info(f"Stored {len(inserted_ids)} documents with IDs: {inserted_ids}")
            return {
                "story_ids": inserted_ids,
                "extracted_data": extracted_data
            }

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)} - Raw result: {result}")
            raise HTTPException(status_code=500, detail=f"Failed to parse Lyzr API response: {str(e)}")
        except Exception as e:
            logger.error(f"Error extracting from PRD: {str(e)}")
            raise HTTPException(status_code=500, detail=f"PRD extraction failed: {str(e)}")

    async def analyze_stories(self, stories: List[str], agent_id: str, analyze_prompt: str = None, batch_size: int = 5):
        """Analyze user stories to extract structured information."""
        if not analyze_prompt:
            analyze_prompt = """
            Analyze this user story and extract detailed information:

            User Story: "{text}"

            Provide the following in JSON format:
            1. title: Short title for the story
            2. description: Brief description of what the story is about
            3. category: Main category (e.g., Create, Read, Update, Delete, or other functional category)
            4. priority: Estimated priority (high, medium, low)
            5. topics: List of 3-5 relevant topics/keywords
            6. user_role: The role of the user in the story
            7. features: List of key features referenced in the story

            Return ONLY valid JSON with no additional text or formatting.
            """

        logger.info(f"Analyzing {len(stories)} user stories...")
        analysis_results = []
        print("agent_id", agent_id)

        # Process in batches
        for i in range(0, len(stories), batch_size):
            batch = stories[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1} of {(len(stories) + batch_size - 1) // batch_size}")

            for story in batch:
                try:
                    logger.info(f"Story: {story}")
                    result = self._call_lyzr_api(
                        agent_id=agent_id,
                        session_id=agent_id,
                        text=story,
                        prompt=analyze_prompt
                    )
                    logger.info(f"Result from Lyzr agent: {result}")
                    result = json.loads(result)
                    logger.info(f"Parsed result from Lyzr agent: {result}")
                    result['original_text'] = story
                    analysis_results.append(result)
                except Exception as e:
                    logger.error(f"Error analyzing story: {e}")
                    analysis_results.append({
                        "original_text": story,
                        "error": str(e)
                    })

        self.analysis_results = analysis_results
        self._extract_metadata_fields()

        return {
            "analysis_results": analysis_results,
            "categories": self.categories,
            "topics": self.topics,
            "features": self.features
        }

    def _extract_metadata_fields(self):
        """Extract unique categories, topics, and features from analysis results."""
        categories = set()
        topics = set()
        features = set()

        for result in self.analysis_results:
            if 'category' in result:
                categories.add(result['category'])
            if 'topics' in result:
                if isinstance(result['topics'], list):
                    for topic in result['topics']:
                        topics.add(str(topic).lower())
                elif isinstance(result['topics'], str):
                    for topic in result['topics'].split(','):
                        topics.add(topic.strip().lower())
            if 'features' in result:
                if isinstance(result['features'], list):
                    for feature in result['features']:
                        features.add(str(feature).lower())
                elif isinstance(result['features'], str):
                    for feature in result['features'].split(','):
                        features.add(feature.strip().lower())

        self.categories = sorted(list(categories))
        self.topics = sorted(list(topics))
        self.features = sorted(list(features))

    async def search_stories(self, query=None, category=None, priority=None, 
                            topic=None, feature=None, project_id=None, n_results=5):
        """Search for user stories with filters using MongoDB queries."""
        if not self.db:
            logger.error("MongoDB connection not configured")
            return []

        try:
            filter_query = {}
            if project_id:
                filter_query["project_id"] = project_id

            if query:
                await self.db.user_stories.create_index([("text", "text"), 
                                                       ("analysis_result.title", "text"),
                                                       ("analysis_result.description", "text")])
                filter_query["$text"] = {"$search": query}

            analysis_filters = {}
            if category:
                analysis_filters["analysis_result.category"] = category
            if priority:
                analysis_filters["analysis_result.priority"] = priority.lower()

            if topic:
                filter_query["$or"] = [
                    {"analysis_result.topics": {"$in": [topic.lower()]}},
                    {"analysis_result.topics": {"$regex": topic.lower(), "$options": "i"}}
                ]

            if feature:
                feature_filter = {
                    "$or": [
                        {"analysis_result.features": {"$in": [feature.lower()]}},
                        {"analysis_result.features": {"$regex": feature.lower(), "$options": "i"}}
                    ]
                }
                if "$or" in filter_query:
                    filter_query["$and"] = [{"$or": filter_query.pop("$or")}, {"$or": feature_filter["$or"]}]
                else:
                    filter_query["$or"] = feature_filter["$or"]

            filter_query.update(analysis_filters)
            filter_query["analyzed"] = True

            cursor = self.db.user_stories.find(filter_query).limit(n_results)
            results = await cursor.to_list(length=n_results)

            formatted_results = []
            for result in results:
                formatted_result = {
                    "id": str(result["_id"]),
                    "document": result.get("text", ""),
                    "metadata": result.get("analysis_result", {}),
                    "project_id": str(result.get("project_id", ""))
                }
                formatted_results.append(formatted_result)

            return formatted_results

        except Exception as e:
            logger.error(f"Error during MongoDB search: {e}")
            return []

    def get_metadata_options(self):
        """Return all available metadata options for filtering."""
        return {
            "categories": self.categories,
            "topics": self.topics,
            "features": self.features,
            "priorities": ["high", "medium", "low"]
        }