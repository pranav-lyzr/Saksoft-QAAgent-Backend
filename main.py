from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, HttpUrl
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from git import Repo
import tempfile
import shutil
from typing import Optional, List
import json
import datetime
import uuid
import time
import requests
from urllib.parse import urlencode
import base64
from requests.auth import HTTPBasicAuth
from testing.core import RepositoryAnalyzer
from testing.user_story_manager import UserStoryManager
from testing.techninal_insight_manager import TechnicalInsightsManager
from testing.test_plan_generator import TestPlanGenerator
from testing.llm_provider import LLMProvider
from testing.architecture_test_generator import ArchitectureTestGenerator
from testing.api_test_generator import APITestGenerator
from testing.security_test_generator import SecurityTestGenerator
from testing.database_test_generator import DatabaseTestGenerator
from testing.performance_test_generator import PerformanceTestGenerator
from testing.story_test_generator import StoryTestGenerator
from testing.base_test_generator import TestGenerationManager
from fastapi.middleware.cors import CORSMiddleware
from cryptography.fernet import Fernet
import os
import traceback  # Add this line
from dotenv import load_dotenv  # Import python-dotenv

load_dotenv()

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# MongoDB Connection
# MONGO_URI = "mongodb://root:example@mongo:27017"
MONGO_URI = os.getenv("MONGO_URI","mongodb://root:example@mongo:27017")
client = AsyncIOMotorClient(MONGO_URI)
db = client["mydatabase"]
analyzer = RepositoryAnalyzer()


LYZR_RAG_API_URL = "https://rag-prod.studio.lyzr.ai/v3/rag"
LYZR_AGENT_API_URL = "https://agent-prod.studio.lyzr.ai/v3/agent"
LYZR_API_URL = "https://agent-prod.studio.lyzr.ai/v3/inference/chat/"
API_KEY = "sk-default-yStV4gbpjadbQSw4i7QhoOLRwAs5dEcl"
USER_ID = "pranav@lyzr.ai"

class ProjectCreate(BaseModel):
    name: str

class GitHubLink(BaseModel):
    github_url: HttpUrl

class CodeQuery(BaseModel):
    message: str

class AgentQuery(BaseModel):
    message: str
    project_id: str

class UserStoryCreate(BaseModel):
    text: str

class UserStorySearch(BaseModel):
    query: Optional[str] = None
    category: Optional[str] = None
    priority: Optional[str] = None
    topic: Optional[str] = None
    feature: Optional[str] = None
    n_results: int = 5

class TechnicalInsightCreate(BaseModel):
    text: str

class TechnicalInsightAnalysisRequest(BaseModel):
    pass

class PRDInput(BaseModel):
    prd_text: str

class ArchitectureDocInput(BaseModel):
    doc_text: str

class NotionIntegration(BaseModel):
    database_id: str
    api_secret: str

story_manager = UserStoryManager()
test_plan_generator = TestPlanGenerator()


USER_STORY_INSTRUCTIONS = """# User Story Agent Instructions

Your TASK is to ASSIST users in understanding and managing user stories within the context of the provided codebase. You MUST follow these STEPS:

1. **INTERPRET USER STORIES**: Analyze the user story to identify key requirements, features, and acceptance criteria.
2. **MAP TO CODEBASE**: Identify relevant parts of the codebase that relate to the user story, such as existing features or components that might need modification.
3. **SUGGEST IMPLEMENTATIONS**: If the user story requires new functionality, suggest how it can be implemented based on the existing codebase structure and patterns.
4. **IDENTIFY DEPENDENCIES**: Highlight any dependencies or potential impacts on other parts of the system.
5. **PROVIDE EXAMPLES**: Offer code snippets or references to similar implementations in the codebase.
6. **CLARIFY REQUIREMENTS**: If the user story is unclear, ask for clarification or suggest ways to refine it.
7. **LIMIT TO AVAILABLE DATA**: Base all responses solely on the information available in the provided RAG and codebase. Do not make assumptions about unavailable information.
Return ONLY valid JSON with no additional text or formatting.
"""

TECHNICAL_INSIGHT_INSTRUCTIONS = """# Technical Insight Agent Instructions

Your TASK is to PROVIDE technical insights about the codebase, focusing on architecture, performance, security, and best practices. You MUST follow these STEPS:

1. **ANALYZE CODE STRUCTURE**: Examine the overall architecture, design patterns, and modularity of the codebase.
2. **IDENTIFY STRENGTHS AND WEAKNESSES**: Highlight areas where the codebase excels and areas that may need improvement.
3. **PERFORMANCE ANALYSIS**: Identify potential performance bottlenecks or optimizations based on the code.
4. **SECURITY ASSESSMENT**: Point out any security vulnerabilities or best practices that are being followed or missed.
5. **CODE QUALITY**: Comment on code readability, maintainability, and adherence to coding standards.
6. **SUGGEST IMPROVEMENTS**: Provide actionable recommendations for enhancing the codebase.
7. **USE AVAILABLE DATA**: Rely only on the information present in the RAG and codebase. Do not speculate about unprovided details.
Return ONLY valid JSON with no additional text or formatting.
"""

TEST_PLAN_INSTRUCTIONS = """# Test Plan Generator Agent Instructions

Your **TASK** is to **GENERATE** a comprehensive and highly accurate test plan that ensures thorough coverage of the provided user stories, technical insights, and codebase. The test plan must align closely with the actual implementation to maximize effectiveness and achieve high accuracy. Follow these steps meticulously:

"""


# 1️⃣ Create Project
@app.post("/create_project")
async def create_project(project: ProjectCreate):
    project_collection = db.projects
    new_project = {
        "name": project.name,
        "github_links": [],
        "analysis_results": {}
    }
    result = await project_collection.insert_one(new_project)
    return {"project_id": str(result.inserted_id), "message": "Project created successfully"}


@app.get("/projects")
async def list_projects():
    """
    Retrieve a list of all projects with their names and IDs from the database.
    
    Returns:
        A JSON response containing a list of dictionaries, each with 'project_id' and 'name'.
    Raises:
        HTTPException: If there is an error retrieving projects from the database.
    """
    try:
        # Access the projects collection in MongoDB
        project_collection = db.projects
        
        # Query all projects, projecting only the '_id' and 'name' fields
        projects_cursor = project_collection.find({}, {"_id": 1, "name": 1})
        
        # Convert the cursor to a list (limit to 1000 for safety)
        projects = await projects_cursor.to_list(length=1000)
        
        # Format the response as a list of dictionaries
        formatted_projects = [
            {"project_id": str(project["_id"]), "name": project["name"]}
            for project in projects
        ]
        
        # Return the formatted list
        return {"projects": formatted_projects}
    
    except Exception as e:
        # Handle any errors and raise an HTTP exception
        raise HTTPException(status_code=500, detail=f"Error retrieving projects: {str(e)}")



# 2️⃣ Add GitHub Link with RAG Analysis
@app.post("/project/{project_id}/repo")
async def add_github_link(project_id: str, link: GitHubLink):
    try:
        obj_id = ObjectId(project_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    project = await db.projects.find_one({"_id": obj_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    github_url_str = str(link.github_url)

    await db.projects.update_one(
        {"_id": obj_id},
        {"$push": {"github_links": github_url_str}}
    )

    # Ensure direct execution instead of background task
    await analyze_repository_background(obj_id, github_url_str)

    return {"message": "GitHub link added successfully. Analysis completed."}

async def analyze_repository_background(project_id: ObjectId, repo_url: str):
    temp_dir = tempfile.mkdtemp()
    try:
        # Create out directory if it doesn't exist
        out_dir = os.path.join(os.getcwd(), "out")
        os.makedirs(out_dir, exist_ok=True)

        # Fetch project data first to check for existing analysis
        project = await db.projects.find_one({"_id": project_id})
        if not project:
            raise Exception("Project not found")

        # Only analyze if no existing results
        if not project.get("analysis_results"):
            # Clone and analyze repository
            Repo.clone_from(repo_url, temp_dir)
            analysis_result = await analyzer.analyze_repository(temp_dir)
            result_dict = json.loads(json.dumps(analysis_result, default=str))
            new_api_data = result_dict.get("api_data", [])
            new_db_schemas = result_dict.get("db_schemas", [])
            new_ui_data = result_dict.get("ui_data", [])
            new_rag_data = result_dict.get("rag_data", [])
            # Update project with new analysis results
            await db.projects.update_one(
                {"_id": project_id},
                {
                    "$push": {
                        "repo_analyses": {"repo_url": repo_url, "analysis_result": result_dict},
                        "api_specs": {"$each": new_api_data},
                        "db_schemas": {"$each": new_db_schemas},
                        "ui_data": {"$each": new_ui_data},
                        "code_samples": {"$each": new_rag_data}
                    }
                }
            )
        else:
            # Use existing analysis results
            result_dict = project["analysis_results"]

        project_name = project["name"]
        
        # Save analysis results to out folder (optional: add timestamp to prevent overwrite)
        output_dir = os.path.join(out_dir, project_name)
        os.makedirs(output_dir, exist_ok=True)  # Create project-specific directory
        output_file = os.path.join(output_dir, "analysis.json")
        with open(output_file, "w") as f:
            json.dump(result_dict, f, indent=2)

        rag_data = result_dict.get("rag_data", [])
        chunked_documents = []
        for file_info in rag_data:
            file_path = file_info.get("file_path")
            if not file_path:
                continue
            full_path = os.path.join(temp_dir, file_path)
            if os.path.isfile(full_path):
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    # Simple chunking (adjust as needed)
                    text_chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
                    for chunk in text_chunks:
                        chunked_documents.append({
                            "id_": str(uuid.uuid4()),
                            "embedding": None,
                            "metadata": {"source": file_path, "chunked": True},
                            "text": chunk.strip(),
                            "excluded_embed_metadata_keys": [],
                            "excluded_llm_metadata_keys": []
                        })
                except UnicodeDecodeError as e:
                    print(f"Skipping file {full_path} due to decode error: {str(e)}")
                    continue
                except Exception as e:
                    print(f"Skipping file {full_path} due to unexpected error: {str(e)}")
                    continue

        if "rag_id" not in project:
            rag_id = create_rag_collection()
            if not rag_id:
                raise Exception("Failed to create RAG collection")

            if chunked_documents:
                train_rag(rag_id, chunked_documents)

            # Create all five agents
            user_story_agent = create_agent(rag_id, "user_story", USER_STORY_INSTRUCTIONS, project_name)
            tech_insight_agent = create_agent(rag_id, "technical_insight", TECHNICAL_INSIGHT_INSTRUCTIONS, project_name)
            test_plan_agent = create_agent(rag_id, "test_plan", TEST_PLAN_INSTRUCTIONS, project_name)

            # Verify agent creation
            if not all([user_story_agent, tech_insight_agent, test_plan_agent]):
                raise Exception("Failed to create one or more agents")

            # Update project with RAG and agent IDs
            update_data = {
                "rag_id": rag_id,
                "user_story_agent_id": user_story_agent.get("agent_id"),
                "technical_insight_agent_id": tech_insight_agent.get("agent_id"),
                "test_plan_agent_id": test_plan_agent.get("agent_id")
            }
            await db.projects.update_one(
                {"_id": project_id},
                {"$set": update_data}
            )
        else:
            # Retrain existing RAG with new repository data
            rag_id = project["rag_id"]
            if chunked_documents:
                train_rag(rag_id, chunked_documents)

    except Exception as e:
        print(f"Error analyzing repository: {str(e)}")
        raise  # Re-raise to ensure caller is aware of failure
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def create_rag_collection():
    try:
        print("Inside Create Rag Collection Function")
        response = requests.post(
            f"{LYZR_RAG_API_URL}/",
            headers={"x-api-key": API_KEY},
            json={
                "user_id": USER_ID,
                "llm_credential_id": "lyzr_openai",
                "embedding_credential_id": "lyzr_openai",
                "vector_db_credential_id": "lyzr_weaviate",
                "vector_store_provider": "Weaviate [Lyzr]",
                "description": "Repository analysis RAG",
                "collection_name": f"repo_rag_{int(time.time())}",
                "llm_model": "gpt-4o-mini",
                "embedding_model": "text-embedding-ada-002"
            }
        )
        print("Rag creating ",response.json())
        return response.json().get('id')
    except Exception as e:
        print(f"RAG creation failed: {str(e)}")
        return None
    
def train_rag(rag_id, documents):
    try:
        print("TRAIN RAG");

        # response = requests.post(
        #     f"{LYZR_RAG_API_URL}/train/{rag_id}/"
        # )
        print('document',documents);
        response = requests.post(
            f"{LYZR_RAG_API_URL}/train/{rag_id}/",
            headers={"x-api-key": API_KEY},
            json=documents
        )

        print("RESPONSE FOR TRANING ")
    
        return True
    except Exception as e:
        print(f"RAG training failed: {str(e)}")
        return False


def create_agent(rag_id, agent_type, instructions, project_name):
    try:
        url = "https://agent-prod.studio.lyzr.ai/v3/agents/template/single-task"  # Correct API URL
        headers = {
            "x-api-key": API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "name": f"repo_{project_name}_{agent_type}_agent",
            "description": f"QA agent Project Name: {project_name}",
            "agent_instructions": instructions,  
            "agent_role": f"Agent for code {agent_type}",
            "llm_credential_id": "lyzr_openai",
            "provider_id": "OpenAI",
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "top_p": 0.9,
            "features": [
                {
                    "type": "KNOWLEDGE_BASE",  # Ensure this is a valid type in API docs
                    "config": {
                        "lyzr_rag": {
                            "base_url": "https://rag-prod.studio.lyzr.ai",
                            "rag_id": rag_id,
                            "rag_name": "SakSoft Code Rag"
                        }
                    },
                    "priority": 0
                }
            ],
            "tools": []  # Fixed: should be an array, not `None`
        }
        
        response = requests.post(url, headers=headers, json=payload)
        print("Creating agent response:", payload, response.status_code, response.text)

        if response.status_code == 405:
            print("⚠️ Method Not Allowed: Check if POST is the correct method.")
        elif response.status_code == 403:
            print("❌ Forbidden: Check if your API key is correct.")
        
        return response.json() if response.status_code == 200 else None
    
    except Exception as e:
        print(f"⚠️ Agent creation failed: {str(e)}")
        return None

    
# 5️⃣ Update Project (PUT)
@app.put("/project/{project_id}")
async def update_project(project_id: str, project: ProjectCreate):
    obj_id = ObjectId(project_id)
    result = await db.projects.update_one({"_id": obj_id}, {"$set": {"name": project.name}})
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Project not found or no changes made")
    return {"message": "Project updated successfully"}

# 6️⃣ Get Project Details (GET)
@app.get("/project/{project_id}")
async def get_project(project_id: str):
    try:
        # Convert the string project_id to ObjectId for MongoDB query
        obj_id = ObjectId(project_id)
        project = await db.projects.find_one({"_id": obj_id})
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Convert the _id field (ObjectId) to a string
        project["_id"] = str(project["_id"])
        
        # Handle any other potential ObjectId fields in the document
        for key, value in project.items():
            if isinstance(value, ObjectId):
                project[key] = str(value)
        
        return project
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid project ID format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving project: {str(e)}")


# 7️⃣ Delete Project (DELETE)
@app.delete("/project/{project_id}")
async def delete_project(project_id: str):
    result = await db.projects.delete_one({"_id": ObjectId(project_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"message": "Project deleted successfully"}

# This should be called after app initialization when db is available
@app.on_event("startup")
async def configure_story_manager():
    story_manager.set_db(db)
    # Setup OpenAI - in production, use environment variables for the API key
    # openai_api_key = os.getenv("OPENAI_API_KEY")
    # if openai_api_key:
    #     story_manager.setup_openai(openai_api_key)

# Add the endpoint to create a user story
@app.post("/project/{project_id}/user_story")
async def create_user_story(project_id: str, story: UserStoryCreate):
    try:
        # Validate project exists
        obj_id = ObjectId(project_id)
        project = await db.projects.find_one({"_id": obj_id})
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Create a new user story document
        story_doc = {
            "project_id": obj_id,
            "text": story.text,
            "analyzed": False,
            "analysis_result": None,
            "created_at": time.time()
        }
        
        # Insert the story into MongoDB
        user_story_collection = db.user_stories
        result = await user_story_collection.insert_one(story_doc)
        
        # Return the created story ID
        return {"story_id": str(result.inserted_id), "message": "User story created successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating user story: {str(e)}")

# Endpoint to analyze user stories for a project
@app.post("/project/{project_id}/analyze_stories")
async def analyze_user_stories(project_id: str, background_tasks: BackgroundTasks):
    try:
        # Verify the project exists
        obj_id = ObjectId(project_id)
        project = await db.projects.find_one({"_id": obj_id})
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Queue the analysis task in the background
        background_tasks.add_task(analyze_stories_background, obj_id)
        
        return {"message": "Story analysed"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting analysis: {str(e)}")

async def analyze_stories_background(project_id: ObjectId):
    try:
        # Get all unanalyzed stories for this project
        user_story_collection = db.user_stories
        project = await db.projects.find_one({"_id": project_id})
        print("Project",project, project_id)
        if not project or "user_story_agent_id" not in project:
            raise Exception("User story agent not configured for this project")
        agent_id = project["user_story_agent_id"]

        stories_cursor = user_story_collection.find({
            "project_id": project_id,
            "analyzed": False
        })
        
        stories = await stories_cursor.to_list(length=100)
        
        if not stories:
            print(f"No unanalyzed stories found for project {project_id}")
            return
        
        # Extract the story texts
        story_texts = [story["text"] for story in stories]
        
        # Analyze the stories
        analysis_results = await story_manager.analyze_stories(story_texts, agent_id=agent_id)
        
        # Update each story with its analysis result
        for i, story in enumerate(stories):
            if i < len(analysis_results["analysis_results"]):
                # Update the story with analysis results
                await user_story_collection.update_one(
                    {"_id": story["_id"]},
                    {"$set": {
                        "analyzed": True,
                        "analysis_result": analysis_results["analysis_results"][i]
                    }}
                )
        
        # Update project with metadata
        await db.projects.update_one(
            {"_id": project_id},
            {"$set": {
                "story_categories": analysis_results["categories"],
                "story_topics": analysis_results["topics"],
                "story_features": analysis_results["features"]
            }}
        )
        
        print(f"Successfully analyzed {len(stories)} stories for project {project_id}")
    
    except Exception as e:
        print(f"Error analyzing stories: {str(e)}")

async def get_db():
    try:
        yield db
    finally:
        pass

def get_user_story_manager():
    manager = UserStoryManager()
    return manager

@app.post("/project/{project_id}/process_prd")
async def process_prd(project_id: str, prd_input: PRDInput):
    """Process a PRD document, extract user stories, and analyze them using the project's user story agent."""
    # Validate project exists and fetch user_story_agent_id
    project = await db.projects.find_one({"_id": ObjectId(project_id)})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    user_story_agent_id = project.get("user_story_agent_id")
    if not user_story_agent_id:
        raise HTTPException(status_code=500, detail="User story agent not configured for this project")

    # Use the global story_manager (db set at startup)
    extraction_result = await story_manager.extract_from_prd(prd_input.prd_text, project_id)
    print(f"Extraction result: {extraction_result}")
    
    # Ensure extraction_result has the expected keys
    if not isinstance(extraction_result, dict) or "story_ids" not in extraction_result:
        raise HTTPException(status_code=500, detail="Invalid response from extract_from_prd")

    story_ids = extraction_result["story_ids"]
    extracted_data = extraction_result["extracted_data"]
    user_stories = extracted_data.get("user_stories", [])

    # Analyze extracted user stories using the project's user_story_agent_id
    if user_stories:
        analysis_result = await story_manager.analyze_stories(
            stories=user_stories,
            agent_id=user_story_agent_id
        )
        
        # Update the stored user stories with analysis results
        for i, story in enumerate(analysis_result["analysis_results"]):
            if i < len(story_ids):
                await db.user_stories.update_one(
                    {"_id": ObjectId(story_ids[i])},
                    {"$set": {
                        "analysis_result": {k: v for k, v in story.items() if k != "original_text"},
                        "analyzed": True
                    }}
                )
    else:
        analysis_result = {"message": "No user stories extracted from PRD to analyze"}

    return {
        "story_ids": story_ids,
        "extracted_data": extracted_data,
        "analysis_result": analysis_result if user_stories else None
    }


# Add to Pydantic models
class ArchitectureDocInput(BaseModel):
    doc_text: str

# Add new endpoint
@app.post("/project/{project_id}/process_arch_doc")
async def process_architecture_doc(project_id: str, doc_input: ArchitectureDocInput):
    """Process architecture documents (TDD, HLD) to extract patterns, tools and components"""
    ARCH_AGENT_ID = "67e6e0baec01d3b71add7cff"  # Architecture analysis agent ID
    
    # Validate project exists
    project = await db.projects.find_one({"_id": ObjectId(project_id)})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Define extraction prompt
    arch_extraction_prompt = """Analyze this technical architecture document and extract key elements:

    Document Text: "{text}"

    Extract and structure the following in JSON format:
    1. architecture_patterns: List of architectural patterns used (e.g., microservices, event-driven)
    2. tools: Categorized tools/technologies used:
       - cloud_platforms
       - containerization
       - orchestration
       - monitoring
       - ci_cd
    3. components: List of system components/modules
    4. interfaces: Key system interfaces/APIs
    5. quality_attributes: Key quality attributes (scalability, reliability, etc.)

    Example Output:
    {{
        "architecture_patterns": ["microservices", "event-driven"],
        "tools": {{
            "cloud_platforms": ["AWS"],
            "containerization": ["Docker"],
            "orchestration": ["Kubernetes"],
            "monitoring": ["Prometheus"],
            "ci_cd": ["Jenkins"]
        }},
        "components": ["User Service", "Order Processor"],
        "interfaces": ["REST API", "gRPC"],
        "quality_attributes": ["high-availability", "horizontal-scalability"]
    }}

    Return ONLY valid JSON with no additional text.
    """

    try:
        # Call Lyzr architecture analysis agent
        result = requests.post(
            LYZR_API_URL,
            headers={
                "Content-Type": "application/json",
                "x-api-key": API_KEY
            },
            json={
                "user_id": USER_ID,
                "agent_id": ARCH_AGENT_ID,
                "session_id": f"arch_{project_id}",
                "message": arch_extraction_prompt.format(text=doc_input.doc_text)
            }
        ).json()

        # Clean and parse response
        cleaned_result = result.get("response", "").replace('```json', '').replace('```', '').strip()
        extracted_data = json.loads(cleaned_result)

        # Store in architecture_profiles collection
        arch_profile = {
            "project_id": ObjectId(project_id),
            "extracted_data": extracted_data,
            "source_doc": doc_input.doc_text[:1000] + "..." if len(doc_input.doc_text) > 1000 else doc_input.doc_text,
            "created_at": datetime.datetime.utcnow()
        }
        
        # Insert into architecture_profiles collection
        insert_result = await db.architecture_profiles.insert_one(arch_profile)
        
        # Update project with extracted data
        update_data = {
                "$addToSet": {
                    "architecture.patterns": {"$each": extracted_data.get("architecture_patterns", [])},
                    "architecture.components": {"$each": extracted_data.get("components", [])},
                    "architecture.interfaces": {"$each": extracted_data.get("interfaces", [])},
                    "architecture.quality_attributes": {"$each": extracted_data.get("quality_attributes", [])}
                },
                "$set": {
                    "architecture.tools": {
                        # Merge existing tools with new ones
                        **project.get("architecture", {}).get("tools", {}),
                        **extracted_data.get("tools", {})
                    }
                }
            }

        await db.projects.update_one(
            {"_id": ObjectId(project_id)},
            update_data,
            upsert=True  # Create structure if it doesn't exist
        )

        return {
            "profile_id": str(insert_result.inserted_id),
            "extracted_data": extracted_data,
            "message": "Architecture profile created successfully"
        }

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {str(e)} - Raw result: {result}")
        raise HTTPException(status_code=500, detail="Failed to parse architecture document")
    except Exception as e:
        print(f"Error processing architecture doc: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Architecture processing failed: {str(e)}")
    



@app.post("/project/{project_id}/technical_insight")
async def create_technical_insight(project_id: str, insight: TechnicalInsightCreate):
    try:
        # Validate project exists
        obj_id = ObjectId(project_id)
        project = await db.projects.find_one({"_id": obj_id})
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Create new technical insight document
        insight_doc = {
            "project_id": obj_id,
            "text": insight.text,
            "analyzed": False,
            "analysis_result": None,
            "created_at": time.time()
        }
        
        # Insert the insight into MongoDB
        tech_insight_collection = db.technical_insights
        result = await tech_insight_collection.insert_one(insight_doc)
        
        return {"insight_id": str(result.inserted_id), "message": "Technical insight created successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating technical insight: {str(e)}")

@app.post("/project/{project_id}/analyze_insights")
async def analyze_technical_insights(project_id: str, background_tasks: BackgroundTasks):
    try:
        # Verify the project exists
        obj_id = ObjectId(project_id)
        project = await db.projects.find_one({"_id": obj_id})
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Queue the analysis task in the background (only project_id is passed)
        background_tasks.add_task(process_technical_insights_background, project_id)
        
        return {"message": "Technical insights analysed"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting analysis: {str(e)}")

async def process_technical_insights_background(project_id: str):
    try:
        obj_id = ObjectId(project_id)
        print(f"Starting analysis for technical insights for project {obj_id}")
        project = await db.projects.find_one({"_id": obj_id})
        print("Project",project, project_id)
        if not project or "technical_insight_agent_id" not in project:
            raise Exception("User story agent not configured for this project")
        agent_id = project["technical_insight_agent_id"]


        tech_insights_collection = db.technical_insights  # Motor collection
        
        # Retrieve un-analyzed technical insights for this project
        cursor = tech_insights_collection.find({
            "project_id": ObjectId(project_id),
            "analyzed": False
        })
        docs = await cursor.to_list(length=100)
        
        if not docs:
            print(f"No unanalyzed technical insights found for project {project_id}")
            return
        
        # Extract texts from the documents
        insights_texts = [doc["text"] for doc in docs]
        print(f"Retrieved {len(insights_texts)} technical insights for analysis")
        
        # Call the analysis function (synchronous using requests)
        manager = TechnicalInsightsManager()
        print("Calling analyze_insights on TechnicalInsightsManager")
        analysis = manager.analyze_insights(insights_texts, agent_id = agent_id)
        print("Analysis completed. Processing analysis results...")
        
        total_results = len(analysis.get("analysis_results", []))
        print(f"Found {total_results} analysis results.")
        
        # Process each analysis result and update the corresponding document
        for idx, analysis_result in enumerate(analysis["analysis_results"], start=1):
            print(f"Processing analysis result {idx}/{total_results}")
            if "error" in analysis_result:
                print(f"Skipping analysis result {idx} due to error: {analysis_result.get('error')}")
                continue
            
            # Assuming the order of analysis results corresponds to the order of documents retrieved.
            doc = docs[idx - 1]
            update_doc = {
                "analyzed": True,
                "analysis_result": analysis_result
            }
            
            result = await tech_insights_collection.update_one(
                {"_id": doc["_id"]},
                {"$set": update_doc}
            )
            print(f"Updated technical insight {idx}/{total_results} with update result: {result.modified_count}")
        
        print(f"Successfully processed technical insights for project {project_id}")
    
    except Exception as e:
        print(f"Error processing technical insights for project {project_id}: {str(e)}")

# Endpoint to search user stories
@app.post("/project/{project_id}/search_stories")
async def search_user_stories(project_id: str, search: UserStorySearch):
    try:
        # Verify the project exists
        obj_id = ObjectId(project_id)
        project = await db.projects.find_one({"_id": obj_id})
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Perform the search using UserStoryManager's MongoDB search
        results = await story_manager.search_stories(
            query=search.query,
            category=search.category,
            priority=search.priority,
            topic=search.topic,
            feature=search.feature,
            project_id=obj_id,
            n_results=search.n_results
        )
        
        # Return the formatted results
        return {"results": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching stories: {str(e)}")


from bson import ObjectId


from bson import ObjectId

@app.get("/generate-test-plan")
async def generate_test_plan(project_id: str):
    """
    Fetches all relevant data for a project from MongoDB, generates a test plan,
    and saves it with a timestamp in the test_plans collection.
    """
    print(f"Starting generate_test_plan for project_id: {project_id}")
    obj_id = ObjectId(project_id)
    project = await db.projects.find_one({"_id": obj_id})
    if not project or "test_plan_agent_id" not in project:
        raise Exception("User story agent not configured for this project")
    agent_id = project["test_plan_agent_id"]
    # Convert project_id to ObjectId
    try:
        project_obj_id = ObjectId(project_id)
        print(f"Converted project_id to ObjectId: {project_obj_id}")
    except Exception as e:
        print(f"Error converting project_id: {e}")
        return {"error": "Invalid project ID format"}

    # 1. Fetch project data
    try:
        project_data = await db.projects.find_one({"_id": project_obj_id})
        if not project_data:
            print(f"Project not found for _id: {project_obj_id}")
            return {"error": "Project not found"}
        print(f"Project data retrieved: {project_data}")
    except Exception as e:
        print(f"Error fetching project data: {e}")
        return {"error": "Error fetching project data"}

    # 2. Fetch analysis results from the project document
    analysis_results = project_data.get("analysis_results", {})
    print(f"Analysis results: {analysis_results}")
    api_data = analysis_results.get("api_data", {})
    db_data = analysis_results.get("db_schemas", {})
    ui_data = analysis_results.get("ui_data", {})
    rag_data = analysis_results.get("rag_data", {})

    # 3. Fetch all user stories for the project
    try:
        user_stories_cursor = db.user_stories.find({"project_id": project_obj_id})
        user_stories_list = await user_stories_cursor.to_list(length=100)
        print(f"Found {len(user_stories_list)} user stories for project {project_obj_id}")
    except Exception as e:
        print(f"Error fetching user stories: {e}")
        return {"error": "Error fetching user stories"}

    # 4. Fetch all technical insights for the project
    try:
        tech_insights_cursor = db.technical_insights.find({
            "project_id": project_obj_id,
            "analyzed": True
        })
        tech_insights_list = await tech_insights_cursor.to_list(length=100)
        print(f"Found {len(tech_insights_list)} technical insights for project {project_obj_id}")
    except Exception as e:
        print(f"Error fetching technical insights: {e}")
        tech_insights_list = []

    # 5. Generate test plan
    try:
        print("Generating test plan with gathered data...")
        test_plan = test_plan_generator.generate_test_plan(
            user_story=user_stories_list,
            api_data=api_data,
            db_data=db_data,
            ui_data=ui_data,
            rag_Data=rag_data,
            technical_insights=tech_insights_list,
            agent_id = agent_id
        )
        print("Test plan generated successfully.")
    except Exception as e:
        print(f"Error generating test plan: {e}")
        return {"error": "Error generating test plan"}

    # 6. Save the test plan with a timestamp
    try:
        test_plan_doc = {
            "project_id": project_obj_id,
            "test_plan": test_plan,
            "timestamp": datetime.datetime.utcnow()
        }
        result = await db.test_plans.insert_one(test_plan_doc)
        print(f"Test plan saved with ID: {result.inserted_id}")
    except Exception as e:
        print(f"Error saving test plan: {e}")
        return {"error": "Error saving test plan"}

    print("Returning test plan response")
    return {
        "test_plan": test_plan,
        "test_plan_id": str(result.inserted_id),
        "message": "Test plan generated and saved successfully"
    }



# Endpoint to get all user stories for a project
@app.get("/project/{project_id}/user_stories")
async def get_project_user_stories(project_id: str):
    try:
        # Verify the project exists
        obj_id = ObjectId(project_id)
        project = await db.projects.find_one({"_id": obj_id})
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Get all stories for this project
        user_story_collection = db.user_stories
        stories_cursor = user_story_collection.find({"project_id": obj_id})
        stories = await stories_cursor.to_list(length=100)
        
        # Format the stories for the API response
        formatted_stories = []
        for story in stories:
            formatted_story = {
                "id": str(story["_id"]),
                "project_id": str(story["project_id"]),
                "text": story["text"],
                "analyzed": story["analyzed"],
                "analysis_result": story.get("analysis_result"),
                "created_at": story.get("created_at")
            }
            formatted_stories.append(formatted_story)
        
        return {"stories": formatted_stories}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving stories: {str(e)}")

# Endpoint to get metadata options for filtering
@app.get("/project/{project_id}/user_stories/metadata")
async def get_user_story_metadata(project_id: str):
    try:
        # Verify the project exists
        obj_id = ObjectId(project_id)
        project = await db.projects.find_one({"_id": obj_id})
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Return metadata from the project document
        return {
            "categories": project.get("story_categories", []),
            "topics": project.get("story_topics", []),
            "features": project.get("story_features", []),
            "priorities": ["high", "medium", "low"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving metadata: {str(e)}")
    

@app.get("/project/{project_id}/technical_insights")
async def get_technical_insights(project_id: str):
    try:
        obj_id = ObjectId(project_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid project ID")

    # Assuming 'db' is your MongoDB client connection
    tech_insights = await db.technical_insights.find({"project_id": obj_id}).to_list(1000)
    
    # Convert ObjectId fields to strings
    for insight in tech_insights:
        insight["_id"] = str(insight["_id"])
        if "project_id" in insight:
            insight["project_id"] = str(insight["project_id"])
    
    return {"insights": tech_insights}


@app.get("/project/{project_id}/test-plans")
async def get_test_plans(project_id: str):
    """
    Fetches all test plans for a given project, sorted by timestamp in descending order.
    """
    print(f"Fetching test plans for project_id: {project_id}")

    # Convert project_id to ObjectId
    try:
        project_obj_id = ObjectId(project_id)
        print(f"Converted project_id to ObjectId: {project_obj_id}")
    except Exception as e:
        print(f"Error converting project_id: {e}")
        return {"error": "Invalid project ID format"}

    # Fetch all test plans, sorted by timestamp (latest first)
    try:
        test_plans_cursor = db.test_plans.find({"project_id": project_obj_id}).sort("timestamp", -1)
        test_plans_list = await test_plans_cursor.to_list(length=100)
        print(f"Found {len(test_plans_list)} test plans for project {project_obj_id}")
    except Exception as e:
        print(f"Error fetching test plans: {e}")
        return {"error": "Error fetching test plans"}

    # Convert ObjectId and timestamp for JSON compatibility
    for test_plan in test_plans_list:
        test_plan["_id"] = str(test_plan["_id"])
        test_plan["project_id"] = str(test_plan["project_id"])
        test_plan["timestamp"] = test_plan["timestamp"].isoformat()

    print("Returning test plans response")
    return {"test_plans": test_plans_list}


@app.post("/project/{project_id}/generate_tests")
async def generate_tests(project_id: str, test_type: str):
    # Fetch project from MongoDB
    project = await db.projects.find_one({"_id": ObjectId(project_id)})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    agent_id = project.get("test_plan_agent_id")
    if not agent_id:
        raise HTTPException(status_code=500, detail="Test plan agent ID not configured for this project")

    input_data = {}

    if test_type == "api":
        # Directly fetch API specs from the project
        api_specs = project.get("api_specs", [])
        all_endpoints = []
        auth_types = set()

        # Extract endpoints and security schemes from api_specs
        for api_entry in api_specs:
            endpoints = api_entry.get("endpoints", [])
            all_endpoints.extend(endpoints)
            security_schemes = api_entry.get("security", [])
            if security_schemes:
                auth_types.update(security_schemes)

        # Determine authentication type
        if not auth_types:
            auth_type = "None"
        elif "JWT" in auth_types or "passport" in str(project).lower():
            auth_type = "JWT"
        else:
            auth_type = ", ".join(auth_types)

        # Populate input_data for API tests
        input_data["api"] = {
            "endpoints": all_endpoints,
            "auth_type": auth_type
        }

        # Directly fetch code samples
        code_samples_list = project.get("code_samples", [])

        code_samples_dict = {}
        for item in code_samples_list:
            if "file_path" in item and "rag_metadata" in item:
                # Try to extract code content from different metadata patterns
                content = None
                
                # Pattern 1: Direct code structure description
                if "Code Structure" in item["rag_metadata"]:
                    content = str(item["rag_metadata"]["Code Structure"])
                    
                # Pattern 2: Core analysis sections
                elif "Core Analysis Areas" in item["rag_metadata"]:
                    content = str(item["rag_metadata"]["Core Analysis Areas"])
                    
                # Pattern 3: Code analysis metadata
                elif "Code Analysis" in item["rag_metadata"]:
                    content = str(item["rag_metadata"]["Code Analysis"])
                
                # If content found, add to dictionary
                if content:
                    code_samples_dict[item["file_path"]] = content

        print("code_samples_dict", code_samples_dict)
        input_data["code_samples"] = code_samples_dict

        generator = APITestGenerator()

    elif test_type == "database":
        # Set database type (assuming MongoDB based on context)
        input_data["architecture"] = project.get("db_schemas", [])

        # Directly fetch database schemas
        input_data["database_schema"] = project.get("db_schemas", [])

        # Directly fetch code samples
        code_samples_list = project.get("code_samples", [])

        code_samples_dict = {}
        for item in code_samples_list:
            if "file_path" in item and "rag_metadata" in item:
                # Try to extract code content from different metadata patterns
                content = None
                
                # Pattern 1: Direct code structure description
                if "Code Structure" in item["rag_metadata"]:
                    content = str(item["rag_metadata"]["Code Structure"])
                    
                # Pattern 2: Core analysis sections
                elif "Core Analysis Areas" in item["rag_metadata"]:
                    content = str(item["rag_metadata"]["Core Analysis Areas"])
                    
                # Pattern 3: Code analysis metadata
                elif "Code Analysis" in item["rag_metadata"]:
                    content = str(item["rag_metadata"]["Code Analysis"])
                
                # If content found, add to dictionary
                if content:
                    code_samples_dict[item["file_path"]] = content

        print("code_samples_dict", code_samples_dict)
        input_data["code_samples"] = code_samples_dict

        generator = DatabaseTestGenerator()

    elif test_type == "security":
        # Directly fetch code samples, API specs, and database schemas
        code_samples_list = project.get("code_samples", [])
        print("code_samples_list",code_samples_list)
        # If content is within 'rag_metadata' under 'content'
        code_samples_dict = {}
        for item in code_samples_list:
            if "file_path" in item and "rag_metadata" in item:
                # Try to extract code content from different metadata patterns
                content = None
                
                # Pattern 1: Direct code structure description
                if "Code Structure" in item["rag_metadata"]:
                    content = str(item["rag_metadata"]["Code Structure"])
                    
                # Pattern 2: Core analysis sections
                elif "Core Analysis Areas" in item["rag_metadata"]:
                    content = str(item["rag_metadata"]["Core Analysis Areas"])
                    
                # Pattern 3: Code analysis metadata
                elif "Code Analysis" in item["rag_metadata"]:
                    content = str(item["rag_metadata"]["Code Analysis"])
                
                # If content found, add to dictionary
                if content:
                    code_samples_dict[item["file_path"]] = content

        print("code_samples_dict", code_samples_dict)
        input_data["code_samples"] = code_samples_dict
        input_data["api"] = project.get("api_specs", [])
        input_data["architecture"] = {"database": {"type": "MongoDB"}}
        input_data["database_schema"] = project.get("db_schemas", [])

        generator = SecurityTestGenerator()

    elif test_type == "story":
        user_stories = await db.user_stories.find({"project_id": ObjectId(project_id)}).to_list(None)
        print("user_Stories",user_stories)
        input_data["user_stories"] = [us["analysis_result"] for us in user_stories if us.get("analysis_result")]
        input_data["features"] = project.get("features", [])
        input_data["requirements"] = project.get("requirements", {})
        input_data["app_type"] = project.get("app_type", "unknown")
        generator = StoryTestGenerator()

    elif test_type == "architecture":
        # Now uses the updated project data
        input_data["architecture"] = project.get("architecture", {})
        input_data["patterns"] = project.get("architecture", {}).get("patterns", [])
        input_data["tools"] = project.get("architecture", {}).get("tools", {})
        
        # Fetch technical insights with error handling
        try:
            tech_insights_cursor = db.technical_insights.find({
                "project_id": ObjectId(project_id),
                "analyzed": True
            })
            tech_insights_list = await tech_insights_cursor.to_list(length=100)
            
            # Process technical insights
            technical_insights = [
                {
                    "text": insight["text"],
                    "analysis_result": insight.get("analysis_result", {})
                } for insight in tech_insights_list
            ]
            input_data["technical_insights"] = technical_insights
        except Exception as e:
            # Log the error and proceed with empty insights
            print(f"Error fetching technical insights: {str(e)}")
            input_data["technical_insights"] = []
        
        generator = ArchitectureTestGenerator()

    else:
        raise HTTPException(status_code=400, detail="Unsupported test type")

    # Generate tests using the configured generator
    try:
        tests = generator.generate_tests(input_data, agent_id)
        return {"tests": tests}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating tests: {str(e)}")



from notion_client import Client

class NotionIntegration(BaseModel):
    database_id: str
    internal_token: str

# -------------------------------------------------------------------
# 1. Connect Integration
# -------------------------------------------------------------------
@app.post("/project/{project_id}/notion/connect")
async def connect_notion(
    project_id: str,
    integration: NotionIntegration
):
    try:
        obj_id = ObjectId(project_id)
        project = await db.projects.find_one({"_id": obj_id})
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Validate Notion credentials by retrieving the database
        notion = Client(auth=integration.internal_token)
        try:
            notion.databases.retrieve(database_id=integration.database_id)
        except Exception as e:
            print(f"Notion API error: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail="Invalid Notion credentials or database ID"
            )

        # Store integration details
        await db.projects.update_one(
            {"_id": obj_id},
            {"$set": {
                "notion_integration": {
                    "database_id": integration.database_id,
                    "internal_token": integration.internal_token,
                    "connected_at": datetime.datetime.utcnow()
                }
            }}
        )

        return {"message": "Notion integration connected successfully"}
    
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Integration setup failed: {str(e)}"
        )


# -------------------------------------------------------------------
# 2. Fetch Pages (ID & Title Only)
# -------------------------------------------------------------------
@app.get("/project/{project_id}/notion/pages")
async def get_notion_pages(project_id: str):
    try:
        # Retrieve project and integration details
        obj_id = ObjectId(project_id)
        project = await db.projects.find_one({"_id": obj_id})
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        integration = project.get("notion_integration")
        if not integration:
            raise HTTPException(status_code=400, detail="Notion integration not configured")

        # Initialize Notion client and query the database
        notion = Client(auth=integration["internal_token"])
        query_response = notion.databases.query(database_id=integration["database_id"])
        pages = query_response.get("results", [])

        # Process pages to extract id and title (assuming the title property is named "Name")
        processed_pages = []
        for page in pages:
            page_id = page["id"]
            title = ""
            title_prop = page["properties"].get("Name")
            if title_prop and title_prop["type"] == "title" and title_prop["title"]:
                # Extract the first title block's plain text
                title = title_prop["title"][0].get("plain_text", "")
            processed_pages.append({"id": page_id, "title": title})

        return {"pages": processed_pages}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch pages: {str(e)}")


# -------------------------------------------------------------------
# 3. Get Page Content by Page ID
# -------------------------------------------------------------------
@app.get("/project/{project_id}/notion/pages/{page_id}/content")
async def get_notion_page_content(project_id: str, page_id: str):
    try:
        # Retrieve project and integration details
        obj_id = ObjectId(project_id)
        project = await db.projects.find_one({"_id": obj_id})
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        integration = project.get("notion_integration")
        if not integration:
            raise HTTPException(status_code=400, detail="Notion integration not configured")

        # Initialize Notion client and get the page content (blocks)
        notion = Client(auth=integration["internal_token"])
        blocks_response = notion.blocks.children.list(block_id=page_id)
        blocks = blocks_response.get("results", [])

        return {"page_id": page_id, "content": blocks}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch page content: {str(e)}")


NOTION_ANALYSIS_PROMPT = """Analyze the following Notion page content and extract user stories and technical insights:

Content: "{content}"

Your TASK is to:
1. **Extract User Stories**: Identify text that represents user stories, including requirements, features, or acceptance criteria. Format each as a distinct user story.
2. **Extract Technical Insights**: Identify text related to architecture, performance, security, code quality, or best practices. Format each as a technical insight.
3. **Structure Output**: Return the results in JSON with two keys: "user_stories" and "technical_insights", each containing a list of extracted items.

Guidelines:
- A **user story** typically follows the format "As a [user], I want [feature] so that [benefit]" or describes a feature/requirement.
- A **technical insight** includes observations about code structure, performance bottlenecks, security concerns, or improvement suggestions.
- If no clear user stories or technical insights are found, return empty lists.
- Use the context of software development to interpret the content.

Example Output:
{
    "user_stories": [
        "As a user, I want to log in with my email so that I can access my account securely.",
        "As an admin, I want to view user activity logs so that I can monitor system usage."
    ],
    "technical_insights": [
        "The authentication system uses JWT, which is secure but could benefit from refresh tokens.",
        "Database queries in the user module may cause performance issues due to missing indexes."
    ]
}

Return ONLY valid JSON with no additional text.
"""

@app.post("/project/{project_id}/notion/pages/{page_id}/analyze")
async def analyze_notion_page_content(project_id: str, page_id: str, background_tasks: BackgroundTasks):
    """
    Analyzes Notion page content to extract user stories and technical insights, analyzes them with Lyzr agents,
    trains the RAG, saves them to MongoDB, and returns the results.
    """
    try:
        # Step 1: Validate project and fetch integration details
        obj_id = ObjectId(project_id)
        project = await db.projects.find_one({"_id": obj_id})
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        print("DEBUG 1: Project validated")

        integration = project.get("notion_integration")
        if not integration:
            raise HTTPException(status_code=400, detail="Notion integration not configured")
        print("DEBUG 2: Notion integration found")

        # Fetch agent IDs
        user_story_agent_id = project.get("user_story_agent_id")
        tech_insight_agent_id = project.get("technical_insight_agent_id")
        rag_id = project.get("rag_id")
        if not all([user_story_agent_id, tech_insight_agent_id, rag_id]):
            raise HTTPException(status_code=500, detail="Required agent IDs or RAG ID not configured")
        print(f"DEBUG 3: Agent IDs - User Story: {user_story_agent_id}, Tech Insight: {tech_insight_agent_id}, RAG: {rag_id}")

        # Step 2: Fetch Notion page content
        notion = Client(auth=integration["internal_token"])
        blocks_response = notion.blocks.children.list(block_id=page_id)
        blocks = blocks_response.get("results", [])
        print(f"DEBUG 4: Fetched {len(blocks)} blocks from Notion")

        # Extract text content from blocks
        content = ""
        for block in blocks:
            block_type = block.get("type")
            if block_type in ["paragraph", "heading_1", "heading_2", "heading_3", "bulleted_list_item", "numbered_list_item"]:
                rich_text = block[block_type].get("rich_text", [])
                for text in rich_text:
                    content += text.get("plain_text", "") + "\n"
        if not content:
            return {"message": "No analyzable content found in the page"}
        print(f"DEBUG 5: Extracted content length: {len(content)} characters")
        print(f"DEBUG 5.1: Raw content: {content[:500]}...")  # Log first 500 chars for inspection

        # Step 3: Analyze content using Lyzr agent
        # Safely construct the prompt
        try:
            prompt_with_content = NOTION_ANALYSIS_PROMPT.replace("{content}", content)
            print(f"DEBUG 6: Formatted prompt length: {len(prompt_with_content)}")
            print(f"DEBUG 6.1: Formatted prompt preview: {prompt_with_content[:500]}...")
        except Exception as e:
            print(f"DEBUG ERROR: Failed to format prompt: {str(e)}")
            raise

        payload = {
            "user_id": USER_ID,
            "agent_id": "67ebb82eec01d3b71add9dc4",
            "session_id": f"notion_{project_id}_{page_id}",
            "message": prompt_with_content
        }
        print(f"DEBUG 7: Sending request to Lyzr API with payload: {json.dumps(payload, indent=2)}")

        analysis_result = requests.post(
            LYZR_API_URL,
            headers={
                "Content-Type": "application/json",
                "x-api-key": API_KEY
            },
            json=payload
        ).json()
        print(f"DEBUG 8: Raw Lyzr response: {json.dumps(analysis_result, indent=2)}")

        cleaned_result = analysis_result.get("response", "").replace('```json', '').replace('```', '').strip()
        print(f"DEBUG 9: Cleaned Lyzr response: {cleaned_result}")

        # Step 4: Parse the response with fallback
        try:
            extracted_data = json.loads(cleaned_result)
            if not isinstance(extracted_data, dict) or "user_stories" not in extracted_data or "technical_insights" not in extracted_data:
                raise ValueError("Response does not match expected format")
        except json.JSONDecodeError as e:
            print(f"DEBUG 10: JSON parsing failed: {str(e)}. Falling back to empty results.")
            extracted_data = {"user_stories": [], "technical_insights": []}
        except ValueError as e:
            print(f"DEBUG 10: Validation error: {str(e)}. Falling back to empty results.")
            extracted_data = {"user_stories": [], "technical_insights": []}

        user_stories = extracted_data.get("user_stories", [])
        technical_insights = extracted_data.get("technical_insights", [])
        print(f"DEBUG 11: Extracted {len(user_stories)} user stories and {len(technical_insights)} technical insights")

        # Step 5: Save and analyze user stories
        story_ids = []
        if user_stories:
            for story_text in user_stories:
                story_doc = {
                    "project_id": obj_id,
                    "text": story_text,
                    "analyzed": False,
                    "analysis_result": None,
                    "created_at": time.time(),
                    "source": f"notion_page_{page_id}"
                }
                result = await db.user_stories.insert_one(story_doc)
                story_ids.append(str(result.inserted_id))

            # Analyze user stories
            analysis_results = await story_manager.analyze_stories(user_stories, agent_id=user_story_agent_id)
            for i, story in enumerate(analysis_results["analysis_results"]):
                if i < len(story_ids):
                    await db.user_stories.update_one(
                        {"_id": ObjectId(story_ids[i])},
                        {"$set": {
                            "analyzed": True,
                            "analysis_result": {k: v for k, v in story.items() if k != "original_text"}
                        }}
                    )
        print(f"DEBUG 12: Saved and analyzed {len(story_ids)} user stories")

        # Step 6: Save and analyze technical insights
        insight_ids = []
        if technical_insights:
            for insight_text in technical_insights:
                insight_doc = {
                    "project_id": obj_id,
                    "text": insight_text,
                    "analyzed": False,
                    "analysis_result": None,
                    "created_at": time.time(),
                    "source": f"notion_page_{page_id}"
                }
                result = await db.technical_insights.insert_one(insight_doc)
                insight_ids.append(str(result.inserted_id))

            # Analyze technical insights in background
            background_tasks.add_task(process_technical_insights_background, project_id)
        print(f"DEBUG 13: Saved {len(insight_ids)} technical insights, queued for analysis")

        # Step 7: Train RAG with the page content
        chunked_documents = [{
            "id_": str(uuid.uuid4()),
            "embedding": None,
            "metadata": {"source": f"notion_page_{page_id}", "chunked": True},
            "text": content[i:i+1000].strip(),
            "excluded_embed_metadata_keys": [],
            "excluded_llm_metadata_keys": []
        } for i in range(0, len(content), 1000)]
        if chunked_documents:
            train_rag(rag_id, chunked_documents)
        print(f"DEBUG 14: Trained RAG with {len(chunked_documents)} chunks")

        # Step 8: Fetch analyzed data to return
        analyzed_stories = await db.user_stories.find({"_id": {"$in": [ObjectId(sid) for sid in story_ids]}}).to_list(None)
        analyzed_insights = await db.technical_insights.find({"_id": {"$in": [ObjectId(iid) for iid in insight_ids]}}).to_list(None)

        # Format response
        formatted_stories = [{"id": str(s["_id"]), "text": s["text"], "analysis_result": s.get("analysis_result")} for s in analyzed_stories]
        formatted_insights = [{"id": str(i["_id"]), "text": i["text"], "analysis_result": i.get("analysis_result")} for i in analyzed_insights]

        print("DEBUG 15: Preparing response")
        return {
            "message": "Notion page analyzed, user stories and technical insights extracted and processed",
            "user_stories": formatted_stories,
            "technical_insights": formatted_insights,
            "story_ids": story_ids,
            "insight_ids": insight_ids
        }

    except Exception as e:
        print(f"DEBUG ERROR: Error analyzing Notion page: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

##################################JIRA Integration###################################

ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "82nsFp4l4c-qOb2G7MPWLwJEmpRLlG2EiAF_443VgdA=")

cipher = Fernet(ENCRYPTION_KEY)

def encrypt_token(token: str) -> str:
    """Encrypt the Jira API token."""
    return cipher.encrypt(token.encode()).decode()

def decrypt_token(encrypted_token: str) -> str:
    """Decrypt the Jira API token."""
    return cipher.decrypt(encrypted_token.encode()).decode()

# Pydantic Models
class JiraTokenInput(BaseModel):
    email: str          # User's Atlassian email address
    api_token: str      # Jira API token
    jira_url: HttpUrl   # Client provides their Jira instance URL


class JiraSyncRequest(BaseModel):
    project_key: Optional[str] = None        # e.g., "MYPROJ"
    statuses: Optional[List[str]] = None     # e.g., ["To Do", "In Progress"]
    issue_types: Optional[List[str]] = None  # e.g., ["Bug", "Task"]
    assignee: Optional[str] = None           # e.g., "john.doe" (username or email)
    priority: Optional[str] = None           # e.g., "High", "Medium"
    labels: Optional[List[str]] = None       # e.g., ["frontend", "urgent"]

# Updated Helper Function to Get Decrypted Token and Email
async def get_jira_token(project_id: str) -> tuple[str, str, str]:
    """Retrieve and decrypt the Jira email, token, and URL for a project."""
    project = await db.projects.find_one({"_id": ObjectId(project_id)})
    if not project or "jira_integration" not in project:
        raise HTTPException(status_code=400, detail="Jira integration not configured")
    email = project["jira_integration"]["email"]
    encrypted_token = project["jira_integration"]["api_token"]
    jira_url = project["jira_integration"]["jira_url"]
    return email, decrypt_token(encrypted_token), jira_url

# 1. Add Jira Token Endpoint
@app.post("/project/{project_id}/jira/token")
async def add_jira_token(project_id: str, jira_input: JiraTokenInput):
    """
    Allows a client to submit their Jira email, API token, and URL for a project, validates it, and stores it securely.
    """
    try:
        # Validate project existence
        obj_id = ObjectId(project_id)
        project = await db.projects.find_one({"_id": obj_id})
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        print(f"DEBUG: Project {project_id} validated")

        # Validate the Jira credentials using Basic Auth
        test_url = f"{jira_input.jira_url}/rest/api/3/myself"
        auth = HTTPBasicAuth(jira_input.email, jira_input.api_token)
        headers = {
            "Accept": "application/json"
        }
        response = requests.get(test_url, headers=headers, auth=auth)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid Jira email, API token, or URL: {response.status_code} - {response.text}"
            )
        print(f"DEBUG: Jira credentials validated successfully for {jira_input.jira_url}")

        # Encrypt the API token
        encrypted_token = encrypt_token(jira_input.api_token)
        print(f"DEBUG: Token encrypted")

        # Store the email, encrypted token, and URL in the project document
        await db.projects.update_one(
            {"_id": obj_id},
            {
                "$set": {
                    "jira_integration": {
                        "email": jira_input.email,
                        "api_token": encrypted_token,
                        "jira_url": str(jira_input.jira_url),
                        "connected_at": datetime.datetime.utcnow()
                    }
                }
            }
        )
        print(f"DEBUG: Jira integration stored in project {project_id}")

        return {"message": "Jira email, API token, and URL added and validated successfully"}

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"DEBUG ERROR: Error adding Jira token: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add Jira token: {str(e)}")
    
@app.post("/project/{project_id}/jira/sync")
async def sync_jira_issues(project_id: str, sync_request: JiraSyncRequest, background_tasks: BackgroundTasks):
    try:
        obj_id = ObjectId(project_id)
        project = await db.projects.find_one({"_id": obj_id})
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        print(f"DEBUG: Project {project_id} validated for sync")

        email, token, jira_url = await get_jira_token(project_id)
        print(f"DEBUG: Retrieved Jira email and URL: {jira_url}")

        # Construct JQL query dynamically
        jql_parts = []
        project_key = sync_request.project_key or project.get("name", "DEFAULT")
        jql_parts.append(f"project = \"{project_key}\"")

        if sync_request.statuses:
            # Pre-quote statuses and join them
            quoted_statuses = [f'"{status}"' for status in sync_request.statuses]
            jql_parts.append(f"status in ({', '.join(quoted_statuses)})")

        if sync_request.issue_types:
            quoted_types = [f'"{issue_type}"' for issue_type in sync_request.issue_types]
            jql_parts.append(f"issuetype in ({', '.join(quoted_types)})")

        if sync_request.assignee:
            jql_parts.append(f"assignee = \"{sync_request.assignee}\"")

        if sync_request.priority:
            jql_parts.append(f"priority = \"{sync_request.priority}\"")

        if sync_request.labels:
            quoted_labels = [f'"{label}"' for label in sync_request.labels]
            jql_parts.append(f"labels in ({', '.join(quoted_labels)})")

        jql_query = " AND ".join(jql_parts)
        print(f"DEBUG: Generated JQL query: {jql_query}")

        # Fetch issues with additional fields
        search_url = f"{jira_url}/rest/api/3/search"
        auth = HTTPBasicAuth(email, token)
        headers = {"Accept": "application/json"}
        params = {
            "jql": jql_query,
            "maxResults": 100,
            "fields": "summary,description,status,issuetype,created,updated,assignee,reporter,comment,labels,priority,components,customfield_10010,subtasks,attachment,issuelinks"
        }
        print(f"DEBUG: Sending request to {search_url} with params: {params}")

        response = requests.get(search_url, headers=headers, auth=auth, params=params)
        print(f"DEBUG: Jira response status: {response.status_code}")

        if response.status_code != 200:
            print(f"DEBUG: Jira response text: {response.text}")
            raise HTTPException(status_code=400, detail=f"Failed to fetch Jira issues: {response.status_code} - {response.text}")

        issues = response.json().get("issues", [])
        print(f"DEBUG: Fetched {len(issues)} issues from Jira")

        background_tasks.add_task(sync_issues_to_db, obj_id, issues)
        return {"message": f"Syncing {len(issues)} Jira issues in the background"}

    except HTTPException as he:
        raise he
    except Exception as e:
        error_msg = f"Error syncing Jira issues: {str(e)}\n{traceback.format_exc()}"
        print(f"DEBUG ERROR: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")

async def sync_issues_to_db(project_id: ObjectId, issues: list):
    """Background task to sync Jira issues into MongoDB with additional fields."""
    try:
        for issue in issues:
            issue_key = issue["key"]
            fields = issue["fields"]
            issue_doc = {
                "project_id": project_id,
                "jira_key": issue_key,
                "summary": fields["summary"],
                "description": fields.get("description", ""),
                "status": fields["status"]["name"],
                "issue_type": fields["issuetype"]["name"],
                "created_at": fields["created"],
                "updated_at": fields["updated"],
                "synced_at": datetime.datetime.utcnow(),
                # Additional fields
                "assignee": fields["assignee"]["displayName"] if fields.get("assignee") else None,
                "reporter": fields["reporter"]["displayName"] if fields.get("reporter") else None,
                "comments": [comment["body"] for comment in fields.get("comment", {}).get("comments", [])],
                "labels": fields.get("labels", []),
                "priority": fields["priority"]["name"] if fields.get("priority") else None,
                "components": [comp["name"] for comp in fields.get("components", [])],
                "epic_link": fields.get("customfield_10010"),  # Adjust custom field ID as needed
                "subtasks": [sub["key"] for sub in fields.get("subtasks", [])],
                "attachments": [{"filename": att["filename"], "url": att["content"]} for att in fields.get("attachment", [])],
                "linked_issues": [{"type": link["type"]["name"], "key": link.get("outwardIssue", {}).get("key", link.get("inwardIssue", {}).get("key"))} for link in fields.get("issuelinks", [])]
            }
            await db.jira_issues.update_one(
                {"project_id": project_id, "jira_key": issue_key},
                {"$set": issue_doc},
                upsert=True
            )
        print(f"DEBUG: Synced {len(issues)} issues to MongoDB for project {project_id}")
    except Exception as e:
        error_msg = f"Error in sync_issues_to_db: {str(e)}\n{traceback.format_exc()}"
        print(f"DEBUG ERROR: {error_msg}")
        raise

# 3. Get Synced Jira Issues Endpoint
@app.get("/project/{project_id}/jira/issues")
async def get_jira_issues(project_id: str):
    """
    Retrieves all synced Jira issues for a project.
    """
    try:
        obj_id = ObjectId(project_id)
        project = await db.projects.find_one({"_id": obj_id})
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        print(f"DEBUG: Project {project_id} validated for issue retrieval")

        issues_cursor = db.jira_issues.find({"project_id": obj_id})
        issues = await issues_cursor.to_list(length=1000)  # Adjust limit as needed
        
        # Convert ObjectId to string for JSON response
        for issue in issues:
            issue["_id"] = str(issue["_id"])
            issue["project_id"] = str(issue["project_id"])
        
        print(f"DEBUG: Retrieved {len(issues)} issues for project {project_id}")
        return {"issues": issues}

    except Exception as e:
        print(f"DEBUG ERROR: Error retrieving Jira issues: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve issues: {str(e)}")

# 4. Remove Jira Integration Endpoint
@app.delete("/project/{project_id}/jira/token")
async def remove_jira_token(project_id: str):
    """
    Removes Jira integration from a project.
    """
    try:
        obj_id = ObjectId(project_id)
        project = await db.projects.find_one({"_id": obj_id})
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        print(f"DEBUG: Project {project_id} validated for removal")

        result = await db.projects.update_one(
            {"_id": obj_id},
            {"$unset": {"jira_integration": ""}}
        )
        if result.modified_count == 0:
            raise HTTPException(status_code=400, detail="No Jira integration to remove")
        
        # Optionally, clean up synced issues
        await db.jira_issues.delete_many({"project_id": obj_id})
        print(f"DEBUG: Removed Jira integration and issues for project {project_id}")
        
        return {"message": "Jira integration removed successfully"}

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"DEBUG ERROR: Error removing Jira token: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Removal failed: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "healthy"}