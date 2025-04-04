import os
import json
import logging
import requests
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm  # For Colab progress bars

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TechnicalInsightsManager:
    """
    Manager for technical insights from HLDs and DLDs with search capabilities.
    Supports categorization by API, Server, Database, Architecture, Tools, and Patterns.
    """

    def __init__(self, persist_directory: str = "./chromadb_insights"):
        """Initialize the manager with a directory for ChromaDB."""
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        self.collections = {}
        self.analysis_results = []
        self.categories = []
        self.components = []
        self.tech_areas = []
        self.design_patterns = []
        self.openai_ef = None
        self.api_url = "https://agent-prod.studio.lyzr.ai/v3/inference/chat/"
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": "sk-default-yStV4gbpjadbQSw4i7QhoOLRwAs5dEcl"
        }
        self.user_id = "pranav@lyzr.ai"

    # def setup_openai(self, api_key: str):
    #     """Set up OpenAI API for embeddings and analysis."""
    #     openai.api_key = api_key
    #     os.environ["OPENAI_API_KEY"] = api_key

    #     # Initialize embedding function
    #     self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    #         api_key=api_key,
    #         model_name="text-embedding-3-small"
    #     )

    #     logger.info("OpenAI API configured successfully")

    def _call_lyzr_api(self, agent_id: str, session_id: str, text: str, prompt: str) -> str:
        """Helper function to call Lyzr API"""
        formatted_prompt = prompt.format(text=text)
        

        payload = {
            "user_id": self.user_id,
            "agent_id": agent_id,
            "session_id": session_id,
            "message": formatted_prompt
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

    def analyze_insights(self, insights: List[str], agent_id: str, analyze_prompt: str = None, batch_size: int = 5):
        """
        Analyze technical insights to extract structured information.

        Args:
            insights: List of technical insight texts
            analyze_prompt: Custom prompt for analysis (or use default)
            batch_size: Number of insights to analyze in each batch
        """
        if not analyze_prompt:
            analyze_prompt = """
            Analyze this technical insight and extract detailed information:

            Technical Insight: "{text}"

            Provide the following in JSON format:
            1. title: Short title for the insight
            2. description: Brief description of what the insight is about
            3. level: Whether this is a high-level (HLD) or detailed-level (DLD) insight
            4. category: Main category (one of: API, Server, Database, Architecture, Tools, Patterns)
            5. components: List of specific technical components referenced (frameworks, libraries, etc.)
            6. tech_areas: List of 3-5 technical areas this insight relates to
            7. design_patterns: List of design patterns or architectural approaches mentioned, if any
            8. impact: Estimated impact (high, medium, low)
            9. implementation_complexity: Estimated implementation complexity (high, medium, low)

            Return ONLY valid JSON with no additional text or formatting.
            """

        logger.info(f"Analyzing {len(insights)} technical insights...")
        analysis_results = []
        print("agent_id",agent_id)

        # Process in batches
        for i in tqdm(range(0, len(insights), batch_size), desc="Analyzing insights"):
            batch = insights[i:i+batch_size]

            for insight in batch:
                try:
                    result = self._call_lyzr_api(
                        agent_id=agent_id,
                        session_id=agent_id,
                        text=insight,
                        prompt=analyze_prompt
                    )
                    logger.info(f"result from lyzr agent: {result}")
                    result = json.loads(result)
                    logger.info(f"result from lyzr agent 2: {result}")
                    result['original_text'] = insight
                    analysis_results.append(result)
                except Exception as e:
                    logger.error(f"Error analyzing insight: {e}")
                    # Add a minimal entry
                    analysis_results.append({
                        "original_text": insight,
                        "error": str(e)
                    })

        self.analysis_results = analysis_results

        # Extract unique categories, components, etc.
        self._extract_metadata_fields()

        return {
            "analysis_results": analysis_results,
            "categories": self.categories,
            "components": self.components,
            "tech_areas": self.tech_areas,
            "design_patterns": self.design_patterns
        }

    # def _analyze_with_openai(self, text: str, prompt_template: str) -> Dict[str, Any]:
    #     """Analyze text with OpenAI API."""
    #     formatted_prompt = prompt_template.format(text=text)

    #     response = openai.chat.completions.create(
    #         model="gpt-3.5-turbo",  # Using a more cost-effective model
    #         messages=[{"role": "user", "content": formatted_prompt}],
    #         temperature=0.2,
    #         response_format={"type": "json_object"},  # Request JSON directly
    #         max_tokens=800
    #     )

    #     content = response.choices[0].message.content.strip()

    #     try:
    #         return json.loads(content)
    #     except json.JSONDecodeError:
    #         # Try to extract JSON if surrounded by other text
    #         import re
    #         json_match = re.search(r'(\{.*\})', content, re.DOTALL)
    #         if json_match:
    #             try:
    #                 return json.loads(json_match.group(1))
    #             except:
    #                 pass

    #         # Return as plain text if JSON parsing fails
    #         return {
    #             "content": content,
    #             "error": "Failed to parse as JSON"
    #         }

    def _extract_metadata_fields(self):
        """Extract unique categories, components, tech areas, and design patterns from analysis results."""
        categories = set()
        components = set()
        tech_areas = set()
        design_patterns = set()

        for result in self.analysis_results:
            # Extract categories
            if 'category' in result:
                categories.add(result['category'])

            # Extract components
            if 'components' in result:
                if isinstance(result['components'], list):
                    for component in result['components']:
                        components.add(str(component).lower())
                elif isinstance(result['components'], str):
                    # Handle comma-separated components
                    for component in result['components'].split(','):
                        components.add(component.strip().lower())

            # Extract tech areas
            if 'tech_areas' in result:
                if isinstance(result['tech_areas'], list):
                    for area in result['tech_areas']:
                        tech_areas.add(str(area).lower())
                elif isinstance(result['tech_areas'], str):
                    # Handle comma-separated tech areas
                    for area in result['tech_areas'].split(','):
                        tech_areas.add(area.strip().lower())

            # Extract design patterns
            if 'design_patterns' in result:
                if isinstance(result['design_patterns'], list):
                    for pattern in result['design_patterns']:
                        design_patterns.add(str(pattern).lower())
                elif isinstance(result['design_patterns'], str):
                    # Handle comma-separated design patterns
                    for pattern in result['design_patterns'].split(','):
                        design_patterns.add(pattern.strip().lower())

        self.categories = sorted(list(categories))
        self.components = sorted(list(components))
        self.tech_areas = sorted(list(tech_areas))
        self.design_patterns = sorted(list(design_patterns))

    # def create_collections(self):
    #     """Create ChromaDB collections for insights."""
    #     if not self.openai_ef:
    #         raise ValueError("OpenAI API not configured. Call setup_openai() first.")

    #     # Create main collection for all insights
    #     self._create_collection("all_insights", self.analysis_results)

    #     # Create collections for each category (API, Server, Database, etc.)
    #     for category in self.categories:
    #         category_docs = [doc for doc in self.analysis_results
    #                         if doc.get('category', '').lower() == category.lower()]
    #         if category_docs:
    #             self._create_collection(f"category_{category.lower().replace(' ', '_')}",
    #                                    category_docs)

    #     # Create collections for HLD vs DLD
    #     hld_docs = [doc for doc in self.analysis_results if doc.get('level', '').lower() == 'hld']
    #     dld_docs = [doc for doc in self.analysis_results if doc.get('level', '').lower() == 'dld']

    #     if hld_docs:
    #         self._create_collection("hld_insights", hld_docs)
    #     if dld_docs:
    #         self._create_collection("dld_insights", dld_docs)

    #     return self.collections

    # def _create_collection(self, name: str, documents: List[Dict[str, Any]]):
    #     """Create a ChromaDB collection with documents."""
    #     # Delete if exists
    #     try:
    #         self.client.delete_collection(name)
    #     except:
    #         pass

    #     # Create collection
    #     collection = self.client.create_collection(
    #         name=name,
    #         embedding_function=self.openai_ef
    #     )

    #     if documents:
    #         # Prepare data for collection
    #         docs = []
    #         metadatas = []
    #         ids = []

    #         for i, doc in enumerate(documents):
    #             # Use original text for embedding
    #             text = doc.get('original_text', '')
    #             docs.append(text)

    #             # Create metadata from all fields except original_text
    #             metadata = {k: v for k, v in doc.items() if k != 'original_text'}

    #             # Convert lists to strings for metadata
    #             for key, value in metadata.items():
    #                 if isinstance(value, list):
    #                     metadata[key] = ", ".join(str(item) for item in value)

    #             metadatas.append(metadata)
    #             ids.append(f"{name}-{i}")

    #         # Add to collection
    #         collection.add(
    #             documents=docs,
    #             metadatas=metadatas,
    #             ids=ids
    #         )

    #     self.collections[name] = collection
    #     return collection

    def search(self, query: str, collection_name: str = "all_insights", n_results: int = 5):
        """Search in a specific collection."""
        if collection_name not in self.collections:
            logger.error(f"Collection '{collection_name}' not found")
            return []

        collection = self.collections[collection_name]

        try:
            # Perform search
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )

            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i],
                    "id": results['ids'][0][i]
                }
                formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []

    def filter_search(self,
                     query: str = None,
                     category: str = None,
                     level: str = None,
                     component: str = None,
                     tech_area: str = None,
                     design_pattern: str = None,
                     impact: str = None,
                     complexity: str = None,
                     n_results: int = 5):
        """
        Search with filters for category, level, component, tech area, design pattern, etc.

        If query is None, returns all documents matching the filters.
        """
        collection = self.collections.get("all_insights")
        if not collection:
            logger.error("Main collection not found")
            return []

        # Build where clause
        where_clause = {}
        if category:
            where_clause["category"] = {"$eq": category}
        if level:
            where_clause["level"] = {"$eq": level.lower()}
        if component:
            where_clause["components"] = {"$contains": component.lower()}
        if tech_area:
            where_clause["tech_areas"] = {"$contains": tech_area.lower()}
        if design_pattern:
            where_clause["design_patterns"] = {"$contains": design_pattern.lower()}
        if impact:
            where_clause["impact"] = {"$eq": impact.lower()}
        if complexity:
            where_clause["implementation_complexity"] = {"$eq": complexity.lower()}

        try:
            # Perform search
            if query:
                results = collection.query(
                    query_texts=[query],
                    where=where_clause if where_clause else None,
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"]
                )
            else:
                # If no query, get all documents matching filters
                results = collection.query(
                    where=where_clause if where_clause else None,
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"]
                )

            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else 1.0,
                    "id": results['ids'][0][i]
                }
                formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logger.error(f"Error during filtered search: {e}")
            return []

    def display_results(self, results):
        """Display search results in a readable format."""
        if not results:
            print("No results found.")
            return

        print(f"\nFound {len(results)} result(s):")
        print("-" * 80)

        for i, result in enumerate(results, 1):
            # Extract document content
            document = result.get('document', 'Content not available')

            # Get metadata
            metadata = result.get('metadata', {})

            # Get relevance score
            distance = result.get('distance', 1.0)
            relevance = 1.0 - distance

            # Print result
            print(f"Result #{i}:")

            # Print title if available
            if 'title' in metadata:
                print(f"🔷 Title: {metadata['title']}")

            # Print category and level if available
            if 'category' in metadata:
                print(f"📁 Category: {metadata['category']}")
            if 'level' in metadata:
                print(f"🔍 Level: {metadata['level']}")

            # Print impact and complexity if available
            if 'impact' in metadata:
                print(f"💥 Impact: {metadata['impact']}")
            if 'implementation_complexity' in metadata:
                print(f"⚙️ Complexity: {metadata['implementation_complexity']}")

            # Print components if available
            if 'components' in metadata:
                print(f"🧩 Components: {metadata['components']}")

            # Print tech areas if available
            if 'tech_areas' in metadata:
                print(f"🔬 Tech Areas: {metadata['tech_areas']}")

            # Print design patterns if available
            if 'design_patterns' in metadata:
                print(f"📐 Design Patterns: {metadata['design_patterns']}")

            # Print relevance
            print(f"📊 Relevance: {relevance:.2f}")

            # Print description if available
            if 'description' in metadata:
                print(f"📝 Description: {metadata['description']}")

            # Print the insight text
            print(f"📄 Insight: {document}")

            print("-" * 80)

    def get_metadata_options(self):
        """Return all available metadata options for filtering."""
        return {
            "categories": self.categories,
            "components": self.components,
            "tech_areas": self.tech_areas,
            "design_patterns": self.design_patterns,
            "levels": ["hld", "dld"],
            "impact": ["high", "medium", "low"],
            "complexity": ["high", "medium", "low"]
        }

    def to_dataframe(self):
        """Convert analysis results to pandas DataFrame for easy viewing."""
        # Extract key fields for DataFrame
        data = []
        for result in self.analysis_results:
            # Create base row with all standard fields
            row = {
                "insight": result.get("original_text", ""),
                "title": result.get("title", ""),
                "description": result.get("description", ""),
                "level": result.get("level", ""),
                "category": result.get("category", ""),
                "impact": result.get("impact", ""),
                "implementation_complexity": result.get("implementation_complexity", "")
            }

            # Handle components (convert list to string if needed)
            components = result.get("components", [])
            row["components"] = ", ".join(components) if isinstance(components, list) else components

            # Handle tech areas (convert list to string if needed)
            tech_areas = result.get("tech_areas", [])
            row["tech_areas"] = ", ".join(tech_areas) if isinstance(tech_areas, list) else tech_areas

            # Handle design patterns (convert list to string if needed)
            design_patterns = result.get("design_patterns", [])
            row["design_patterns"] = ", ".join(design_patterns) if isinstance(design_patterns, list) else design_patterns

            # Add any additional fields from the result that weren't explicitly mapped
            for key, value in result.items():
                if key not in row and key != "original_text" and key != "error":
                    # Convert lists to strings
                    if isinstance(value, list):
                        row[key] = ", ".join(str(item) for item in value)
                    else:
                        row[key] = value

            data.append(row)

        # Create DataFrame
        df = pd.DataFrame(data)

        # Reorder columns to place most important ones first
        important_cols = ["insight", "title", "description", "level", "category",
                         "impact", "implementation_complexity", "components",
                         "tech_areas", "design_patterns"]

        # Filter to only include columns that actually exist in the dataframe
        important_cols = [col for col in important_cols if col in df.columns]

        # Get remaining columns
        other_cols = [col for col in df.columns if col not in important_cols]

        # Reorder columns
        df = df[important_cols + other_cols]

        return df

    def create_interactive_dashboard(self):
        """Create an interactive dashboard for exploring insights."""
        try:
            # Try to import ipywidgets - install if not present
            import ipywidgets as widgets
        except ImportError:
            print("Installing ipywidgets...")
            import subprocess
            subprocess.check_call(["pip", "install", "ipywidgets"])
            import ipywidgets as widgets

        from IPython.display import display, HTML, clear_output

        # Get metadata options
        options = self.get_metadata_options()

        # Create widgets
        search_input = widgets.Text(
            value='',
            placeholder='Enter search term...',
            description='Search:',
            disabled=False,
            layout=widgets.Layout(width='50%')
        )

        category_dropdown = widgets.Dropdown(
            options=['All'] + options['categories'],
            value='All',
            description='Category:',
            disabled=False,
        )

        level_dropdown = widgets.Dropdown(
            options=['All', 'HLD', 'DLD'],
            value='All',
            description='Level:',
            disabled=False,
        )

        impact_dropdown = widgets.Dropdown(
            options=['All', 'high', 'medium', 'low'],
            value='All',
            description='Impact:',
            disabled=False,
        )

        complexity_dropdown = widgets.Dropdown(
            options=['All', 'high', 'medium', 'low'],
            value='All',
            description='Complexity:',
            disabled=False,
        )

        # Only include components if there are some
        if options['components']:
            component_dropdown = widgets.Dropdown(
                options=['All'] + options['components'][:30],  # Limit to first 30
                value='All',
                description='Component:',
                disabled=False,
            )
        else:
            component_dropdown = None

        # Only include tech areas if there are some
        if options['tech_areas']:
            tech_area_dropdown = widgets.Dropdown(
                options=['All'] + options['tech_areas'][:30],  # Limit to first 30
                value='All',
                description='Tech Area:',
                disabled=False,
            )
        else:
            tech_area_dropdown = None

        # Only include design patterns if there are some
        if options['design_patterns']:
            design_pattern_dropdown = widgets.Dropdown(
                options=['All'] + options['design_patterns'][:30],  # Limit to first 30
                value='All',
                description='Pattern:',
                disabled=False,
            )
        else:
            design_pattern_dropdown = None

        results_count = widgets.IntSlider(
            value=5,
            min=1,
            max=20,
            step=1,
            description='# Results:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )

        search_button = widgets.Button(
            description='Search',
            disabled=False,
            button_style='primary',
            tooltip='Click to search',
            icon='search'
        )

        output = widgets.Output()

        # Function to handle search button click
        def on_search_button_clicked(b):
            with output:
                clear_output()

                # Get filter values
                query = search_input.value if search_input.value else None
                category = category_dropdown.value if category_dropdown.value != 'All' else None
                level = level_dropdown.value if level_dropdown.value != 'All' else None
                impact = impact_dropdown.value if impact_dropdown.value != 'All' else None
                complexity = complexity_dropdown.value if complexity_dropdown.value != 'All' else None
                component = component_dropdown.value if component_dropdown and component_dropdown.value != 'All' else None
                tech_area = tech_area_dropdown.value if tech_area_dropdown and tech_area_dropdown.value != 'All' else None
                design_pattern = design_pattern_dropdown.value if design_pattern_dropdown and design_pattern_dropdown.value != 'All' else None
                n_results = results_count.value

                # Perform search
                results = self.filter_search(
                    query=query,
                    category=category,
                    level=level,
                    component=component,
                    tech_area=tech_area,
                    design_pattern=design_pattern,
                    impact=impact,
                    complexity=complexity,
                    n_results=n_results
                )

                # Display results
                self.display_results(results)

        # Connect button to function
        search_button.on_click(on_search_button_clicked)

        # Create layout - first row
        first_row_widgets = [search_input, level_dropdown, category_dropdown]

        # Create layout - second row
        second_row_widgets = [impact_dropdown, complexity_dropdown]
        if component_dropdown:
            second_row_widgets.append(component_dropdown)

        # Create layout - third row
        third_row_widgets = []
        if tech_area_dropdown:
            third_row_widgets.append(tech_area_dropdown)
        if design_pattern_dropdown:
            third_row_widgets.append(design_pattern_dropdown)
        third_row_widgets.append(results_count)
        third_row_widgets.append(search_button)

        # Display dashboard
        display(HTML("<h2>Technical Insights Explorer</h2>"))
        display(widgets.VBox([
            widgets.HBox(first_row_widgets),
            widgets.HBox(second_row_widgets),
            widgets.HBox(third_row_widgets),
            output
        ]))

        # Initial search to show all results
        on_search_button_clicked(None)

    def visualize_insights(self):
        """Create visualizations for insight metrics."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("Installing matplotlib and seaborn...")
            import subprocess
            subprocess.check_call(["pip", "install", "matplotlib", "seaborn"])
            import matplotlib.pyplot as plt
            import seaborn as sns

        # Get data as DataFrame
        df = self.to_dataframe()

        # Set up the figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Category distribution
        if 'category' in df.columns:
            category_counts = df['category'].value_counts()
            sns.barplot(x=category_counts.index, y=category_counts.values, ax=axes[0, 0], palette='viridis')
            axes[0, 0].set_title('Insights by Category', fontsize=14)
            axes[0, 0].set_xlabel('Category', fontsize=12)
            axes[0, 0].set_ylabel('Count', fontsize=12)
            axes[0, 0].tick_params(axis='x', rotation=45)

            # Add count labels
            for i, count in enumerate(category_counts.values):
                axes[0, 0].text(i, count + 0.1, str(count), ha='center')

        # 2. Level distribution (HLD vs DLD)
        if 'level' in df.columns:
            level_counts = df['level'].value_counts()
            colors = ['#1f77b4', '#ff7f0e']  # Blue for HLD, Orange for DLD
            sns.barplot(x=level_counts.index, y=level_counts.values, ax=axes[0, 1], palette=colors)
            axes[0, 1].set_title('Insights by Design Level', fontsize=14)
            axes[0, 1].set_xlabel('Level', fontsize=12)
            axes[0, 1].set_ylabel('Count', fontsize=12)

            # Add count labels
            for i, count in enumerate(level_counts.values):
                axes[0, 1].text(i, count + 0.1, str(count), ha='center')

        # 3. Impact vs Complexity heatmap
        if 'impact' in df.columns and 'implementation_complexity' in df.columns:
            # Create a cross-tabulation
            impact_order = ['high', 'medium', 'low']
            complexity_order = ['low', 'medium', 'high']

            # Convert to lowercase for consistency
            df['impact'] = df['impact'].str.lower()
            df['implementation_complexity'] = df['implementation_complexity'].str.lower()

            # Create cross-tabulation
            heatmap_data = pd.crosstab(
                df['impact'],
                df['implementation_complexity'],
                margins=False
            ).reindex(index=impact_order, columns=complexity_order, fill_value=0)

            # Create heatmap
            sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlGnBu', ax=axes[1, 0])
            axes[1, 0].set_title('Impact vs Implementation Complexity', fontsize=14)
            axes[1, 0].set_xlabel('Implementation Complexity', fontsize=12)
            axes[1, 0].set_ylabel('Impact', fontsize=12)

        # 4. Top components or tech areas
        if 'tech_areas' in df.columns:
            # Extract individual tech areas
            try:
                from wordcloud import WordCloud

                # Combine all tech areas
                all_tech_areas = " ".join(df['tech_areas'].fillna(''))

                # Create word cloud
                wordcloud = WordCloud(width=800, height=400, background_color='white',
                                     colormap='plasma', max_words=100).generate(all_tech_areas)

                # Display word cloud
                axes[1, 1].imshow(wordcloud, interpolation='bilinear')
                axes[1, 1].set_title('Tech Areas Word Cloud', fontsize=14)
                axes[1, 1].axis('off')
            except ImportError:
                # Fall back to bar chart if wordcloud is not available
                # Extract individual tech areas
                tech_areas = []
                for area_list in df['tech_areas'].fillna(''):
                    tech_areas.extend([area.strip() for area in area_list.split(',')])

                # Count tech areas
                from collections import Counter
                area_counts = Counter(tech_areas)
                top_areas = dict(area_counts.most_common(10))

                # Create bar chart
                sns.barplot(x=list(top_areas.values()), y=list(top_areas.keys()), ax=axes[1, 1], palette='cool')
                axes[1, 1].set_title('Top 10 Technical Areas', fontsize=14)
                axes[1, 1].set_xlabel('Count', fontsize=12)
                axes[1, 1].set_ylabel('Tech Area', fontsize=12)

        plt.tight_layout()
        plt.show()

        return fig

