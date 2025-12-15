import requests
from bs4 import BeautifulSoup
import weaviate
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.classes.query import Filter
import datetime
import warnings
from duckduckgo_search import DDGS
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import time
import io
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

class RAGSystem:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = None
        self.collection = None

    def connect_weaviate(self):
        """Connect to Weaviate instance"""
        print("Connecting to Weaviate...")
        try:
            self.client = weaviate.connect_to_local(
                host="localhost",
                port=8080,
                grpc_port=50051,
                additional_config=AdditionalConfig(
                    timeout=Timeout(init=2, query=45, insert=30)
                )
                )
            print("Connected successfully!")
            return True
        except Exception as e:
            print(f"Connection failed: {str(e)}")
            return False

    def setup_collection(self):
        """Create or get collection with vector support"""
        collection_name = "KnowledgeBase"
        
        try:
            # First check if collection exists
            if self.client.collections.exists(collection_name):
                print(f"Found existing collection '{collection_name}'")
                self.collection = self.client.collections.get(collection_name)
                print("Successfully accessed collection")
                return True
            
            # Create new collection if it doesn't exist
            print(f"Creating new collection '{collection_name}'")
            self.collection = self.client.collections.create(
                name=collection_name,
                properties=[
                    Property(name="url", data_type=DataType.TEXT),
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="source_type", data_type=DataType.TEXT),
                    Property(name="timestamp", data_type=DataType.DATE),
                ],
                vectorizer_config=Configure.Vectorizer.none()
            )
            print(f"Successfully created collection '{collection_name}'")
            return True
            
        except Exception as e:
            print(f"Collection operation failed: {str(e)}")
            return False

    def search_web(self, query, num_results=10):
        """Search web using DuckDuckGo"""
        print(f"\nSearching web for: '{query}'")
        try:
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(query, max_results=num_results)]
                return [result['href'] for result in results]
        except Exception as e:
            print(f"Web search failed: {str(e)}")
            return []

    def extract_pdf_text(self, url):
        """Extract text from PDF using pypdf"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            with io.BytesIO(response.content) as f:
                reader = PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text[:50000]  # Limit to 50k characters
        except Exception as e:
            print(f"Error processing PDF {url}: {str(e)}")
            return None

    def extract_webpage_text(self, url):
        """Extract text from webpage"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            for element in soup(['script', 'style', 'nav', 'footer']):
                element.decompose()
                
            text = soup.get_text(separator=' ', strip=True)
            return text[:50000]  # Limit to 50k characters
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None

    def get_embedding(self, text):
        """Generate embedding for text"""
        return self.embedding_model.encode(text)

    def store_knowledge(self, query, urls):
        """Store knowledge with embeddings"""
        print("\nProcessing and storing knowledge...")
        stored_count = 0
        
        for url in urls:
            try:
                # Check if URL already exists
                existing = self.collection.query.fetch_objects(
                    filters=Filter.by_property("url").equal(url),
                    limit=1
                )
                if len(existing.objects) > 0:
                    print(f"Document already exists: {url}")
                    continue
                    
                # Process content based on URL type
                if url.lower().endswith('.pdf'):
                    content = self.extract_pdf_text(url)
                    source_type = "pdf"
                else:
                    content = self.extract_webpage_text(url)
                    source_type = "webpage"
                    
                if not content:
                    continue
                    
                # Generate embeddings
                embedding = self.get_embedding(content)
                title = url.split('/')[-1][:100]  # Simple title from URL
                
                # Get current time in RFC3339 format
                now = datetime.datetime.now(datetime.timezone.utc)
                timestamp = now.isoformat(timespec='milliseconds').replace('+00:00', 'Z')
                
                # Store with vector
                self.collection.data.insert(
                    properties={
                        "url": url,
                        "title": title,
                        "content": content[:10000],  # Store first 10k chars
                        "source_type": source_type,
                        "timestamp": timestamp
                    },
                    vector=embedding.tolist()
                )
                stored_count += 1
                print(f"Stored: {title} ({source_type})")
                time.sleep(1)  # Be polite
                
            except Exception as e:
                print(f"Error storing {url}: {str(e)}")
        
        return stored_count

    def get_rag_answer(self, query):
        """Get RAG answer using vector similarity"""
        try:
            # Embed the query
            query_embedding = self.get_embedding(query)
            
            # Vector search
            response = self.collection.query.near_vector(
                near_vector=query_embedding.tolist(),
                limit=3,
                return_metadata=["distance"],
                return_properties=["title", "url", "content"]
            )
            
            if not response.objects:
                return None
                
            # Format context for answer
            context = "\n\n".join([
                f"Source {i+1} ({obj.properties['title']}):\n{obj.properties['content'][:1000]}..."
                for i, obj in enumerate(response.objects)
            ])
            
            # Simple answer generation
            answer = (
                f"Based on {len(response.objects)} sources:\n\n"
                f"{context}\n\n"
                f"Most relevant source: {response.objects[0].properties['url']}"
            )
            
            return answer
        except Exception as e:
            print(f"Error in RAG: {str(e)}")
            return None

    def run(self):
        """Main interactive function"""
        if not self.connect_weaviate():
            return
            
        if not self.setup_collection():
            self.client.close()
            return
            
        print("\nKnowledge RAG System Ready. Enter your question (or 'exit' to quit)")
        
        while True:
            try:
                query = input("\nYour question: ").strip()
                
                if query.lower() in ['exit', 'quit']:
                    break
                    
                if not query:
                    continue
                
                # Step 1: Search web for relevant URLs
                urls = self.search_web(query)
                if not urls:
                    print("No search results found. Try a different query.")
                    continue
                
                # Step 2: Process and store knowledge
                stored = self.store_knowledge(query, urls)
                print(f"\nProcessed {stored} new knowledge sources")
                
                # Step 3: Get RAG answer
                print("\nGenerating answer...")
                answer = self.get_rag_answer(query)
                
                if not answer:
                    print("Could not generate answer. Try a different query.")
                    continue
                
                # Display answer
                print(f"\nAnswer for: '{query}'")
                print("-"*50)
                print(answer)
                print("="*50)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                
        self.client.close()
        print("\nWeaviate connection closed")

if __name__ == "__main__":
    rag_system = RAGSystem()
    rag_system.run()