RAG Web Scraper with Weaviate

An end-to-end Retrieval-Augmented Generation (RAG) system that automatically searches the web, scrapes webpages and PDFs, converts content into vector embeddings, stores them in **Weaviate**, and retrieves relevant information using **semantic similarity search**.

This project demonstrates a real-world AI pipeline combining **web scraping, NLP embeddings, vector databases, and Dockerized deployment**.



 Key Features

-  Web search using DuckDuckGo
-  Webpage scraping with BeautifulSoup
-  PDF text extraction
-  Sentence Transformer embeddings
-  Vector storage using Weaviate
-  Semantic vector similarity search
-  Docker-based Weaviate deployment
-  Interactive question-answering CLI


 System Architecture

1. User inputs a question
2. DuckDuckGo retrieves relevant URLs
3. Webpages and PDFs are scraped
4. Text is cleaned and truncated
5. Sentence Transformers generate embeddings
6. Embeddings are stored in Weaviate
7. Query embedding is compared using vector similarity
8. Most relevant documents are returned

Tech Stack

| Layer | Technology |
|-----|------------|
| Programming Language | Python |
| Embedding Model | Sentence Transformers |
| Vector Database | Weaviate |
| Web Scraping | Requests, BeautifulSoup |
| PDF Processing | PyPDF |
| Search Engine | DuckDuckGo |
| Containerization | Docker, Docker Compose |



Project Structure
rag-weaviate-webscraper/
│
├── main.py # Main RAG pipeline
├── docker-compose.yml # Weaviate Docker setup
├── requirements.txt # Python dependencies
├── .gitignore # Ignored files
└── webscraper_with_weaviate.code-workspace

Running Weaviate (Docker)

Ensure **Docker Desktop** is running.

bash:
docker-compose up -d

Weaviate will run at:

http://localhost:8080

To stop
docker-compose down

Running the Application
Create Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate   # Windows

Install Dependencies
pip install -r requirements.txt

Start the RAG System
python main.py

Sample Interaction
Your question: What is vector database?

Searching web...
Processing and storing knowledge...
Generating answer...

Answer:
Based on 3 sources:
- Explanation of vector databases
- Usage in AI systems
- Semantic similarity search

Use Cases

AI-powered research assistant

Knowledge base creation

Academic document analysis

Semantic search engines

RAG-based AI systems

Future Improvements

LLM integration (Gemini / GPT / LLaMA)

Web UI (Streamlit / React)

Citation-based answers

Multilingual document ingestion

Cloud deployment

**Author**

**Kavin Saran**
GitHub: https://github.com/KavinSaran29


