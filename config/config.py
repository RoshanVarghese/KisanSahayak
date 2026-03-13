import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
 
DEFAULT_PROVIDER = "gemini"          
GEMINI_MODEL = "gemini-3-flash-preview"
GROQ_MODEL = "llama-3.1-8b-instant"
OPENAI_MODEL = "gpt-3.5-turbo"

EMBEDDING_MODEL = "gemini-embedding-001"

CHUNK_SIZE = 1000   
CHUNK_OVERLAP = 200    
RETRIEVAL_TOP_K = 4      

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
PROJECT_ROOT = os.path.dirname(BASE_DIR)                   
DATA_DIR = os.path.join(PROJECT_ROOT, "data")          
CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_db")     
