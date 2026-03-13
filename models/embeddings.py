from config.config import EMBEDDING_MODEL, GEMINI_API_KEY

def get_embedding_model():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    try:
        model = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GEMINI_API_KEY,
        )
        return model
    except Exception as e:
        last_error = e
    raise RuntimeError(f"Failed to initialise embedding model: {last_error}") from last_error
