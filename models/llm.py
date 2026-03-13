from config.config import (
    DEFAULT_PROVIDER,
    GEMINI_MODEL,
    GROQ_MODEL,
    OPENAI_MODEL,
    OPENAI_API_KEY,
    GEMINI_API_KEY,
    GROQ_API_KEY,
)

def get_llm(provider: str = None, model: str = None, temperature: float = 0.3):
    provider = (provider or DEFAULT_PROVIDER).lower()
    try:
        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model or GEMINI_MODEL,
                google_api_key=GEMINI_API_KEY,
                temperature=temperature,
            )
        elif provider == "openai":
            from langchain_openai import ChatOpenAI 
            return ChatOpenAI(
                model=model or OPENAI_MODEL,
                api_key=OPENAI_API_KEY,
                temperature=temperature,
            )
        elif provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=model or GROQ_MODEL,
                api_key=GROQ_API_KEY,
                temperature=temperature,
            )
        else:
            raise ValueError(
                f"Unknown provider '{provider}'. Choose: gemini, openai or groq."
            )
    except Exception as e:
        raise RuntimeError(f"Failed to initialise {provider} model: {e}") from e


