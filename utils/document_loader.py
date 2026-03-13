from config.config import CHUNK_SIZE, CHUNK_OVERLAP

def load_documents(data_dir: str) -> list:
    from pathlib import Path
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader

    # TextLoader handles both .txt and .md; CSVLoader for tabular data
    loader_map = {
        ".pdf": lambda p: PyPDFLoader(p),
        ".txt": lambda p: TextLoader(p, encoding="utf-8"),
        ".md":  lambda p: TextLoader(p, encoding="utf-8"),
        ".csv": lambda p: CSVLoader(p, encoding="utf-8"),
    }

    docs = []
    try:
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"[DocumentLoader] Data directory not found: {data_dir}")
            return []

        for file_path in sorted(data_path.rglob("*")):
            suffix = file_path.suffix.lower()
            if file_path.is_file() and suffix in loader_map:
                try:
                    loader = loader_map[suffix](str(file_path))
                    loaded = loader.load()
                    # Tag each document with its source file name
                    for doc in loaded:
                        doc.metadata.setdefault("source", str(file_path))
                    docs.extend(loaded)
                    print(f"[DocumentLoader] Loaded {len(loaded)} page(s) from '{file_path.name}'")
                except Exception as e:
                    print(f"[DocumentLoader] Warning – could not load '{file_path.name}': {e}")
    except Exception as e:
        print(f"[DocumentLoader] Error scanning '{data_dir}': {e}")

    return docs

def chunk_documents(docs: list) -> list:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        print(f"[DocumentLoader] Created {len(chunks)} chunks from {len(docs)} document page(s).")
        return chunks
    except Exception as e:
        print(
            f"[DocumentLoader] Chunking error: {e}. "
            "Install or update 'langchain-text-splitters' and retry."
        )
        return docs

def load_and_chunk(data_dir: str) -> list:
    """Convenience wrapper: load all documents then chunk them in one call."""
    docs = load_documents(data_dir)
    if not docs:
        print("[DocumentLoader] No documents found – knowledge base will be empty.")
        return []
    return chunk_documents(docs)
