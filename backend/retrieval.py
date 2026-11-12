"""PDF ingestion and stable per-chunk IDs, with idempotent index reuse."""
import hashlib
import json
import os
import threading

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langsmith import traceable
from pypdf import PdfReader

from .settings import ROOT

INDEX_LOCK = threading.Lock()
PDFS = {'policy': ('tenants_rights.pdf', 2000, 200), 'cases': ('previous_judgements.pdf', 1150, 50)}


def embedding_model():
    return OpenAIEmbeddings(model=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'), request_timeout=60, max_retries=1)


def fingerprint():
    digest = hashlib.sha256()
    for filename, size, overlap in PDFS.values():
        path = ROOT / 'data' / filename
        if not path.is_file():
            raise ValueError(f'Add {filename} to the app data/ folder before building the index.')
        digest.update(path.read_bytes())
        digest.update(f'{filename}:{size}:{overlap}'.encode())
    digest.update(os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small').encode())
    digest.update(os.getenv('OPENAI_BASE_URL', '').encode())
    return digest.hexdigest()[:20]


def index_ready():
    try:
        return (ROOT / '.runtime' / fingerprint() / 'manifest.json').exists()
    except ValueError:
        return False


def collection(kind, signature):
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=str(ROOT / '.runtime' / signature), settings=Settings(anonymized_telemetry=False))
    return client.get_or_create_collection(f'tenantflow_{kind}')


@traceable(name='build_pdf_indexes')
def build_indexes():
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError('Set OPENAI_API_KEY in .env before building the index.')
    with INDEX_LOCK:
        signature = fingerprint()
        manifest = ROOT / '.runtime' / signature / 'manifest.json'
        if manifest.exists():
            return json.loads(manifest.read_text()) | {'reused': True}
        counts = {}
        embedder = embedding_model()
        for kind, (filename, size, overlap) in PDFS.items():
            reader = PdfReader(str(ROOT / 'data' / filename))
            pages = [Document(page_content=page.extract_text() or '', metadata={'source': filename, 'page': i + 1}) for i, page in enumerate(reader.pages)]
            chunks = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap).split_documents(pages)
            chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
            if not chunks:
                raise ValueError(f'{filename} has no extractable text. Use a text PDF or OCR it first.')
            store = collection(kind, signature)
            for start in range(0, len(chunks), 64):
                batch = chunks[start:start + 64]
                texts = [d.page_content for d in batch]
                ids = [hashlib.sha256(f'{filename}:{d.metadata["page"]}:{start+i}:{d.page_content}'.encode()).hexdigest() for i, d in enumerate(batch)]
                store.upsert(ids=ids, documents=texts, metadatas=[d.metadata for d in batch], embeddings=embedder.embed_documents(texts))
            counts[kind] = store.count()
        manifest.write_text(json.dumps(counts))
        return counts | {'reused': False}


def search(kind, query, k):
    if not index_ready():
        raise ValueError('Build the PDF index before starting a live run.')
    store = collection(kind, fingerprint())
    count = store.count()
    if not count:
        return []
    results = store.query(query_embeddings=[embedding_model().embed_query(query)], n_results=min(k, count), include=['documents', 'metadatas', 'distances'])
    return [
        {'id': chunk_id, 'text': text, 'source': metadata['source'], 'page': metadata['page'], 'distance': distance}
        for chunk_id, text, metadata, distance in zip(results['ids'][0], results['documents'][0], results['metadatas'][0], results['distances'][0])
    ]
