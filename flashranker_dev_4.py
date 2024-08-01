import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json
import pandas as pd
import numpy as np
from flashrank import Ranker, RerankRequest

# Load cases data
with open("cases.txt", "r") as file:
    cases = json.loads(file.read())

# Setup Chroma client with new architecture
CHROMA_DATA_PATH = "./chroma_data"
COLLECTION_NAME = "demo_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

# Check if the collection exists
collections = client.list_collections()
collection_names = [col.name for col in collections]

if COLLECTION_NAME in collection_names:
    collection = client.get_collection(name=COLLECTION_NAME)
else:
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"},
    )

# Ask if there are new documents to add
add_new_docs = input("Are there any new documents to add? (yes/no): ").strip().lower()

if add_new_docs == 'yes':
    # Create embeddings
    embed_model = SentenceTransformer(EMBED_MODEL)
    embeddings_dict = {}
    for case in cases:
        case_id = case['id']
        content_embedding = embed_model.encode(case['text'])
        embeddings_dict[case_id] = content_embedding.tolist()

    # Add documents and embeddings to the collection 
    documents = [case["text"] for case in cases]
    ids = [str(case["id"]) for case in cases]
    embeddings = [embeddings_dict[case["id"]] for case in cases]

    collection.add(
        documents=documents,
        ids=ids,
        embeddings=embeddings,
        # metadatas=metadatas  # Add metadata if needed
    )
else:
    # Load existing documents and their embeddings from the collection
    stored_cases = collection.query(query_texts=[""], n_results=len(cases))
    result_docs = stored_cases['documents']
    result_ids = stored_cases['ids']

    # Flatten any nested lists if necessary
    if isinstance(result_ids[0], list):
        result_ids = [item for sublist in result_ids for item in sublist]
    if isinstance(result_docs[0], list):
        result_docs = [item for sublist in result_docs for item in sublist]

    # Create a dictionary from the retrieved cases
    result_cases = {key: value for key, value in zip(result_ids, result_docs)}

# Query the collection
query_texts = input("Ask a question: ")
query_results = collection.query(query_texts=query_texts, n_results=30)

# Extract the results
result_docs = query_results['documents']
result_ids = query_results['ids']

# Flatten any nested lists if necessary
if isinstance(result_ids[0], list):
    result_ids = [item for sublist in result_ids for item in sublist]
if isinstance(result_docs[0], list):
    result_docs = [item for sublist in result_docs for item in sublist]

# Create a DataFrame
df = pd.DataFrame({
    'id': result_ids,
    'document': result_docs
})

print(df)

# Create a list of tuples using zip()
tuples = [(key, value) for key, value in zip(result_ids, result_docs)]

# Convert list of tuples to dictionary using dict()
result_cases = dict(tuples)

print(result_cases)

# Function to rank retrieved cases
def ranker(query_texts, retrieved_cases):
    cases_to_rank = []
    for key, value in retrieved_cases.items():
        cases_to_rank.append({"id": key, "text": value})
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/opt")
    rerankrequest = RerankRequest(query=query_texts, passages=cases_to_rank)
    results = ranker.rerank(rerankrequest)

    extracted_data = [(result['id'], result['score']) for result in results]
    df = pd.DataFrame(extracted_data, columns=['Case ID', 'Score'])

    print(query_texts)
    print(df)

ranker(query_texts, result_cases)
