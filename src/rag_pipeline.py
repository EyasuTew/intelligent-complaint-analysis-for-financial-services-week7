# src/rag_pipeline.py
# Task 3: RAG Pipeline for Complaint Analysis
# This module implements the retriever, prompt engineering, and generator for the RAG system.
# Requires: pip install chromadb sentence-transformers langchain langchain-community transformers torch pandas
# Note: For LLM, using Hugging Face's transformers with Mistral-7B-Instruct-v0.1 (local inference; adjust for GPU/CPU).
#       Pre-built vector store: Load from complaint_embeddings.parquet and persist as ChromaDB.

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict, Any
import os

# Global setup
os.makedirs('./data/vector_store_full', exist_ok=True)  # For full dataset persistence

def load_and_index_vector_store(parquet_path: str = './data/complaint_embeddings.parquet', persist_path: str = './data/vector_store_full'):
    """
    Load pre-built embeddings from parquet and create/persist ChromaDB collection.
    Run once; subsequent loads use persistence.
    """
    print("Loading pre-built embeddings from parquet...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} chunks")

    # Embedder (for queries)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    # ChromaDB setup
    client = chromadb.PersistentClient(path=persist_path)
    collection_name = 'cfpb_full_complaints'

    try:
        collection = client.get_collection(name=collection_name)
        print(f"Loaded existing full collection with {collection.count} items.")
        return collection, embed_model
    except:
        print("Creating new full collection...")
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Prepare data from parquet (assumes columns: chunk_text, embedding, complaint_id, product_category, etc.)
        ids = [f"{row['complaint_id']}_chunk_{row['chunk_index']}" for _, row in df.iterrows()]
        embeddings = df['embedding'].tolist()  # Pre-computed vectors
        metadatas = [
            {
                'complaint_id': row['complaint_id'],
                'product_category': row['product_category'],
                'chunk_index': row['chunk_index'],
                'total_chunks': row['total_chunks'],
                'text_preview': row['chunk_text'][:100]
            }
            for _, row in df.iterrows()
        ]

        # Batch add
        batch_size = 5000  # Larger for full dataset
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )
            print(f"Indexed batch {i//batch_size + 1}/{(len(ids)-1)//batch_size + 1} ({len(batch_ids)} items)")

        print(f"Full collection ready: {collection.count} vectors.")
        return collection, embed_model

def retrieve_chunks(collection, embed_model, question: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Retriever: Embed question and fetch top-k chunks.
    """
    query_embedding = embed_model.encode([question])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=k,
        include=['metadatas', 'distances', 'documents']  # Assumes 'documents' for chunk_text if added; else use metadatas
    )
    # Note: If chunk_text not in metadatas, adjust to fetch from df or add during indexing
    retrieved = []
    for i in range(k):
        if i < len(results['metadatas'][0]):
            meta = results['metadatas'][0][i]
            meta['chunk_text'] = meta.get('text_preview', '') + '...'  # Placeholder; full text from parquet if needed
            meta['distance'] = results['distances'][0][i]
            retrieved.append(meta)
    return retrieved

# Prompt Template
PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. 
Use the following retrieved complaint excerpts to formulate your answer. Base your response only on this context. 
If the context doesn't contain the answer, state that you don't have enough information.

Context: {context}

Question: {question}

Answer:
"""
)

def generate_response(prompt, llm_pipeline) -> str:
    """
    Generator: Format prompt and generate with LLM.
    """
    formatted_prompt = prompt.format(**prompt.input_variables)  # Wait, no: use .format(context=..., question=...)
    # Actually, pass as string
    response = llm_pipeline(formatted_prompt, max_new_tokens=200, do_sample=True, temperature=0.7)[0]['generated_text']
    # Extract answer (post-process to trim prompt)
    answer = response.split("Answer:")[-1].strip()
    return answer

# LLM Setup (run once)
def setup_llm(model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"):
    """
    Initialize local LLM pipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False  # To get only generation
    )
    return llm_pipeline

# End-to-End RAG Function
def rag_query(question: str, collection, embed_model, llm_pipeline, k: int = 5) -> Dict[str, Any]:
    """
    Full RAG: Retrieve + Generate.
    """
    chunks = retrieve_chunks(collection, embed_model, question, k)
    context = "\n\n".join([chunk['chunk_text'] for chunk in chunks])
    
    prompt_str = PROMPT_TEMPLATE.format(context=context, question=question)
    answer = generate_response(prompt_str, llm_pipeline)
    
    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": chunks
    }

# Example Usage (for testing; integrate in notebook)
if __name__ == "__main__":
    # Load
    collection, embed_model = load_and_index_vector_store()
    llm = setup_llm()
    
    # Query
    result = rag_query("Why are people unhappy with Credit Cards?", collection, embed_model, llm)
    print(result["answer"])