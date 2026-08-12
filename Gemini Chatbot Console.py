# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:34:56 2024

@author: shikh
"""

import google.generativeai as genai
import os
import pickle
import time
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Set the API key for Google Generative AI
api_key = os.environ.get("API_KEY") or os.environ.get("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    # Fallback/Placeholder so the user can easily paste it here if they don't want to use environment variables.
    api_key = ""  # ENTER_YOUR_API_KEY_HERE
    if api_key:
        genai.configure(api_key=api_key)
    else:
        print("Warning: API_KEY/GEMINI_API_KEY environment variable is not set.")

# Load the models
generation_model = None  # Will be resolved dynamically
EMBED_MODEL = "models/embedding-001"  # Default fallback, will be resolved dynamically

def find_embedding_model():
    """Find a supported embedding model from the API, defaulting to models/embedding-001."""
    try:
        models = genai.list_models()
        embed_models = [m.name for m in models if 'embedContent' in m.supported_generation_methods]
        # Prefer models/text-embedding-004 first, then models/embedding-001
        for model_name in ["models/text-embedding-004", "models/embedding-001"]:
            if model_name in embed_models:
                return model_name
        if embed_models:
            return embed_models[0]
    except Exception as e:
        print(f"Warning: Could not list models programmatically ({e}). Defaulting to models/embedding-001.")
    return "models/embedding-001"

def find_generation_model():
    """Find a supported text generation model, preferring gemini-1.5-flash, then gemini-pro/gemini-1.0-pro."""
    try:
        models = genai.list_models()
        gen_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        gen_models_clean = [name.replace("models/", "") for name in gen_models]
        for preferred in ["gemini-1.5-flash", "gemini-pro", "gemini-1.0-pro"]:
            if preferred in gen_models_clean:
                return preferred
        if gen_models_clean:
            return gen_models_clean[0]
    except Exception as e:
        print(f"Warning: Could not list generation models ({e}). Defaulting to gemini-1.5-flash.")
    return "gemini-1.5-flash"
VECTOR_DB_FILE = "fiqa_vector_db.pkl"
CORPUS_SIZE = 200

def get_embedding_with_retry(text, task_type="retrieval_document", retries=3):
    """Generate embedding for a single text chunk with retries for rate limits."""
    if not text.strip():
        return None
    for i in range(retries):
        try:
            result = genai.embed_content(
                model=EMBED_MODEL,
                content=text,
                task_type=task_type
            )
            return result['embedding']
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "quota" in err_msg.lower():
                time.sleep(2 ** i + 1)
            else:
                print(f"Embedding error: {e}")
                return None
    return None

def load_or_create_vector_db():
    """Load cached vector DB or download FiQA corpus, embed it, and cache it."""
    if os.path.exists(VECTOR_DB_FILE):
        print("Loading cached vector database...")
        with open(VECTOR_DB_FILE, 'rb') as f:
            db = pickle.load(f)
        print(f"Loaded {len(db['documents'])} documents from cache.")
        return db

    print("Cache not found. Loading FiQA dataset from Hugging Face...")
    # Load FiQA corpus config
    dataset = load_dataset('BeIR/fiqa', 'corpus', split=f'corpus[:{CORPUS_SIZE}]')
    
    documents = []
    texts_to_embed = []
    for doc in dataset:
        title = doc.get('title', '').strip()
        text = doc.get('text', '').strip()
        # Combine title and text for embedding context if title exists
        full_text = f"{title}\n{text}" if title else text
        
        documents.append({
            'id': doc['_id'],
            'title': title,
            'text': text
        })
        texts_to_embed.append(full_text)

    print(f"Generating embeddings for {len(texts_to_embed)} documents in parallel...")
    embeddings = [None] * len(texts_to_embed)
    
    def embed_worker(item):
        idx, text = item
        embeddings[idx] = get_embedding_with_retry(text, task_type="retrieval_document")

    # Use ThreadPoolExecutor to run embedding calls in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        list(tqdm(executor.map(embed_worker, enumerate(texts_to_embed)), total=len(texts_to_embed), desc="Embedding Progress"))

    # Filter out any documents that failed to embed
    valid_documents = []
    valid_embeddings = []
    for doc, emb in zip(documents, embeddings):
        if emb is not None:
            valid_documents.append(doc)
            valid_embeddings.append(emb)

    if len(valid_embeddings) == 0:
        raise ValueError(
            "Failed to generate embeddings for any documents. "
            "Please check if your API key is correct and valid."
        )

    valid_embeddings = np.array(valid_embeddings, dtype=np.float32)
    
    # Normalize embeddings for fast cosine similarity via dot product
    norms = np.linalg.norm(valid_embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    normalized_embeddings = valid_embeddings / norms

    db = {
        'documents': valid_documents,
        'embeddings': normalized_embeddings
    }

    print(f"Saving vector database to {VECTOR_DB_FILE}...")
    with open(VECTOR_DB_FILE, 'wb') as f:
        pickle.dump(db, f)
    print(f"Successfully indexed and cached {len(valid_documents)} documents.")
    return db

def retrieve_relevant_docs(query, db, top_k=3):
    """Retrieve top k documents matching the query."""
    query_emb = get_embedding_with_retry(query, task_type="retrieval_query")
    if query_emb is None:
        return []
    
    query_emb = np.array(query_emb, dtype=np.float32)
    q_norm = np.linalg.norm(query_emb)
    if q_norm > 0:
        query_emb = query_emb / q_norm
        
    # Calculate similarity scores
    similarities = np.dot(db['embeddings'], query_emb)
    
    # Get top k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'document': db['documents'][idx],
            'score': similarities[idx]
        })
    return results

def chatbot():
    global EMBED_MODEL, generation_model
    print("\nInitializing RAG Chatbot with FiQA financial dataset...")
    EMBED_MODEL = find_embedding_model()
    print(f"Using embedding model: {EMBED_MODEL}")
    
    gen_model_name = find_generation_model()
    print(f"Using generation model: {gen_model_name}")
    generation_model = genai.GenerativeModel(gen_model_name)
    try:
        db = load_or_create_vector_db()
    except Exception as e:
        print(f"Error initializing vector database: {e}")
        print("Please ensure your API_KEY is set correctly and you have internet access.")
        return

    print("\nWelcome to the Financial RAG Chatbot! Type 'exit' to end the conversation.")
    print("Example questions: 'What are the main risks of day trading?', 'How do interest rates affect bonds?'")
    
    conversation_summary = ""  # To keep track of conversation context

    while True:
        try:
            user_input = input("\nYou: ")
        except (KeyboardInterrupt, EOFError):
            print("\nConversation Ended. Have a good day!")
            break

        if user_input.strip().lower() == "exit":
            print("Conversation Ended. Have a good day!")
            break

        if not user_input.strip():
            continue

        # 1. Retrieve relevant contexts
        print("[Retrieving relevant financial documents...]")
        retrieved = retrieve_relevant_docs(user_input, db, top_k=3)
        
        context_str = ""
        sources_str = ""
        for i, item in enumerate(retrieved):
            doc = item['document']
            score = item['score']
            title_prefix = f"Title: {doc['title']}\n" if doc['title'] else ""
            context_str += f"--- Document {i+1} (Similarity: {score:.3f}) ---\n{title_prefix}{doc['text']}\n\n"
            sources_str += f"- [Doc ID: {doc['id']}] {doc['title'] if doc['title'] else doc['text'][:60]}... (Similarity: {score:.3f})\n"

        # 2. Construct RAG prompt
        rag_prompt = (
            f"You are a helpful financial assistant. Answer the user's question using ONLY the provided financial contexts. "
            f"If the answer cannot be found or inferred from the contexts, explain that you do not have enough information. "
            f"Always keep your answer professional, accurate, and concise.\n\n"
            f"Contexts:\n{context_str}\n"
            f"Conversation Summary So Far: {conversation_summary}\n\n"
            f"User's Question: {user_input}\n"
            f"Answer:"
        )

        # 3. Generate response using Gemini
        print("[Generating response...]")
        responses = generation_model.generate_content(rag_prompt)

        if responses and responses.candidates:
            response = responses.candidates[0].content.parts[0].text
            print(f"\nGemini: {response}")
            
            # Print references
            if sources_str:
                print("\nSources retrieved:")
                print(sources_str)

            # Update conversation context
            summary_prompt = f"Summarize the conversation so far, including the latest question: '{user_input}' and answer: '{response}'"
            try:
                summary_response = generation_model.generate_content(summary_prompt)
                if summary_response and summary_response.candidates:
                    conversation_summary = summary_response.candidates[0].content.parts[0].text
            except Exception as e:
                # Fallback if summary generation fails or rate limited
                conversation_summary = f"{conversation_summary} Q: {user_input} A: {response}"

if __name__ == "__main__":
    chatbot()
