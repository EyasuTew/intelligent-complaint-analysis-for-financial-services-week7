# app.py
# Task 4: Interactive RAG Chatbot UI with Gradio
# Run with: python app.py (opens at http://127.0.0.1:7860)
# Requires: pip install gradio chromadb sentence-transformers transformers torch langchain langchain-community pandas
# Note: Assumes src/rag_pipeline.py from Task 3 is available. For full run, ensure vector_store_full/ is built.
# LLM: Uses Mistral-7B; for faster demo, swap to 'gpt2' or API-based (e.g., via langchain_openai).

import gradio as gr
from src.rag_pipeline import load_and_index_vector_store, rag_query, setup_llm  # From Task 3
import os

# Global setup (run once)
print("Initializing RAG components...")
collection, embed_model = load_and_index_vector_store(parquet_path='./complaint_embeddings.parquet', persist_path='./vector_store_full')
llm_pipeline = setup_llm()  # Or use a lighter model for testing

def chat_interface(question, history):
    """
    Gradio chat handler: Process query, generate response, return formatted output.
    """
    if not question.strip():
        return history, "", ""  # No-op if empty

    # Run RAG
    result = rag_query(question, collection, embed_model, llm_pipeline)
    
    # Format answer and sources
    answer = result["answer"]
    sources = []
    for chunk in result["retrieved_chunks"]:
        source_text = f"Source: Complaint {chunk['complaint_id']} (Chunk {chunk['chunk_index']}/{chunk['total_chunks']}, Product: {chunk['product_category']})\nText: {chunk['chunk_text']}\nDistance: {chunk['distance']:.3f}\n---"
        sources.append(source_text)
    sources_text = "\n\n".join(sources)
    
    # Append to history
    new_history = history + [(question, answer)]
    
    # Clear input
    return new_history, "", sources_text

# Gradio Interface
with gr.Blocks(title="CrediTrust Complaint Insights Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# CrediTrust RAG Chatbot\nAsk questions about customer complaints, e.g., 'Why are people unhappy with Credit Cards?'")
    
    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(
        label="Your Question",
        placeholder="Type your question here...",
        scale=4
    )
    submit_btn = gr.Button("Ask", variant="primary", scale=1)
    clear_btn = gr.Button("Clear", variant="secondary", scale=1)
    
    # Sources display (below chatbot)
    sources = gr.Markdown(label="Retrieved Sources", visible=False)
    
    # Event handlers
    def submit_message(msg, history):
        history, _, sources_text = chat_interface(msg, history)
        sources.update(visible=True)
        return history, "", sources_text
    
    msg.submit(submit_message, [msg, chatbot], [chatbot, msg, sources])
    submit_btn.click(submit_message, [msg, chatbot], [chatbot, msg, sources])
    clear_btn.click(lambda: ([], "", ""), outputs=[chatbot, msg, sources])
    
    # Optional: Streaming (commented; enable by replacing with generator func)
    # def stream_response(question, history):
    #     # Yield tokens; requires LLM to support streaming (e.g., via langchain)
    #     yield ...  # Placeholder for token-by-token

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)