import streamlit as st
import os
import tempfile # To handle uploaded files safely
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
from transformers import pipeline

# --- PDF Text Extraction ---
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text: # Check if text extraction returned something
                    text += page_text + "\n" # Add newline between pages
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None
    # Clean up excessive whitespace
    cleaned_text = ' '.join(text.split())
    return cleaned_text

# --- Document Indexing ---
@st.cache_resource # Cache the embedding model
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data # Cache the indexed data based on file content
def index_documents(texts):
    model = get_embedding_model()
    embeddings = model.encode(texts)
    if not embeddings.any(): # Check if embeddings are empty
        return None, None
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype('float32')) # Ensure float32 for FAISS
    return index, texts # Return texts along with index

# --- QA Systems ---
@st.cache_resource # Cache the Groq client
def get_groq_client(api_key):
     # Basic check for Groq key format
    if not api_key or not api_key.startswith("gsk_"):
         st.error("Invalid Groq API Key format detected in Secrets.")
         return None
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {e}")
        return None

@st.cache_resource # Cache the HF pipeline
def get_hf_pipeline():
    try:
        # Using a specific revision can sometimes help with consistency
        return pipeline("question-answering", model="deepset/roberta-base-squad2")
    except Exception as e:
        st.error(f"Failed to load Hugging Face model: {e}")
        return None

def answer_with_groq(client, context, question, model="llama3-8b-8192"):
    if not client:
        return "Groq client not initialized."
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"You are an assistant that answers questions based ONLY on the provided context. If the answer is not in the context, say 'The answer is not found in the provided document.'. Context: {context}"},
                {"role": "user", "content": question}
            ],
            model=model,
            temperature=0.3,
            max_tokens=250 # Increased slightly
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Groq API error: {e}"

def answer_with_hf(hf_pipeline, context, question):
    if not hf_pipeline:
        return "Hugging Face model not loaded."
    try:
        result = hf_pipeline(question=question, context=context)
        score = result.get('score', 1.0)
        if score < 0.1: # Low confidence threshold
             return "I found some relevant text, but I'm not confident about the specific answer."
        return result['answer']
    except Exception as e:
        st.warning(f"Hugging Face model error: {e}. The context might be too long or complex for this model.")
        return None # Indicate HF failed

# --- Streamlit UI ---
st.set_page_config(page_title="PDF Q&A Bot", page_icon="📘")
st.title("📘 PDF Document Q&A Bot")
st.markdown("Upload a PDF document and ask questions about its content.")

# Get Groq API Key from Streamlit Secrets
groq_api_key = st.secrets.get("GROQ_API_KEY")

# Check if the key exists in secrets
if not groq_api_key:
    st.error("Groq API Key not found. Please add it to your Streamlit Secrets.")
    st.stop() # Stop the app if key is missing

# Initialize clients/models
groq_client = get_groq_client(groq_api_key)
hf_pipeline = get_hf_pipeline()

# Check if models loaded successfully
if not groq_client or not hf_pipeline:
    st.error("One or more AI models failed to initialize. Please check the logs or API keys.")
    st.stop()

uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

# Initialize session state
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

# Process PDF
if uploaded_file is not None:
    if st.session_state.uploaded_filename != uploaded_file.name:
        st.info(f"Processing '{uploaded_file.name}'...")
        # Use tempfile for secure handling
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        full_text = extract_text_from_pdf(tmp_file_path)
        os.remove(tmp_file_path) # Clean up

        if full_text:
            # Simple chunking logic
            chunk_size = 800
            overlap = 100
            chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size - overlap)]
            
            st.session_state.faiss_index, st.session_state.indexed_docs = index_documents(chunks)
            st.session_state.uploaded_filename = uploaded_file.name
            
            if st.session_state.faiss_index:
                 st.success(f"✅ Indexed '{uploaded_file.name}' ({len(chunks)} chunks). Ready!")
            else:
                 st.error("❌ Failed to create index. The document might have too little text.")
        else:
             st.error("❌ Failed to extract text from PDF.")
             st.session_state.faiss_index = None # Reset state
             st.session_state.indexed_docs = None
             st.session_state.uploaded_filename = None

# Show question input only if ready
if st.session_state.faiss_index is not None:
    question = st.text_input("Ask a question about the document:")

    if st.button("Get Answer") and question:
        embedding_model = get_embedding_model()
        query_embedding = embedding_model.encode([question])
        
        try:
            # Search FAISS
            distances, indices = st.session_state.faiss_index.search(query_embedding.astype('float32'), k=3) # Retrieve top 3 chunks
            
            relevant_chunks = [st.session_state.indexed_docs[i] for i in indices[0]]
            context = "\n\n---\n\n".join(relevant_chunks)
            
            if not context:
                 st.warning("Could not find relevant sections for your question.")
            else:
                 st.markdown("### 💡 Answer:")
                 with st.spinner("Asking Groq..."):
                      answer = answer_with_groq(groq_client, context, question)

                 # Fallback logic
                 if not answer or "not found in the provided document" in answer or "Groq API error" in answer:
                      st.info("Trying the Hugging Face model as a fallback...")
                      with st.spinner("Asking Hugging Face..."):
                           hf_answer = answer_with_hf(hf_pipeline, context, question)
                      
                      if hf_answer:
                           st.write(hf_answer)
                      elif answer and "Groq API error" not in answer: # Show Groq's "not found" if HF also fails
                            st.write(answer)
                      else:
                           st.error("Sorry, I encountered an error answering your question.")
                 else:
                      st.write(answer)

                 # Show context
                 with st.expander("Show Context Used"):
                      st.text(context)

        except Exception as e:
            st.error(f"An error occurred: {e}")
