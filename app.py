import os
import tempfile
import streamlit as st
from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Instead of your key here, use os.environ to keep it safe:
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# 2. Page Configuration
st.set_page_config(
    page_title="RAG Chatbot over PDFs", page_icon="🤖", layout="wide"
)

st.title("📄 RAG Chatbot over Your Own PDFs")
st.markdown(
    "Upload your study notes or documents, chunk them, embed them, and chat with an LLM backed by exact source citations[span_0](start_span)[span_0](end_span)!"
)

# 3. Sidebar for Document Upload
with st.sidebar:
  st.header("Upload Document")
  uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
  process_btn = st.button("Process Document")

# Initialize session state for vector store and chat history
if "vector_store" not in st.session_state:
  st.session_state.vector_store = None
if "chat_history" not in st.session_state:
  st.session_state.chat_history = []

# 4. Process Uploaded PDF
if process_btn:
  if not uploaded_file:
    st.error("Please upload a PDF file first.")
  else:
    with st.spinner("Processing PDF (Chunking & Embedding)..."):
      # Save uploaded file temporarily
      with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

      try:
        # Load PDF
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        # Chunk documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        # Initialize Embeddings (sentence-transformers)
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # Store in Vector DB (ChromaDB)
        vector_store = Chroma.from_documents(splits, embeddings)
        st.session_state.vector_store = vector_store

        st.success(
            f"Successfully processed {len(docs)} pages into {len(splits)} chunks!"
        )
      except Exception as e:
        st.error(f"An error occurred: {e}")
      finally:
        if os.path.exists(tmp_path):
          os.remove(tmp_path)

# 5. Main Chat Interface
if st.session_state.vector_store is not None:
  st.divider()
  st.subheader("Chat with your Document")

  # Display chat history
  for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

  # User Query Input
  query = st.chat_input(
      "Ask a question about your PDF (e.g., What are the main topics covered?)"
  )

  if query:
    # Append user message
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
      st.markdown(query)

    with st.spinner("Retrieving context and generating response..."):
      try:
        # Setup Retriever
        retriever = st.session_state.vector_store.as_retriever(
            search_kwargs={"k": 3}
        )

        # Initialize Gemini Client using your hardcoded key
        client = genai.Client(api_key=GEMINI_API_KEY)

        # Retrieve relevant chunks manually
        retrieved_docs = retriever.invoke(query)
        context_text = "\n\n".join(
            [doc.page_content for doc in retrieved_docs]
        )

        prompt = f"""
                You are an expert study assistant. Answer the user's question accurately using ONLY the provided context below. 
                If you do not know the answer based on the context, state clearly that it is not present in the documents.
                Always cite the source context when applicable.

                Context:
                {context_text}

                Question: {query}
                """

        # Call Gemini API
        response = client.models.generate_content(
            model="gemini-3.8-flash", contents=prompt
        )
        answer = response.text

        # Append assistant response
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )
        with st.chat_message("assistant"):
          st.markdown(answer)

          # Expandable section to inspect retrieved source chunks
          with st.expander("View Retrieved Source Chunks"):
            for i, doc in enumerate(retrieved_docs):
              st.markdown(f"**Chunk {i+1} (Page {doc.metadata.get('page', 'N/A')}):**")
              st.text(doc.page_content[:300] + "...")

      except Exception as e:
        st.error(f"Error generating response: {e}")
else:
  st.info("👈 Please upload a PDF and click 'Process Document' in the sidebar to start chatting.")