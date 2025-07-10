"""
GenAI Bedrock Knowledgebase Application
=====================================

This Streamlit application creates a local knowledgebase system that allows users to:
1. Upload documents (PDF, TXT, MD files)
2. Convert them into searchable vector embeddings using AWS Bedrock
3. Ask questions about the documents using natural language
4. Get AI-powered answers with source citations

Key Components:
- Document Processing: Extracts text from PDFs and processes text files
- Vector Storage: Uses ChromaDB to store document embeddings locally
- AI Integration: Uses AWS Bedrock Nova Micro LLM for question answering
- Web Interface: Streamlit provides an easy-to-use web interface

Prerequisites:
- AWS credentials configured with Bedrock access
- Python packages: streamlit, langchain, PyPDF2, chromadb, boto3
- .env file with AWS configuration (optional)

How to Use:
1. Run: streamlit run app.py
2. Upload documents via the "Upload Files" tab
3. Click "Re-index Knowledgebase" to process documents
4. Ask questions in the "Ask Questions" tab
5. Manage files in the "Delete Files" tab
"""

# --- IMPORTS ---
# Core Python libraries
import os
import shutil
import stat
import time
import tempfile
import atexit

# Streamlit for web interface
import streamlit as st

# Environment and PDF processing
from dotenv import load_dotenv
import PyPDF2

# LangChain components for AI and document processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# --- ENVIRONMENT SETUP ---
# Load environment variables from .env file (AWS credentials, etc.)
load_dotenv()

# --- CONFIGURATION CONSTANTS ---
# Directory where uploaded documents are stored
DATA_DIR = "data"

# Create a temporary directory for ChromaDB that gets cleaned up automatically
# This avoids permission issues and keeps the workspace clean
TEMP_DIR = tempfile.mkdtemp(prefix="knowledgebase_chromadb_")
CHROMA_DIR = os.path.join(TEMP_DIR, "chroma_db")

# Register cleanup function to remove temp directory when app exits
def cleanup_temp_dir():
    """Clean up temporary directory when application exits"""
    try:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR, ignore_errors=True)
            print(f"✅ Cleaned up temporary directory: {TEMP_DIR}")
    except Exception as e:
        print(f"⚠️ Could not clean up temporary directory: {e}")

atexit.register(cleanup_temp_dir)

# AWS Bedrock model IDs - these are the AI models we'll use
AWS_BEDROCK_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v1"  # For converting text to vectors
AWS_BEDROCK_LLM_MODEL_ID = "us.amazon.nova-micro-v1:0"          # For answering questions
AWS_REGION = "us-west-2"  # AWS region where Bedrock is available

# --- AI COMPONENTS INITIALIZATION ---
# Initialize the embedding model (converts text to numerical vectors for similarity search)
embedding_model = BedrockEmbeddings(
    region_name="us-west-2",
    model_id=AWS_BEDROCK_EMBEDDING_MODEL_ID
)

# Initialize text splitter (breaks documents into smaller chunks for processing)
# chunk_size=500: Each chunk is ~500 characters
# chunk_overlap=50: Overlapping chunks to maintain context
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# --- PDF PROCESSING FUNCTIONS ---
def extract_text_from_pdf(pdf_path):
    """
    Extract text content from a PDF file using PyPDF2
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text content, or None if extraction fails
    """
    text = ""
    
    try:
        # Open PDF file in binary read mode
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Loop through each page in the PDF
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"  # Add double newline between pages
                    
    except Exception as e:
        st.error(f"Error extracting text from PDF {pdf_path}: {e}")
        return None
    
    return text.strip()

def convert_pdf_to_text(uploaded_file, filename):
    """
    Convert an uploaded PDF file to text and save it as a .txt file
    
    Args:
        uploaded_file: Streamlit uploaded file object
        filename (str): Original filename of the uploaded PDF
        
    Returns:
        tuple: (text_filename, text_length) or (None, 0) if conversion fails
    """
    try:
        # Step 1: Save the uploaded PDF temporarily
        temp_pdf_path = os.path.join(DATA_DIR, f"temp_{filename}")
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Step 2: Extract text from the temporary PDF
        text = extract_text_from_pdf(temp_pdf_path)
        
        # Step 3: Clean up the temporary PDF file
        os.remove(temp_pdf_path)
        
        if text:
            # Step 4: Save extracted text as a .txt file
            text_filename = filename.replace('.pdf', '.txt')
            text_path = os.path.join(DATA_DIR, text_filename)
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text)
            return text_filename, len(text)
        else:
            return None, 0
            
    except Exception as e:
        st.error(f"Error processing PDF {filename}: {e}")
        # Clean up temporary file if something went wrong
        temp_pdf_path = os.path.join(DATA_DIR, f"temp_{filename}")
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
        return None, 0

# --- DATABASE MANAGEMENT FUNCTIONS ---
def create_new_chroma_dir():
    """
    Create a new ChromaDB directory in the temp space
    
    Returns:
        str: Path to the new ChromaDB directory
    """
    # Create a unique subdirectory for this ChromaDB instance
    timestamp = int(time.time())
    new_chroma_dir = os.path.join(TEMP_DIR, f"chroma_db_{timestamp}")
    os.makedirs(new_chroma_dir, exist_ok=True)
    return new_chroma_dir

def safe_remove_directory(directory):
    """
    Safely remove a directory with proper error handling
    With temp directories, this should be much simpler and more reliable
    
    Args:
        directory (str): Path to directory to remove
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(directory):
        return True
    
    try:
        # Since we're using temp directories, removal should be straightforward
        shutil.rmtree(directory, ignore_errors=True)
        return not os.path.exists(directory)
    except Exception as e:
        print(f"Error removing directory {directory}: {e}")
        return False

# --- CORE KNOWLEDGEBASE FUNCTION ---
def reindex_knowledgebase():
    """
    Re-index the knowledgebase by processing all documents in the data directory
    This is the core function that:
    1. Reads all text files from the data directory
    2. Splits them into chunks
    3. Converts chunks to vector embeddings
    4. Stores embeddings in ChromaDB for similarity search
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    # Step 2: Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Step 3: Process all documents in the data directory
    docs = []  # List to store all document chunks
    processed_files = []  # List to track which files were processed
    
    # Loop through all files in the data directory
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt") or filename.endswith(".md"):
            try:
                # Read the file content
                with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8") as f:
                    text = f.read()
                    
                    if text.strip():  # Only process non-empty files
                        # Split text into chunks using the text splitter
                        splits = text_splitter.create_documents([text])
                        docs.extend(splits)
                        processed_files.append(filename)
                    else:
                        st.warning(f"Skipping empty file: {filename}")
                        
            except Exception as e:
                st.error(f"Error reading file {filename}: {e}")
                continue
    
    # Step 4: Check if we have any documents to process
    if not docs:
        st.error("No valid documents found to index.")
        return False
    
    try:
        print(f"📁 Found {len(processed_files)} file(s) in data directory:")
        
        # Step 5: Create a new ChromaDB directory in temp space
        new_chroma_dir = create_new_chroma_dir()
        print(f"✅ Created new vectorstore directory: {new_chroma_dir}")

        # Step 6: Create new vectorstore with embeddings
        print(f"📁 Creating new vectorstore in temporary directory")
        # This is where the magic happens - documents are converted to vectors
        st.session_state.vectorstore = Chroma.from_documents(
            docs,  # The document chunks
            embedding_model,  # The embedding model (converts text to vectors)
            persist_directory=new_chroma_dir  # Where to save the vector database (temp dir)
        )
        print(f"✅ Created new vectorstore in {new_chroma_dir}")

        # Step 7: Update global variable to point to new location
        globals()['CHROMA_DIR'] = new_chroma_dir
        
        # Step 8: Show success messages
        st.success(f"✅ Processed {len(processed_files)} files: {', '.join(processed_files)}")
        st.success(f"✅ Created {len(docs)} document chunks")
        st.success(f"✅ Database saved to temporary directory")
        
        # Step 9: Test the vectorstore to make sure it works
        try:
            test_results = st.session_state.vectorstore.similarity_search("test", k=1)
            st.success(f"✅ Vectorstore test successful - found {len(test_results)} results")
        except Exception as e:
            st.error(f"Vectorstore test failed: {e}")
            return False

        # Step 10: Celebrate success!
        time.sleep(2)
        st.balloons()
        
        return True
        
    except Exception as e:
        st.error(f"Error creating vectorstore: {e}")
        st.error("Please check your AWS credentials and Bedrock access.")
        return False

# --- STREAMLIT APPLICATION SETUP ---
# Configure the main page
st.set_page_config(page_title="🧠 GenAI Bedrock Knowledgebase")
st.title("🧠 GenAI Bedrock Knowledgebase")

# --- NAVIGATION SIDEBAR ---
# Initialize session state for page navigation (remembers which page user is on)
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Upload Files"

# Initialize vectorstore loading state
if 'vectorstore_loaded' not in st.session_state:
    st.session_state.vectorstore_loaded = False

st.sidebar.title("🧭 Navigation")

# Create navigation buttons for different pages
if st.sidebar.button("💬 Ask Questions", use_container_width=True):
    st.session_state.current_page = "Ask Questions"

if st.sidebar.button("📤 Upload Files / Re-Index", use_container_width=True):
    st.session_state.current_page = "Upload Files"

if st.sidebar.button("🗑️ Delete Files", use_container_width=True):
    st.session_state.current_page = "Delete Files"

# Get the current page from session state
page = st.session_state.current_page

# --- PAGE 1: UPLOAD FILES ---
if page == "Upload Files":
    st.header("📤 Upload Files")
    
    # Instructions for users
    st.info("💡 **Note**: After uploading files, click 'Re-index Knowledgebase' to make them searchable.")

    # Re-index button (this is the most important button!)
    if st.button("🔄 Re-index Knowledgebase"):
        with st.spinner("Re-indexing..."):
            success = reindex_knowledgebase()
        if success:
            st.success("Knowledgebase re-indexed successfully!")
            st.session_state.vectorstore_loaded = True
            # Force reload of vectorstore in session state
            st.session_state.vectorstore_initialized = True
            print(f"📁 Vectorstore initialized in session state: {st.session_state.vectorstore_initialized}")
            st.rerun()  # Refresh the page to show updated status
        else:
            st.warning("No documents found to index or indexing failed.")
            st.session_state.vectorstore_loaded = False

    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # File upload widget
    accepted_types = [".txt", ".md", ".pdf"]
    uploaded_files = st.file_uploader(
        f"Upload {', '.join(accepted_types).upper()} files",
        type=accepted_types,
        accept_multiple_files=True
    )

    # Show existing files in the data directory
    existing_files = [f for f in os.listdir(DATA_DIR) if f.endswith((".txt", ".md"))]
    if existing_files:
        st.subheader("📋 Current Files")
        for file in existing_files:
            st.write(f"• {file}")
    
    # Process uploaded files
    if uploaded_files:
        for file in uploaded_files:
            try:
                if file.name.endswith('.pdf'):
                    # Handle PDF files - convert to text
                    with st.spinner(f"Processing PDF: {file.name}..."):
                        text_filename, text_length = convert_pdf_to_text(file, file.name)
                        if text_filename:
                            st.success(f"✅ Converted {file.name} to {text_filename} ({text_length} characters)")
                        else:
                            st.error(f"❌ Failed to extract text from {file.name}")
                else:
                    # Handle text/markdown files - save directly
                    save_path = os.path.join(DATA_DIR, file.name)
                    with open(save_path, "wb") as f:
                        f.write(file.read())
                    st.success(f"✅ Saved {file.name} to /data")
                    
            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {e}")

# --- PAGE 2: DELETE FILES ---
elif page == "Delete Files":
    st.header("🗑️ Delete Files")
    
    # Show current knowledgebase status
    if st.session_state.vectorstore_loaded:
        st.info("✅ Knowledgebase is loaded and ready to use")
    else:
        st.warning("⚠️ No knowledgebase found.")
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Get list of files
    files = [f for f in os.listdir(DATA_DIR) if f.endswith((".txt", ".md"))]
    
    if not files:
        st.info("📁 No files found in the data directory.")
    else:
        st.subheader("📋 Available Files")
        
        # Show files with checkboxes for selection
        selected_files = st.multiselect(
            "Select files to delete:",
            options=files,
            help="Choose one or more files to delete from your knowledgebase"
        )
        
        # Show selected files for confirmation
        if selected_files:
            st.write("**Files selected for deletion:**")
            for file in selected_files:
                file_type = "📄 PDF→TXT" if file.endswith('.txt') and file.replace('.txt', '.pdf') else "📝 TEXT"
                st.write(f"• ❌ {file} ({file_type})")
            
            # Confirmation button
            if st.button("🗑️ Delete Selected Files", type="primary"):
                deleted_count = 0
                for f in selected_files:
                    try:
                        file_path = os.path.join(DATA_DIR, f)
                        os.remove(file_path)
                        st.success(f"✅ Deleted: {f}")
                        deleted_count += 1
                    except Exception as e:
                        st.error(f"❌ Error deleting {f}: {e}")
                
                if deleted_count > 0:
                    st.success(f"🎉 Successfully deleted {deleted_count} file(s)")
                    st.balloons()
                    
                    # Auto-refresh the page to update file list
                    time.sleep(1)
                    st.rerun()
        else:
            st.info("👆 Select files above to delete them")
        
        # Show all current files for reference
        st.markdown("---")
        st.subheader("📂 All Current Files")
        for i, file in enumerate(files, 1):
            file_path = os.path.join(DATA_DIR, file)
            try:
                file_size = os.path.getsize(file_path)
                file_size_kb = file_size / 1024
                # Indicate if file was converted from PDF
                file_type = " (PDF→TXT)" if file.endswith('.txt') and any(
                    orig_name.replace('.pdf', '.txt') == file 
                    for orig_name in os.listdir(DATA_DIR) 
                    if orig_name.endswith('.pdf')
                ) else ""
                st.write(f"{i}. **{file}**{file_type} ({file_size_kb:.1f} KB)")
            except:
                st.write(f"{i}. **{file}**")
    
    # Reminder about re-indexing
    st.markdown("---")
    st.info("💡 **Note**: After deleting files, click 'Re-index Knowledgebase' to update the search index.")
    
    # Re-index button
    if st.button("🔄 Re-index Knowledgebase"):
        with st.spinner("Re-indexing..."):
            success = reindex_knowledgebase()
        if success:
            st.success("Knowledgebase re-indexed successfully!")
            st.session_state.vectorstore_loaded = True
            st.rerun()  # Refresh the page to show updated status
        else:
            st.warning("No documents found to index or indexing failed.")
            st.session_state.vectorstore_loaded = False

# --- PAGE 3: ASK QUESTIONS ---
elif page == "Ask Questions":
    st.header("💬 Ask Questions")

    # Check if vectorstore is loaded and functional
    if 'vectorstore' not in st.session_state or not st.session_state.vectorstore_loaded:
        st.warning("⚠️ Knowledgebase is empty or failed to load. Upload files and re-index first.")

        # Show helpful info about current state
        os.makedirs(DATA_DIR, exist_ok=True)
        files = [f for f in os.listdir(DATA_DIR) if f.endswith((".txt", ".md"))]

        if files:
            st.info(f"📁 Found {len(files)} file(s) in data directory:")
            st.info("👆 Click 'Re-index Knowledgebase' on the Upload Files page to make these files searchable.")
        else:
            st.info("📁 No files found. Go to 'Upload Files' to add documents first.")

    else:
        try:
            # Step 1: Set up retriever (top 3 chunks)
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

            # Step 2: Initialize the Bedrock LLM (completion-style)
            from langchain_community.llms import Bedrock

            llm = Bedrock(
                region_name=AWS_REGION,
                model_id=AWS_BEDROCK_LLM_MODEL_ID,
                model_kwargs={
                    "temperature": 0,
                    "max_tokens": 1000
                }
            )

            # Step 3: Text input for question
            question = st.text_input("Enter your question:")

            if st.button("Generate Answer") and question:
                with st.spinner("Retrieving and generating answer..."):
                    # Retrieve relevant documents
                    docs = retriever.get_relevant_documents(question)
                    context = "\n\n".join([doc.page_content for doc in docs])

                    # Build prompt
                    prompt = f"""
You are an AI assistant helping answer questions based on the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

                    # Generate completion
                    answer = llm(prompt)

                    # Display answer
                    st.markdown("### 💬 Answer")
                    st.write(answer)

                    # Display sources
                    st.markdown("### 📄 Retrieved Context")
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Source {i+1}:**")
                        st.write(doc.page_content[:500] + "...")
        except Exception as e:
            st.error(f"Error during question answering: {e}")
            st.info("Please re-index your knowledgebase.")
            st.session_state.vectorstore_loaded = False
