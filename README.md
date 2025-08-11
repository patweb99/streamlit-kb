# 🧠 GenAI Bedrock with ChromaDB Knowledgebase (Strands Edition)

A powerful document search and question-answering system built with AWS Bedrock, Strands Agents, ChromaDB, and Streamlit. Upload your documents, ask questions in natural language, and get AI-powered answers with source citations using a lightweight agent framework.

## ✨ Features

- **📄 Multi-format Support**: Upload PDF, TXT, and Markdown files
- **🔍 Intelligent Search**: Vector-based similarity search using AWS Bedrock embeddings
- **💬 Natural Language Q&A**: Ask questions and get contextual answers from your documents
- **🤖 Agent-Powered**: Uses Strands agents framework for tool-based retrieval
- **📚 Source Citations**: See which documents were used to generate each answer
- **🌐 Web Interface**: Easy-to-use Streamlit interface
- **🗂️ File Management**: Upload, view, and delete documents with ease
- **🔄 Real-time Indexing**: Re-index your knowledgebase whenever you add new documents
- **🧪 Debug Mode**: Test retrieval without LLM to verify chunk quality

## 🏗️ Architecture

```
Documents (PDF/TXT/MD) → Text Extraction → Chunking → Vector Embeddings → ChromaDB
                                                                              ↓
User Question → Strands Agent → Retrieval Tool → Relevant Chunks → AWS Bedrock LLM → Answer + Sources
```

**Key Architecture Changes:**
- **Boto 3**: Direct AWS Bedrock API calls using boto3
- **Strands Agents**: Lightweight agent framework with tool-based retrieval
- **Native ChromaDB**: Persistent vector storage without abstraction layers
- **Tool-based Retrieval**: Embeddings and similarity search handled by agent tools

## 🚀 Quick Start

### Prerequisites

1. **AWS Account** with Bedrock access
2. **Python 3.8+**
3. **AWS CLI configured** or environment variables set

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd genai-bedrock-knowledgebase
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up AWS credentials** (choose one method):
   
   **Option A: AWS CLI**
   ```bash
   aws configure
   ```
   
   **Option B: Environment variables**
   ```bash
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_DEFAULT_REGION=us-west-2
   ```
   
   **Option C: .env file**
   ```bash
   # Create .env file in project root
   AWS_REGION=us-west-2
   EMBED_MODEL_ID=amazon.titan-embed-text-v1
   LLM_MODEL_ID=us.anthropic.claude-3-5-haiku-20241022-v1:0
   ```

4. **Enable Bedrock Models** (in AWS Console)
   - Go to AWS Bedrock Console
   - Navigate to "Model access"
   - Enable these models:
     - `amazon.titan-embed-text-v1` (for embeddings)
     - `us.anthropic.claude-3-5-haiku-20241022-v1:0` (for Q&A)
     - Or any other supported Bedrock models

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser** to `http://localhost:8501`

## 📖 Usage Guide

### 1. Upload Documents
- Navigate to the **"📤 Upload & Re-Index"** tab
- Upload PDF, TXT, or MD files
- Click **"🔄 Re-index Knowledgebase"** to process documents

### 2. Test Retrieval (Optional)
- Go to the **"💬 Ask Questions"** tab
- Use **"🧪 Test Retrieval"** to verify chunks are being found correctly
- Adjust top-K settings if needed

### 3. Ask Questions
- Enter your question in the **"💬 Ask via LLM"** section
- Choose your preferred Bedrock model
- Adjust temperature for creativity vs. precision
- Click **"Generate Answer"** to get AI-powered responses
- Review source documents and similarity scores

### 4. Manage Files
- Use the **"🗑️ Delete Files"** tab to remove documents
- Re-index after deleting files to update the search index

## 🔧 Configuration

### Model Settings
You can modify these in your `.env` file or as environment variables:

```bash
# AWS region for Bedrock
AWS_REGION=us-west-2

# Embedding model for document vectorization
EMBED_MODEL_ID=amazon.titan-embed-text-v1

# Language model for question answering
LLM_MODEL_ID=us.anthropic.claude-3-5-haiku-20241022-v1:0
```

### Text Processing Parameters
Modify these constants in `app.py`:

```python
# Text chunking parameters
chunk_size = 500        # Characters per chunk
chunk_overlap = 50      # Overlap between chunks

# Retrieval settings
K_RETRIEVE = 3         # Top-K chunks to retrieve
```

### Directory Structure
```
project/
├── app.py              # Main application
├── data/               # Uploaded documents (auto-created)
├── requirements.txt    # Python dependencies
├── .env               # AWS credentials & config (optional)
└── README.md          # This file
```

## 📦 Dependencies

Create a `requirements.txt` file with:

```txt
streamlit>=1.28.0
python-dotenv>=1.0.0
PyPDF2>=3.0.0
boto3>=1.34.0
chromadb>=0.4.0
strands-agents>=0.1.0
```

## 🛠️ How It Works

### Document Processing Pipeline
1. **Text Extraction**: PDFs are converted to text using PyPDF2
2. **Chunking**: Documents are split into ~500 character chunks with 50 character overlap
3. **Vectorization**: Each chunk is converted to embeddings using AWS Bedrock Titan
4. **Storage**: Vectors are stored in persistent ChromaDB for fast similarity search

### Agent-Based Question Answering
1. **Agent Initialization**: Strands agent is created with Bedrock model and retrieval tool
2. **Tool Invocation**: Agent calls `retrieve_chunks` tool with user question
3. **Similarity Search**: Tool embeds question and finds most relevant document chunks
4. **Context Assembly**: Retrieved chunks are formatted and passed to the LLM
5. **Answer Generation**: Bedrock model generates answer based on document context
6. **Source Attribution**: Original document chunks are shown with similarity scores

### Strands Agent Flow
```python
# Agent with retrieval tool
agent = Agent(model=bedrock_model, tools=[retrieve_chunks])

# Agent automatically:
# 1. Calls retrieve_chunks(question) 
# 2. Gets formatted context from ChromaDB
# 3. Generates answer using context
# 4. Returns final response with citations
```

## 🔒 Security Notes

- **Persistent Storage**: ChromaDB uses persistent directories with configurable cleanup
- **Local Processing**: Documents are processed locally before sending to AWS
- **AWS IAM**: Ensure your AWS credentials have minimal required Bedrock permissions:
  ```json
  {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "bedrock:InvokeModel"
        ],
        "Resource": "*"
      }
    ]
  }
  ```
- **Data Privacy**: Consider data sensitivity when using cloud AI services

## 🐛 Troubleshooting

### Common Issues

**Q: "No module named 'strands'"**
```bash
pip install strands-agents
```

**Q: "Unable to locate credentials"**
- Verify AWS credentials are configured
- Check AWS region has Bedrock access
- Ensure Bedrock models are enabled in AWS Console

**Q: "No valid documents found to index"**
- Upload documents first via "Upload & Re-Index" tab
- Ensure files are PDF, TXT, or MD format
- Check that files have readable content

**Q: "Error generating answer"**
- Re-index your knowledgebase
- Verify Bedrock models are enabled
- Check AWS credentials and permissions
- Try the "Test Retrieval" feature first

**Q: "Collection count error"**
- ChromaDB persistence directory may be corrupted
- Try re-indexing to create a fresh collection

### Debug Mode
Use the "🧪 Test Retrieval" feature to:
- Verify chunks are being retrieved
- Check similarity scores
- Inspect chunk content before LLM processing

Add detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🚀 Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Cloud Deployment
- **Streamlit Cloud**: Connect your GitHub repo
- **AWS EC2**: Deploy on EC2 instance with IAM role
- **Docker**: Containerize the application

### Example Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [AWS Bedrock](https://aws.amazon.com/bedrock/) for AI models
- [Strands](https://github.com/strands-ai/strands) for lightweight agent framework
- [Streamlit](https://streamlit.io/) for web interface
- [ChromaDB](https://www.trychroma.com/) for vector storage

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/patweb99/streamlit-kb/issues)
- 📧 **Email**: patweb99@gmail.com

---

**⭐ Star this repo if you find it helpful!**