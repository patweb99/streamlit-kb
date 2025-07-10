# 🧠 GenAI Bedrock Knowledgebase

A powerful document search and question-answering system built with AWS Bedrock, LangChain, and Streamlit. Upload your documents, ask questions in natural language, and get AI-powered answers with source citations.

## ✨ Features

- **📄 Multi-format Support**: Upload PDF, TXT, and Markdown files
- **🔍 Intelligent Search**: Vector-based similarity search using AWS Bedrock embeddings
- **💬 Natural Language Q&A**: Ask questions and get contextual answers from your documents
- **📚 Source Citations**: See which documents were used to generate each answer
- **🌐 Web Interface**: Easy-to-use Streamlit interface
- **🗂️ File Management**: Upload, view, and delete documents with ease
- **🔄 Real-time Indexing**: Re-index your knowledgebase whenever you add new documents

## 🏗️ Architecture

```
Documents (PDF/TXT/MD) → Text Extraction → Chunking → Vector Embeddings → ChromaDB
                                                                              ↓
User Question → Similarity Search → Relevant Chunks → AWS Bedrock LLM → Answer + Sources
```

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
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_DEFAULT_REGION=us-west-2
   ```

4. **Enable Bedrock Models** (in AWS Console)
   - Go to AWS Bedrock Console
   - Navigate to "Model access"
   - Enable these models:
     - `amazon.titan-embed-text-v1` (for embeddings)
     - `us.amazon.nova-micro-v1:0` (for Q&A)

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser** to `http://localhost:8501`

## 📖 Usage Guide

### 1. Upload Documents
- Navigate to the **"📤 Upload Files"** tab
- Upload PDF, TXT, or MD files
- Click **"🔄 Re-index Knowledgebase"** to process documents

### 2. Ask Questions
- Go to the **"💬 Ask Questions"** tab
- Type your question in natural language
- Click **"Generate Answer"** to get AI-powered responses
- Review source documents to verify accuracy

### 3. Manage Files
- Use the **"🗑️ Delete Files"** tab to remove documents
- Re-index after deleting files to update the search index

## 🔧 Configuration

### Model Settings
You can modify these constants in `app.py`:

```python
# Embedding model for document vectorization
AWS_BEDROCK_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v1"

# Language model for question answering
AWS_BEDROCK_LLM_MODEL_ID = "us.amazon.nova-micro-v1:0"

# AWS region
AWS_REGION = "us-west-2"

# Text chunking parameters
chunk_size = 500        # Characters per chunk
chunk_overlap = 50      # Overlap between chunks
```

### Directory Structure
```
project/
├── app.py              # Main application
├── data/               # Uploaded documents (auto-created)
├── requirements.txt    # Python dependencies
├── .env               # AWS credentials (optional)
└── README.md          # This file
```

## 📦 Dependencies

Create a `requirements.txt` file with:

```txt
streamlit>=1.28.0
langchain>=0.1.0
langchain-community>=0.0.10
boto3>=1.34.0
chromadb>=0.4.0
PyPDF2>=3.0.0
python-dotenv>=1.0.0
```

## 🛠️ How It Works

### Document Processing
1. **Text Extraction**: PDFs are converted to text using PyPDF2
2. **Chunking**: Documents are split into ~500 character chunks with 50 character overlap
3. **Vectorization**: Each chunk is converted to embeddings using AWS Bedrock Titan
4. **Storage**: Vectors are stored in ChromaDB for fast similarity search

### Question Answering
1. **Query Processing**: User question is converted to vector embedding
2. **Similarity Search**: Find the 3 most relevant document chunks
3. **Context Assembly**: Relevant chunks are sent to AWS Bedrock Nova Micro
4. **Answer Generation**: LLM generates answer based on document context
5. **Source Attribution**: Original document chunks are shown for verification

## 🔒 Security Notes

- **Temporary Storage**: ChromaDB uses temporary directories that are cleaned up automatically
- **Local Processing**: Documents are processed locally before sending to AWS
- **AWS IAM**: Ensure your AWS credentials have minimal required Bedrock permissions
- **Data Privacy**: Consider data sensitivity when using cloud AI services

## 🐛 Troubleshooting

### Common Issues

**Q: "No module named 'streamlit'"**
```bash
pip install streamlit
```

**Q: "Unable to locate credentials"**
- Verify AWS credentials are configured
- Check AWS region has Bedrock access
- Ensure Bedrock models are enabled in AWS Console

**Q: "No valid documents found to index"**
- Upload documents first via "Upload Files" tab
- Ensure files are PDF, TXT, or MD format
- Check that files have readable content

**Q: "Error generating answer"**
- Re-index your knowledgebase
- Verify Bedrock models are enabled
- Check AWS credentials and permissions

### Debug Mode
Add this to see more detailed logs:
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
- [LangChain](https://langchain.com/) for AI framework
- [Streamlit](https://streamlit.io/) for web interface
- [ChromaDB](https://www.trychroma.com/) for vector storage

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/your-username/your-repo/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/your-username/your-repo/discussions)
- 📧 **Email**: your-email@example.com

---

**⭐ Star this repo if you find it helpful!**