# Self-Correcting RAG

A fully local Retrieval-Augmented Generation system with built-in self-correction mechanisms. No API keys required, no data leaves your machine.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

Self-Correcting RAG is a privacy-focused question-answering system that runs entirely on your local machine. It uses Ollama for language processing and HuggingFace embeddings to provide accurate answers from your documents while automatically filtering irrelevant information and fact-checking responses.

### Key Features

- **100% Local & Private** - All processing happens on your machine
- **Self-Correcting** - Automatic relevance filtering and fact-checking
- **Zero Cost** - Uses free, open-source tools
- **Easy Setup** - Get started in under 10 minutes
- **Flexible Input** - Load documents from text, files, or folders

---

## Quick Start

### Prerequisites

- Python 3.9 or higher
- 4GB+ RAM (8GB recommended)
- macOS, Linux, or Windows

### Installation

**1. Install Ollama**

macOS:
```bash
brew install ollama
```

Linux:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

Windows: Download from [ollama.ai/download](https://ollama.ai/download)

**2. Download AI Model**

```bash
ollama pull llama3.2
```

**3. Clone and Setup**

```bash
git clone https://github.com/revanthsonu/Self-correcting-RAG.git
cd Self-correcting-RAG

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**4. Run**

```bash
python free_rag.py
```

---

## Usage

### Basic Example

```python
from free_rag import FreeRAG

# Initialize
rag = FreeRAG(model_name="llama3.2")

# Add documents
documents = [
    "The Eiffel Tower is 330 meters tall.",
    "Paris is the capital of France.",
    "Python was created by Guido van Rossum in 1991."
]
rag.add_documents(documents)

# Query
result = rag.query("How tall is the Eiffel Tower?")
print(result['answer'])
```

### Loading from Files

```python
# Single file
docs = rag.load_from_text_file("my_notes.txt")
rag.add_documents(docs)

# Multiple files from folder
docs = rag.load_from_folder("./documents")
rag.add_documents(docs)
```

### Testing the System

```bash
# Run integrated tests
python free_rag.py --test
```

---

## How It Works

The system employs a four-step process with dual self-correction:

1. **Retrieval** - Finds relevant documents using semantic search
2. **Relevance Filtering** - LLM scores and filters documents (threshold: 0.5)
3. **Answer Generation** - Creates response based on filtered context
4. **Fact-Checking** - Verifies answer consistency with source material (threshold: 0.7)

This approach significantly reduces hallucinations compared to standard RAG systems.

---

## Configuration

### Use Different Models

```python
# Higher quality (slower)
rag = FreeRAG(model_name="mistral")

# Faster (lower quality)
rag = FreeRAG(model_name="llama3.2:1b")
```

### Adjust Chunking

```python
rag.add_documents(
    documents,
    chunk_size=1000,      # Larger chunks for more context
    chunk_overlap=100     # More overlap for continuity
)
```

---

## Performance

| Hardware | Model | Query Time | RAM Usage |
|----------|-------|------------|-----------|
| MacBook Air M1 | llama3.2 | 4-6s | 2-3GB |
| MacBook Pro M2 | mistral | 8-12s | 5-7GB |
| Intel Mac (16GB) | llama3.2 | 6-10s | 4-6GB |

---

## Troubleshooting

**Ollama not found**
```bash
ollama --version
# If missing, install from ollama.ai
```

**Model not downloaded**
```bash
ollama list
ollama pull llama3.2
```

**Slow performance**
```bash
ollama pull llama3.2:1b  # Smaller, faster model
```

**Out of memory**
```python
# Use smaller chunks
rag.add_documents(documents, chunk_size=200)
```

---

## Project Structure

```
Self-correcting-RAG/
├── free_rag.py           # Main implementation
├── requirements.txt      # Dependencies
├── README.md            # Documentation
├── LICENSE              # MIT License
└── .gitignore          # Git ignore rules
```

---

## Requirements

```txt
langchain>=0.1.0
langchain-community>=0.0.20
chromadb>=0.4.0
sentence-transformers>=2.2.0
```

---

## Use Cases

- **Personal Knowledge Base** - Query your notes and documents
- **Document Q&A** - Extract information from research papers
- **Code Documentation** - Search through technical documentation
- **Study Aid** - Interactive learning from textbooks

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Ollama](https://ollama.ai/) - Local LLM runtime
- [LangChain](https://langchain.com/) - RAG framework
- [HuggingFace](https://huggingface.co/) - Embedding models
- [ChromaDB](https://www.trychroma.com/) - Vector database

---

## Contact

- **Issues**: [GitHub Issues](https://github.com/revanthsonu/Self-correcting-RAG/issues)
- **Author**: [@revanthsonu](https://github.com/revanthsonu)

---

**Built with privacy and zero API costs in mind.**
