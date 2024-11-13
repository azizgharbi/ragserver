# RAGServer Using Ollama

A Retrieval-Augmented Generation (RAG) model using `Ollama` as the backbone, with `Wikipedia` serving as the primary knowledge source.

## Project Setup

### 1. Create a New Python Environment
To isolate dependencies, create a virtual environment:
```bash
python3 -m venv env_name
source env_name/bin/activate
pip install -r requirements.txt
# for macos
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```
