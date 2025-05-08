# Mercury AI Assistant

Mercury is an AI assistant built using the AIQ Toolkit that combines multiple frameworks (LangChain, LlamaIndex, and Haystack) to provide a versatile conversational experience. It features:

- Wikipedia-based research capabilities
- Chit-chat functionality
- Document retrieval and RAG (Retrieval-Augmented Generation)

## Prerequisites

- Python 3.10 or higher
- AIQ Toolkit (installed from NVIDIA's package repository)
- NVIDIA API Key for LLM access

## Dependencies

The project requires the following main packages:
- `aiq-toolkit`: NVIDIA's AI Query Toolkit
- `langchain`: For LLM interactions and chains
- `llama-index`: For document indexing and retrieval
- `haystack`: For question-answering capabilities
- `wikipedia`: For Wikipedia search functionality

## Features

- **Research Mode**: Uses Wikipedia to search and summarize information about various topics
- **Retrieval Mode**: Accesses and retrieves information from local documentation
- **General Chat Mode**: Handles casual conversation and general queries

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -e .
   ```

3. Configure the environment:
   - Set up NVIDIA API key:
     ```bash
     export NVIDIA_API_KEY="your-api-key"
     ```
   - Adjust the configuration in `mercury/configs/config.yml`

## Usage

Run the assistant with:
```bash
cd mercury
aiq run --config_file=configs/config.yml --input "your question here"
```

## Project Structure

- `mercury/`: Main project directory
  - `configs/`: Configuration files
  - `src/`: Source code
    - `register.py`: Workflow registration and routing logic
    - `wikipedia_search_tool.py`: Wikipedia search implementation
    - `haystack_agent.py`: Chat functionality
    - Other implementation files

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details. 