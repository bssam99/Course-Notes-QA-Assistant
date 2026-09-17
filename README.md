# Course Notes QA Assistant (RAG)

A **Retrieval-Augmented Generation (RAG)** system that answers questions based on uploaded course note PDFs.

The application extracts text from PDFs, splits the content into chunks, generates semantic embeddings, stores them in a FAISS vector database, retrieves relevant content, and uses a Groq-hosted LLM to generate answers based only on the provided course notes.

## Model

* **llama-3.1-8b-instant** — Groq API

The application also provides several alternative Groq-supported models through the Streamlit interface.

## Features

* Upload one or multiple PDF course notes
* Extract text from PDF pages
* Split documents into configurable chunks
* Generate embeddings using Hugging Face
* Store embeddings using FAISS
* Retrieve the most relevant document chunks
* Generate answers using a Groq LLM
* Display retrieved source pages and content
* Configurable chunk size and overlap
* Configurable Top-k retrieval
* Adjustable LLM temperature
* Streamlit-based user interface

## RAG Pipeline

```text
PDF Documents
      │
      ▼
PDF Text Extraction
      │
      ▼
Document Chunking
      │
      ▼
Hugging Face Embeddings
      │
      ▼
FAISS Vector Store
      │
      ▼
User Question
      │
      ▼
Similarity Retrieval
      │
      ▼
Relevant Context
      │
      ▼
Groq LLM
      │
      ▼
Answer + Sources
```

## How It Works

### 1. Upload PDFs

Users upload one or more course-note PDFs through the Streamlit interface.

The application extracts text from each page using `pypdf`.

Each document stores metadata containing:

* PDF filename
* Page number

### 2. Document Chunking

The extracted text is split into smaller chunks using LangChain's `RecursiveCharacterTextSplitter`.

The user can configure:

* Chunk Size
* Chunk Overlap

Default values:

```text
Chunk Size: 1000
Chunk Overlap: 150
```

### 3. Generate Embeddings

Each chunk is converted into a vector representation using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

These embeddings are used for semantic similarity search.

### 4. FAISS Retrieval

The embeddings are stored in a FAISS vector store.

When the user asks a question, the system retrieves the most relevant chunks from the uploaded course notes.

The number of retrieved chunks can be configured using **Top-k**.

Default:

```text
Top-k: 3
```

### 5. Answer Generation

The retrieved content is passed to the selected Groq LLM together with the user's question.

The system prompt instructs the model to:

* Use only the provided context
* Avoid outside knowledge
* Say `"I don't know based on the provided notes."` when the answer is not found in the context
* Keep answers concise and academic

### 6. Source Display

After generating an answer, the application displays the retrieved sources, including:

* PDF filename
* Page number
* Retrieved text

This allows users to inspect the content used to generate the answer.

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/bssam99/Course-Notes-QA-Assistant.git
cd Course-Notes-QA-Assistant
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application

```bash
python -m streamlit run app.py
```

### 4. Enter Your Groq API Key

When the application opens, enter your Groq API key in the sidebar.

### 5. Upload Course Notes

Upload one or more PDF files containing your course materials.

### 6. Build the Document Index

Click **Understand Docs**.

The application will:

1. Extract the PDF text
2. Create document chunks
3. Generate embeddings
4. Build the FAISS vector store

### 7. Ask Questions

Enter a question about the uploaded course notes and click **Get Answer**.

### 8. Review Sources

The application displays the PDF filename, page number, and retrieved text used as context for the answer.

## Configuration

The sidebar provides several configuration options.

### Groq Model

The application supports multiple Groq models, including:

* `llama-3.1-8b-instant`
* `llama-3.3-70b-versatile`
* `qwen/qwen3-32b`
* `moonshotai/kimi-k2-instruct-0905`
* `groq/compound`
* `groq/compound-mini`
* `openai/gpt-oss-20b`
* `openai/gpt-oss-120b`

### Temperature

Controls the randomness of the generated responses.

Default:

```text
0.2
```

### Top-k Chunks

Controls the number of document chunks retrieved for each question.

Range:

```text
1–8
```

Default:

```text
3
```

### Chunk Size

Controls the maximum size of each document chunk.

Range:

```text
400–2000
```

Default:

```text
1000
```

### Chunk Overlap

Controls the amount of overlapping text between consecutive chunks.

Range:

```text
0–400
```

Default:

```text
150
```

## Technologies

| Technology            | Purpose                  |
| --------------------- | ------------------------ |
| Python                | Application development  |
| Streamlit             | Web interface            |
| LangChain             | RAG pipeline             |
| pypdf                 | PDF text extraction      |
| Hugging Face          | Text embeddings          |
| Sentence Transformers | Embedding model          |
| FAISS                 | Vector similarity search |
| Groq                  | LLM inference            |

## Project Structure

```text
Course-Notes-QA-Assistant/
│
├── app.py
├── requirements.txt
└── README.md
```

## Key Components

### PDF Processing

`pypdf` is used to extract text from uploaded PDF files while preserving page-level metadata.

### Embeddings

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

to convert document chunks into semantic vector representations.

### Vector Search

FAISS is used to efficiently search for document chunks that are semantically relevant to the user's question.

### LLM Integration

Groq is integrated through LangChain's `ChatGroq` interface for answer generation.

### Streamlit

Streamlit provides the interactive interface for:

* PDF uploads
* Model configuration
* RAG configuration
* Question answering
* Source inspection

## RAG Prompt

The assistant is instructed to follow these rules:

```text
Use ONLY the provided context to answer.
If the answer is not explicitly in the context, say:
"I don't know based on the provided notes."
Do not use outside knowledge.
Keep the answer concise and academic.
```

This helps keep generated answers grounded in the uploaded course materials.

## Limitations

* Text-based PDFs are supported; scanned PDFs may require OCR.
* Retrieval quality depends on chunking and embedding quality.
* The system relies on Groq for LLM inference.
* FAISS indexes are built during the current application session.
* The system does not currently use a dedicated reranking model.
* Source information is displayed as retrieved document passages rather than formal inline citations.

## Future Improvements

* Add OCR support for scanned PDFs
* Add persistent vector databases
* Add conversation history
* Add hybrid keyword + semantic retrieval
* Add reranking
* Support additional document formats
* Add RAG evaluation metrics
* Add improved citation generation
* Add document management and multiple knowledge bases
* Deploy the application to a cloud platform

## Skills Demonstrated

* Retrieval-Augmented Generation (RAG)
* Natural Language Processing
* Semantic Search
* Text Embeddings
* Vector Databases
* Large Language Models
* Prompt Engineering
* LangChain
* Streamlit
* Python
* PDF Processing
* API Integration

## Author

**Bassam Ahmed**

Data Scientist / AI Engineer

GitHub: [bssam99](https://github.com/bssam99)
