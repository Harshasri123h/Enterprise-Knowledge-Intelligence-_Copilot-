🏢 Enterprise Knowledge Assistant (Hybrid RAG)

A production-style conversational RAG application that answers user questions using enterprise documents (PDF + CSV) with hybrid retrieval, metadata-based routing, and source attribution.

Built using LangChain, FAISS, BM25, Groq LLM, and Streamlit.


🚀 Features

📄 Multi-source knowledge ingestion

PDF documents (APJ Abdul Kalam Speech)

CSV data (Employee information)

🔍 Hybrid Retrieval

Vector search (FAISS + embeddings)

Keyword search (BM25)

Deduplication of results

🧭 Metadata-based Routing

Automatically routes questions to:

PDF content (speech-related questions)

CSV content (employee-related questions)

🧠 Strict RAG

LLM answers only from retrieved context

Responds with “I don't have that information” if data is missing

🧾 Source Attribution

Shows exact document & page number used for answers

💬 Conversation Memory

Maintains chat history across turns

🖥️ Interactive UI

Built using Streamlit

Debug panel to inspect retrieved chunks


🧱 Architecture

User Question
      ↓
Hybrid Retriever
(Vector Search + BM25)
      ↓
Metadata Filtering
(PDF / CSV)
      ↓
Context Construction
      ↓
LLM (Groq - Llama 3.1)
      ↓
Answer + Sources



🛠️ Tech Stack

Python

LangChain

FAISS

BM25 Retriever

Ollama Embeddings (nomic-embed-text)

Groq LLM (llama-3.1-8b-instant)

Streamlit


📂 Project Structure
├── app.py
├── apjspeech.pdf
├── employees.csv
├── requirements.txt
└── README.md


⚙️ Installation & Setup
1️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the app
streamlit run app.py

🧪 Example Questions
PDF-based

Summarize the APJ Abdul Kalam speech

What vision did APJ Kalam share for India?

CSV-based

List employees from the IT department

What is the average salary of employees?

## 🎯 Use Case

This system can be used by enterprises to:
- Query internal policy documents
- Analyze HR employee data
- Enable conversational knowledge assistants


🧠 Key Design Decisions

Hybrid retrieval improves recall compared to vector-only search

Metadata routing avoids irrelevant context mixing

Strict prompt design prevents hallucinations

Source tracing ensures enterprise-grade explainability

📈 Possible Enhancements

Add reranking (cross-encoder)

Persist vector store to disk

Dockerize & deploy on AWS EC2

Add authentication & role-based access

Integrate Snowflake as a data source

🎯 Interview Talking Points

Why hybrid retrieval is better than vector-only

How metadata routing improves accuracy

How hallucinations are controlled

Trade-offs between chunk size & overlap

How this system scales in enterprise settings

👤 Author

Srimathi M
GenAI / AI Engineer | Data Engineering Background
Snowflake Certified | LangChain | RAG | LLMs