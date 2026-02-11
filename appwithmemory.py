import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser



# -------------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="Enterprise RAG", layout="wide")
st.title("🏢 Enterprise Knowledge Assistant")

# -------------------------------------------------------
# LOAD DOCUMENTS
# -------------------------------------------------------
documents = []

pdf_docs = PyPDFLoader("apjspeech.pdf").load()
csv_docs = CSVLoader("employees.csv").load()

for d in pdf_docs:
    d.metadata["source_type"] = "pdf"        # logic
    d.metadata["source"] = "APJ Speech PDF"  # display

for d in csv_docs:
    d.metadata["source_type"] = "csv"        # logic
    d.metadata["source"] = "Employee CSV"    # display



documents.extend(pdf_docs)
documents.extend(csv_docs)

# -------------------------------------------------------
# SPLIT DOCUMENTS
# -------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(documents)

# -------------------------------------------------------
# EMBEDDINGS + VECTOR STORE
# -------------------------------------------------------
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(chunks, embeddings)

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# -------------------------------------------------------
# KEYWORD RETRIEVER
# -------------------------------------------------------
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 4

# -------------------------------------------------------
# LLM
# -------------------------------------------------------
llm=ChatGroq(model="llama-3.1-8b-instant",groq_api_key=os.getenv("GROQ_API_KEY"))


# -------------------------------------------------------
# PROMPT (STRICT RAG)
# -------------------------------------------------------
prompt = ChatPromptTemplate.from_template("""
You are an enterprise assistant.

Answer ONLY using the context provided.
If the answer is not in the context, say:
"I don't have that information in the documents."

Context:
{context}

Question:
{question}

Answer:
""")

# -------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------
def merge_documents(docs1, docs2):
    seen = set()
    merged = []

    for doc in docs1 + docs2:
        content = doc.page_content
        if content not in seen:
            merged.append(doc)
            seen.add(content)

    return merged


def format_sources(docs):
    sources = set()

    for doc in docs:
        source = doc.metadata.get("source", "unknown")

        if "page" in doc.metadata:
            sources.add(f"{source} (page {doc.metadata['page'] + 1})")
        else:
            sources.add(source)

    return list(sources)

def filter_by_question(docs, question):
    q = question.lower()

    # If question is about employees → use CSV only
    if any(word in q for word in ["employee", "salary", "age", "department"]):
        return [d for d in docs if d.metadata.get("source_type") == "csv"]

    # If question is about APJ speech → use PDF only
    if any(word in q for word in ["apj", "kalam", "speech", "vision"]):
        return [d for d in docs if d.metadata.get("source_type") == "pdf"]

    # Otherwise → use everything
    return docs


# -------------------------------------------------------
# CHAT MEMORY
# -------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------------------------------
# UI
# -------------------------------------------------------
question = st.text_input("Ask your question")

if question:
    # Hybrid retrieval
    vector_docs = vector_retriever.invoke(question)
    keyword_docs = bm25_retriever.invoke(question)

    final_docs = merge_documents(vector_docs, keyword_docs)
    final_docs = filter_by_question(final_docs, question)

    if not final_docs:
        st.warning("No relevant documents found.")
    else:
        context = "\n\n".join(doc.page_content for doc in final_docs)

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    sources = format_sources(final_docs)

    # Store memory
    st.session_state.chat_history.append({
        "question": question,
        "answer": answer,
        "sources": sources
    })

    with st.expander("🔍 Retrieval Debug"):
        for d in final_docs:
            st.write(f"{d.metadata['source_type']} → {d.page_content[:300]}")


# -------------------------------------------------------
# DISPLAY CHAT
# -------------------------------------------------------


for chat in st.session_state.chat_history:
    st.markdown(f"### ❓ Question")
    st.write(chat["question"])

    st.markdown(f"### ✅ Answer")
    st.write(chat["answer"])

    st.markdown("### 📄 Sources")
    for src in chat["sources"]:
        st.write(f"- {src}")

    st.markdown("---")
