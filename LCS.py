import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import PyPDF2
import docx
import os

# Load a pre-trained model for QA (e.g., a RAG model or a similar model)
# Assuming the use of a pipeline for simplicity
@st.cache_resource
def load_qa_pipeline():
    tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/rag-token-base")
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return qa_pipeline

qa_pipeline = load_qa_pipeline()

def read_pdf(file):
    reader = PyPDF2.PdfFileReader(file)
    text = ""
    for page_num in range(reader.numPages):
        page = reader.getPage(page_num)
        text += page.extract_text()
    return text

def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def process_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return read_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return read_docx(uploaded_file)
    else:
        st.error("Unsupported file type.")
        return None

# Streamlit app UI
st.title("Legal Document QA with Chain of Thought")

if "document_text" not in st.session_state:
    st.session_state.document_text = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File upload
uploaded_file = st.file_uploader("Upload a legal document (PDF or DOCX)", type=["pdf", "docx"])
if uploaded_file:
    st.session_state.document_text = process_file(uploaded_file)
    st.write("Document successfully uploaded and processed.")

if st.session_state.document_text:
    query = st.text_input("Ask a question about the document:")
    
    if st.button("Submit Query") and query:
        # Use the RAG model for QA
        inputs = {"question": query, "context": st.session_state.document_text}
        answer = qa_pipeline(**inputs)
        
        # Append query and answer to chat history
        st.session_state.chat_history.append({"query": query, "answer": answer['answer']})
        
        # Display the chat history
        for idx, chat in enumerate(st.session_state.chat_history):
            st.write(f"**Q{idx+1}:** {chat['query']}")
            st.write(f"**A{idx+1}:** {chat['answer']}")

        st.write("---")

# Add the ability to reset the session
if st.button("Reset Session"):
    st.session_state.chat_history = []
    st.session_state.document_text = None
    st.experimental_rerun()
