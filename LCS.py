import streamlit as st
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
import PyPDF2
import docx

# Load RAG model and tokenizer
@st.cache_resource
def load_rag_pipeline():
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
    retriever = RagRetriever.from_pretrained("facebook/rag-token-base", use_dummy_dataset=True)
    model = RagTokenForGeneration.from_pretrained("facebook/rag-token-base", retriever=retriever)
    return tokenizer, model

tokenizer, model = load_rag_pipeline()

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
        inputs = tokenizer(query, return_tensors="pt")
        outputs = model.generate(**inputs)
        answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Append query and answer to chat history
        st.session_state.chat_history.append({"query": query, "answer": answer})
        
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
