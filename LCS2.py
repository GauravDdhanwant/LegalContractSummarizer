import streamlit as st
import os
import datetime
import google.generativeai as genai
from docx import Document
import PyPDF2
from bs4 import BeautifulSoup
import pyth

# Initialize folders
input_documents_folder = "input_documents_folder"
output_responses_folder = "output_responses_folder"
log_folder = "log_folder"
os.makedirs(input_documents_folder, exist_ok=True)
os.makedirs(output_responses_folder, exist_ok=True)
os.makedirs(log_folder, exist_ok=True)

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Streamlit Interface
st.title("Your Personalized Legal Advisor")

# API Key Input
api_key = st.text_input("Enter your Google API Key:", type="password")
if api_key:
    genai.configure(api_key=api_key)

# Model Selection
st.sidebar.header("Model Selection")
models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
selected_model = st.sidebar.selectbox("Choose a model:", models)

# Document Upload
uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True, type=['pdf', 'docx', 'txt', 'rtf', 'html'])

# Instructions Input
instructions = st.text_area("Enter your query or instructions for the legal documents:")

# Display the conversation history
st.subheader("Conversation History")
for entry in st.session_state.conversation_history:
    st.markdown(f"**User:** {entry['user']}")
    st.markdown(f"**AI:** {entry['ai']}")

# New prompt input
new_prompt = st.text_area("Enter your next query:")

if st.button("Send"):
    if not new_prompt and not instructions:
        st.error("Please enter a query or upload documents to continue.")
    else:
        combined_text = ""
        if uploaded_files:
            for uploaded_file in uploaded_files:
                filename = uploaded_file.name
                filepath = os.path.join(input_documents_folder, filename)
                with open(filepath, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                if filename.lower().endswith(".docx"):
                    document = Document(filepath)
                    combined_text += "\n".join([paragraph.text for paragraph in document.paragraphs])

                elif filename.lower().endswith(".pdf"):
                    with open(filepath, "rb") as pdf_file:
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        combined_text += "\n".join([pdf_reader.pages[i].extract_text() for i in range(len(pdf_reader.pages))])

                elif filename.lower().endswith(".html"):
                    with open(filepath, "rb") as html_file:
                        soup = BeautifulSoup(html_file, "html.parser")
                        combined_text += soup.get_text()

                elif filename.lower().endswith(".rtf"):
                    with open(filepath, "r") as rtf_file:
                        rtf_content = rtf_file.read()
                        combined_text += pyth.decode(rtf_content)

                elif filename.lower().endswith(".txt"):
                    with open(filepath, "r", encoding="utf-8") as txt_file:
                        combined_text += txt_file.read()

        st.write(f"Total combined text length: {len(combined_text)} characters")

        model = genai.GenerativeModel(selected_model)

        # Construct the conversation history into a single context
        context = "\n".join([f"User: {entry['user']}\nAI: {entry['ai']}" for entry in st.session_state.conversation_history])
        if combined_text:
            context += f"\nContext from documents: {combined_text}"

        prompt = f"{context}\nUser: {new_prompt}"
        response = model.generate_content(prompt)

        # Save the interaction to the session state
        st.session_state.conversation_history.append({
            "user": new_prompt,
            "ai": response.text
        })

        st.write("Conversation Response:")
        st.write(response.text)

        # Logging
        today = datetime.date.today().strftime("%Y-%m-%d")
        log_file = os.path.join(log_folder, f"{today}.log")
        with open(log_file, 'a') as log:
            log.write(f"Conversation Response: {response.text}\n")

        # Save response
        output_filename = f"Conversation_Output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        output_path = os.path.join(output_responses_folder, output_filename)
        with open(output_path, 'w') as output_file:
            output_file.write(response.text)

        st.success("Conversation updated. Check the log and output files.")

if st.button("Clear Conversation History"):
    st.session_state.conversation_history = []
    st.success("Conversation history cleared successfully.")
