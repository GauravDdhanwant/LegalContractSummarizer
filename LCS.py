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

# Streamlit Interface
st.title("Chat with PDF/Docx using Google Gemini AI")

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

# Number of Iterations Input
num_iterations = st.number_input("Enter the number of iterations:", min_value=1, max_value=100, value=1, step=1)

# Instructions Input
instructions = st.text_area("Enter instructions for the Gemini model (e.g., summarize, explain key concepts):")

if st.button("Start Processing"):
    if not uploaded_files:
        st.error("Please upload at least one document.")
    else:
        combined_text = ""
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

        for iteration in range(num_iterations):
            prompt = f"{instructions}: {combined_text}"
            response = model.generate_content(prompt)

            st.write(f"Iteration {iteration + 1} Response:")
            st.write(response.text)

            # Logging
            today = datetime.date.today().strftime("%Y-%m-%d")
            log_file = os.path.join(log_folder, f"{today}.log")
            with open(log_file, 'a') as log:
                log.write(f"Iteration {iteration + 1} Response: {response.text}\n")

            # Save response
            output_filename = f"Output_{filename}_{iteration + 1}.txt"
            output_path = os.path.join(output_responses_folder, output_filename)
            with open(output_path, 'w') as output_file:
                output_file.write(response.text)

        st.success("Processing Complete. Check the log and output files.")

if st.button("Clear Logs"):
    for file in os.listdir(log_folder):
        os.remove(os.path.join(log_folder, file))
    st.success("Logs cleared successfully.")
