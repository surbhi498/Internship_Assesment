import streamlit as st
import os
from tabled.extract import extract_tables
from tabled.fileinput import load_pdfs_images
from tabled.inference.models import load_detection_models, load_recognition_models, load_layout_models

# Load models
det_models, rec_models, layout_models = load_detection_models(), load_recognition_models(), load_layout_models()

# Streamlit interface
st.title("PDF and Image Table Extraction")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded file
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load PDF or images
    images, highres_images, names, text_lines = load_pdfs_images(file_path)

    # Extract tables
    page_results = extract_tables(images, highres_images, text_lines, det_models, layout_models, rec_models)

    # Display tables in markdown format
    for page_result in page_results:
        for table in page_result['tables']:
            markdown_table = table.to_markdown()
            st.markdown(markdown_table)

    # Optionally, delete the uploaded file after processing
    os.remove(file_path)