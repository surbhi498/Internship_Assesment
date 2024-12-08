import streamlit as st
import json
from test3 import query_qdrant, reconstruct_table  # Import necessary functions
from test3 import qdrant, model  # Import Qdrant and SentenceTransformer model
import logging

def query_qdrant(query_text, collection_name="pdf_metadata_surbhi"):
    """Query Qdrant to retrieve relevant data based on the query."""
    try:
        # Generate embedding for the query text
        query_vector = model.encode(query_text).tolist()

        # Query Qdrant to retrieve relevant points
        response = qdrant.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=10 # Adjust the limit as needed
        )

        # Process the results
        results = []
        for point in response:
            payload = point.payload
            results.append(payload)

        return results

    except Exception as e:
        logging.error(f"Error querying Qdrant: {e}")
        return None

# Streamlit interface
st.title("PDF Metadata Query Interface")

# Input query from user
query_text = st.text_input("Enter your query:")

if st.button("Search"):
    if query_text:
        results = query_qdrant(query_text)
        if results:
            st.write("### Results:")
            for result in results:
                if result["type"] == "table":
                    st.subheader("Table")
                    table = reconstruct_table(result["table"])
                    st.write("Table Description:", result["description"])
                    st.write("Page Number:", result["page"])
                    st.write("Table Content:")
                    st.table(table)
                elif result["type"] == "text":
                    st.subheader("Text")
                    st.write("Text Content:", result["text"])
                    st.write("Page Number:", result["page"])
                    st.write("Paragraph Number:", result["paragraph"])
                elif result["type"] == "image":
                    st.subheader("Image")
                    st.write("Image Description:", result["description"])
                    st.write("Page Number:", result["page"])
                else:
                    st.json(result)
        else:
            st.write("No results found.")
    else:
        st.write("Please enter a query.")