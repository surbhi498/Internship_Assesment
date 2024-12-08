# 📄 PDF Processing and Analysis

This project provides a comprehensive solution for extracting and processing content from PDF files, including **text**, **images**, and **tables**. The extracted content is then stored in a **Qdrant vector database** for efficient querying and retrieval.

---

## ✨ Features

- **Text Extraction**: Extracts text from PDF pages and stores it with metadata.
- **Image Extraction and Enhancement**: Extracts images from PDF pages, enhances text within images, and uploads them to AWS S3.
- **Table Extraction**: Extracts tables from PDF pages using Camelot and generates descriptions.
- **Content Classification**: Classifies and processes content in rectangular boxes.
- **Qdrant Integration**: Stores extracted content in a Qdrant vector database for efficient querying.

---

## 🛠 Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/your-repo.git
    cd your-repo
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    - Create a `.env` file in the root directory of the project.
    - Add the following environment variables:
        ```env
        GOOGLE_APPLICATION_CREDENTIALS=path/to/your/google-credentials.json
        OPENAI_API_KEY=your-openai-api-key
        QDRANT_API_KEY=your-qdrant-api-key
        LLAMA_CLOUD_API_KEY=your-llama-cloud-api-key
        ```

---

## 🚀 Usage

1. **Place the PDF file** you want to process in the root directory and rename it to `algo.pdf`.

2. **Run the main script**:
    ```bash
    python Code.py
    ```

   The script will:
   - Load the PDF file.
   - Extract images, tables, and text.
   - Enhance text in images and upload them to AWS S3.
   - Generate descriptions for tables.
   - Store all extracted content in the Qdrant vector database.

---

## 🧰 Functions

Here’s an overview of key functions in the project:

| **Function**                        | **Description**                                                                 |
|-------------------------------------|---------------------------------------------------------------------------------|
| `encode_image_base64`               | Encodes an image from a file path or URL to base64 format.                      |
| `extract_text_from_image`           | Extracts text from an image using Google Vision API.                            |
| `resize_base64_image`               | Resizes an image encoded as Base64 string.                                      |
| `enhance_text_in_image`             | Enhances the text in an image.                                                 |
| `generate_algorithm_description`    | Generates a detailed description of the algorithm.                              |
| `retry_with_exponential_backoff`    | Retries a function with exponential backoff.                                    |
| `generate_image_description`        | Generates a description for an image using GPT-4.                               |
| `generate_image_description_fallback`| Generates a fallback description for an image.                                  |
| `classify_and_process_image`        | Classifies and processes content in rectangular boxes.                          |
| `upload_to_s3`                      | Uploads a file to an S3 bucket and returns the public URL.                      |
| `extract_images`                    | Extracts images from PDF content.                                               |
| `is_table`                          | Checks if the extracted data is a table.                                        |
| `generate_table_description`        | Generates a description for a table.                                            |
| `generate_table_description_fallback`| Generates a fallback description for a table.                                   |
| `extract_tables`                    | Extracts and describes tables from a PDF using Camelot.                         |
| `extract_text`                      | Extracts text from PDF with metadata.                                           |
| `query_qdrant`                      | Queries Qdrant to retrieve relevant data.                                       |
| `reconstruct_table`                 | Reconstructs a table from its JSON representation.                              |
| `load_pdf`                          | Extracts content (text, images, tables) from PDF pages.                         |
| `main`                              | Main function to execute the script.                                            |

---

## Watch BOTH the YOUTUBE LINK FOR RESULTS AND DEMO:
[Watch the video on YouTube](https://youtu.be/ISkghXJ5XKU)
[Watch the video on YouTube](https://youtu.be/bVNKahqvcxM)



## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Open an issue to discuss improvements or bug fixes.
2. Submit a pull request for review.

---

## 🙏 Acknowledgements

This project leverages the following amazing tools and libraries:

- [pdfplumber](https://github.com/jsvine/pdfplumber)
- [pytesseract](https://github.com/madmaze/pytesseract)
- [Google Cloud Vision API](https://cloud.google.com/vision)
- [Qdrant](https://qdrant.tech/)
- [Sentence Transformers](https://www.sbert.net/)
- [Camelot](https://camelot-py.readthedocs.io/)
- [OpenAI](https://openai.com)

---

Made with ❤️ for efficient PDF content processing!
