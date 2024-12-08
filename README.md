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

|Functions
📜 encode_image_base64(image_path_or_url)
Description:
Encodes an image from a file path or URL into Base64 format.

✨ enhance_text_in_image(image)
Description:
Enhances the text in the provided image for improved readability and processing.

📘 generate_algorithm_description(text)
Description:
Generates a detailed and structured description of an algorithm based on the given text.

🖼️ generate_image_description(image_url)
Description:
Creates a comprehensive description for an image using GPT-4o.

🔄 generate_image_description_fallback(image_url)
Description:
Provides a fallback image description using an alternative model when primary methods are unavailable.

📦 classify_and_process_image(image_url, page_num)
Description:
Classifies and processes content within rectangular boxes for the specified image and page number.

☁️ upload_to_s3(file_path, bucket_name, object_name=None)
Description:
Uploads a file to an Amazon S3 bucket and returns its public URL.

🖼️ extract_images(pages_content)
Description:
Extracts all images embedded in the content of PDF pages.

📊 is_table(data)
Description:
Determines whether the extracted data represents a table.

📝 generate_table_description(table_json)
Description:
Generates a detailed description of a table based on its JSON representation first trying with GPT-4o.

🔄 generate_table_description_fallback(table_json)
Description:
Creates a fallback table description using an alternative method when primary approaches fail.

📑 extract_tables(pdf_path)
Description:
Extracts and describes tables from a PDF file using Camelot.

🖋️ extract_text(pages_content)
Description:
Extracts text along with metadata from the content of PDF pages.

🔧 reconstruct_table(table_json)
Description:
Reconstructs a table from its JSON representation.

📂 load_pdf(file_path)
Description:
Processes and extracts content (text, images, tables) from a PDF file.

🚀 main()
Description:
The main function that executes the entire script.

---


## Results:
### Examples of extracted and stored:
■ Text with metadata.
■ Tables with structure and description.
■ Image descriptions with metadata.

[Watch the video on YouTube](https://youtu.be/gJWcaQbjL_Y)

## Example queries demonstrating data retrieval from Qdrant.
### Watch BOTH the YOUTUBE LINK FOR RESULTS AND DEMO:
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
- [Qdrant](https://qdrant.tech/)
- [Sentence Transformers](https://www.sbert.net/)
- [Camelot](https://camelot-py.readthedocs.io/)
- [OpenAI](https://openai.com)
- [LM STUDIO](https://lmstudio.ai/)


---

---

## ⚡ Challenges Encountered

During the development of this project, several challenges were encountered, which required innovative solutions and careful planning:

1. **Handling Different PDF Structures**  
   PDFs come in various formats and structures, making consistent content extraction challenging. Tools like `pdfplumber` and `Camelot` were instrumental in managing these variations.

2. **OCR Accuracy**  
   Extracting text from images using OCR often led to inaccuracies. The combined use of `pytesseract` and the Google Cloud Vision API did not even help me in significantly improving the accuracy of text extraction. Images of FlowChart are in poor quality  and have blurred or fainted text inside flowCharts that was difficult to extract and provide as a context to multimodal in generating descriptions.

3. **Rate Limiting**  
   Interacting with the OpenAI API introduced challenges related to rate limits. Implementing **exponential backoff** and retry mechanisms successfully mitigated these issues.

4. **Embedding and Storage**  
   Storing and querying large amounts of vector data efficiently required seamless integration with the **Qdrant vector database** and careful optimization of query operations.

5. **Generating Descriptions**  
   Ensuring the descriptions generated by OpenAI models/ llava-v1.5-7b were accurate and relevant involved extensive fine-tuning of prompts. A fallback mechanism was also implemented to handle edge cases effectively.

---



Made with ❤️ for efficient PDF content processing!

