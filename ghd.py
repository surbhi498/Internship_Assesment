import pdfplumber
import pytesseract
from PIL import Image
from google.cloud import vision
import qdrant_client
from qdrant_client.models import PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
import io
from llama_parse import LlamaParse
import openai
import os
import logging
import nest_asyncio
from dotenv import load_dotenv
import base64
import requests
import numpy as np
import uuid
import json
import streamlit as st
import camelot
import cv2 as cv
import boto3
from openai import OpenAI as OpenAI_vLLM
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Apply nest_asyncio
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Initialize AWS S3 client with local credentials
s3_client = boto3.client('s3')
s3_bucket_name = 'myexcelbucket123'  # Use your actual bucket name

# Access the LLAMA_CLOUD_API_KEY environment variable
llama_cloud_api_key = os.getenv("llama_cloud")

openai_api_key = os.getenv("OPENAI_API_KEY")  # Read the API key from the environment variable
if not openai_api_key:
    logging.error("OPENAI_API_KEY environment variable is not set.")
    exit(1)
openai.api_key = openai_api_key

# Initialize LlamaParse
llama_parser = LlamaParse(
    api_key=llama_cloud_api_key,  # Use the API key from the environment variable
    result_type="markdown",  # "markdown" and "text" are available
    verbose=True,
)

# Initialize Google Vision client
client = vision.ImageAnnotatorClient()

# Initialize Qdrant client
qdrant_api_key = os.getenv("QDRANT_API_KEY")  # Ensure this is set in your environment or .env file
qdrant = qdrant_client.QdrantClient(
    url="https://fafe48b5-bacc-4cbb-83d7-ec155b073554.us-east4-0.gcp.cloud.qdrant.io",
    api_key=qdrant_api_key
)

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create Qdrant collection if it does not exist
qdrant_collection_name = "pdf_metadata7"
try:
    # Check if the collection exists
    collections = qdrant.get_collections()
    collection_names = [collection.name for collection in collections.collections]
    if qdrant_collection_name not in collection_names:
        # Create the collection if it does not exist
        qdrant.create_collection(
            collection_name=qdrant_collection_name,
            vectors_config=VectorParams(size=384, distance="Cosine")  # Adjust vector size if needed
        )
        logging.debug(f"Created Qdrant collection: {qdrant_collection_name}")
    else:
        logging.debug(f"Qdrant collection already exists: {qdrant_collection_name}")
except qdrant_client.http.exceptions.UnexpectedResponse as e:
    logging.error(f"Failed to check or create Qdrant collection: {e}")
    exit(1)

def encode_image_base64(image_path_or_url):
    """Encode image from a file path or URL to base64 format."""
    if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
        response = requests.get(image_path_or_url)
        image_bytes = response.content
    else:
        with open(image_path_or_url, "rb") as image_file:
            image_bytes = image_file.read()
    return base64.b64encode(image_bytes).decode('utf-8')

def upload_to_s3(file_path, bucket_name, object_name=None):
    """Upload a file to an S3 bucket and return the public URL."""
    if object_name is None:
        object_name = os.path.basename(file_path)

    try:
        # Check if the object already exists in the bucket
        s3_client.head_object(Bucket=bucket_name, Key=object_name)
        logging.debug(f"Image already exists in S3: {object_name}")
        url = f"https://{bucket_name}.s3.amazonaws.com/{object_name}"
        return url
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            # Object does not exist, proceed to upload
            try:
                s3_client.upload_file(file_path, bucket_name, object_name)
                url = f"https://{bucket_name}.s3.amazonaws.com/{object_name}"
                logging.debug(f"Uploaded {file_path} to S3: {url}")
                return url
            except Exception as e:
                logging.error(f"Error uploading {file_path} to S3: {e}")
                return None
        else:
            # Something else has gone wrong.
            logging.error(f"Error checking if {file_path} exists in S3: {e}")
            return None

def enhance_text_in_image(image):
    """Enhance the text in the image."""
    if image is None:
        raise ValueError("Invalid image provided")
    
    im = image.astype(np.float32)
    im = im / 255  # rescale
    im = 1 - im  # inversion. ink is the signal, white paper isn't

    # squares/rectangles
    morph_kernel = np.ones((5,5))

    # estimates intensity of text
    dilated = cv.dilate(im, kernel=morph_kernel)

    # tweak this threshold to catch faint text but not background
    textmask = (dilated >= 0.15)
    # 0.05 catches background noise of 0.02
    # 0.25 loses some text

    # rescale text pixel intensities
    # this will obviously magnify noise around faint text
    enhanced = im / dilated

    # copy unmodified background back in
    # (division magnified noise on background)
    enhanced[~textmask] = im[~textmask]

    # invert again for output
    output = 1 - enhanced

    return output

def retry_with_exponential_backoff(func, max_retries=5, initial_delay=1, backoff_factor=2):
    """Retry a function with exponential backoff."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return func()
        except openai.error.RateLimitError as e:
            logging.warning(f"Rate limit exceeded. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= backoff_factor
    raise Exception("Max retries exceeded")

def generate_image_description(image_url):
    """Generate a description for an image using GPT-4."""
    logging.debug(f"Generating image description for image URL: {image_url}")
    prompt = (
        "You are given the following algorithm description in the form of a flowchart. "
        "Please provide a detailed and accurate explanation of the algorithm, "
        "including the specific steps involved. "
        "Focus on the logical flow and decisions made at each step, and avoid making any assumptions or adding information that is not present in the flowchart. "
        "Ensure that the description is clear, concise, and directly related to the content of the flowchart."
    )
    logging.debug(f"Prompt: {prompt}")

    def request():
        return openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful algorithmic flowchart assistant."},
                {"role": "user", "content": prompt},
                {"role": "user", "content": f"Here is the image URL: {image_url}"}
            ],
            max_tokens=1000
        )

    try:
        response = retry_with_exponential_backoff(request)

        # Print the entire response for debugging
        logging.debug(f"Chat response: {response}")

        # Check if the response contains choices
        if response and 'choices' in response and len(response.choices) > 0:
            description = response.choices[0].message['content']
            logging.debug(f"Generated description: {description.strip()}")
            return description.strip()
        else:
            logging.error("No choices found in the response.")
            logging.debug(f"Full response content: {response}")
            return "No description generated."

    except Exception as e:
        logging.error(f"Error generating image description with GPT-4: {e}")
        return None

def generate_image_description_fallback(image_url):
    """Generate a description for an image using llava."""
    logging.debug(f"Generating image description for image URL: {image_url}")
    prompt = (
        "You are given the following algorithm description in the form of a flowchart. "
        "Please provide a detailed and accurate explanation of the algorithm, "
        "including the specific steps involved. "
        "Focus on the logical flow and decisions made at each step, and avoid making any assumptions or adding information that is not present in the flowchart. "
        "Ensure that the description is clear, concise, and directly related to the content of the flowchart."
    )
    logging.debug(f"Prompt: {prompt}")

    try:
        # Encode the image to base64
        base64_image = encode_image_base64(image_url)

        def request():
            return openai.ChatCompletion.create(
                model="llava-v1.5-7b",
                messages=[
                    {"role": "system", "content": "You are a helpful algorithmic flowchart assistant."},
                    {"role": "user", "content": prompt},
                    {"role": "user", "content": f"data:image/png;base64,{base64_image}"}
                ],
                max_tokens=1000
            )

        response = retry_with_exponential_backoff(request)

        # Print the entire response for debugging
        logging.debug(f"Chat response: {response}")

        # Check if the response contains choices
        if response and 'choices' in response and len(response.choices) > 0:
            description = response.choices[0].message['content']
            logging.debug(f"Generated description: {description.strip()}")
            return description.strip()
        else:
            logging.error("No choices found in the response.")
            logging.debug(f"Full response content: {response}")
            return "No description generated."

    except Exception as e:
        logging.error(f"Error generating image description with llava: {e}")
        return "Error generating description."

def classify_and_process_image(image_url, page_num):
    """Classify and process content in rectangular boxes."""
    logging.debug("Classifying and processing content in rectangular boxes")
    logging.debug(f"Image URL: {image_url}")

    # Generate a description using the image URL
    description = generate_image_description(image_url)
    if not description:
        description = generate_image_description_fallback(image_url)
    metadata = {"type": "image", "description": description, "page": page_num}
    vector = model.encode(description).tolist()
    point_id = str(uuid.uuid4())
    qdrant.upsert(
        collection_name=qdrant_collection_name,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload=metadata
            )
        ]
    )

def extract_images(pages_content):
    """Extract images from the PDF pages content."""
    logging.debug("Extracting images from PDF")
    image_metadata = []
    for page_data in pages_content:
        page_num = page_data["page_number"]
        page = page_data["page"]  # Get the actual page object
        for image_data in page_data["images"]:
            # Extract the image from the PDF using the image data
            x0, top, x1, bottom = image_data['x0'], image_data['top'], image_data['x1'], image_data['bottom']
            cropped_image = page.within_bbox((x0, top, x1, bottom)).to_image().original
            # Convert the cropped image to a numpy array
            cropped_image_np = np.array(cropped_image)

            # Enhance the text in the image
            enhanced_image = enhance_text_in_image(cropped_image_np)

            # Save the enhanced image for verification
            enhanced_image_filename = f"enhanced_{uuid.uuid4()}.png"
            cv.imwrite(enhanced_image_filename, enhanced_image)
            logging.debug(f"Saved enhanced image: {enhanced_image_filename}")

            # Upload the enhanced image to S3 and get the public URL
            image_url = upload_to_s3(enhanced_image_filename, s3_bucket_name)
            if not image_url:
                logging.error(f"Failed to upload image to S3: {enhanced_image_filename}")
                continue

            # Call classify_and_process_image to handle the image
            classify_and_process_image(image_url, page_num)

            image_metadata.append({"image_path": image_url, "page_num": page_num})

    logging.debug(f"Extracted image metadata: {image_metadata}")
    return image_metadata

def extract_tables(pdf_path):
    """Extract and describe tables from the PDF using Camelot."""
    logging.debug("Extracting tables from PDF using Camelot")
    table_metadata = []

    # Extract tables from the PDF using Camelot
    tables = camelot.read_pdf(pdf_path, pages='all')

    for i, table in enumerate(tables):
        # Convert table to DataFrame
        df = table.df

        # Convert DataFrame to list of lists
        table_data = df.values.tolist()
        
         # Check if the extracted data is a table
        if not is_table(table_data):
            logging.debug(f"Skipping non-table content on page {table.page}")
            continue

        # Remove cells with None values but keep the rows
        cleaned_table = [[cell for cell in row if cell is not None] for row in table_data]

        # Convert table to JSON for structured storage
        table_json = json.dumps(cleaned_table)
        description = generate_table_description(table_json)
        metadata = {"type": "table", "table": table_json, "description": description, "page": table.page}
        table_metadata.append(metadata)

        # Log the metadata
        logging.info(f"Table stored with metadata: {metadata}")

        # Generate embedding and store in Qdrant
        vector = model.encode(description).tolist()
        point_id = str(uuid.uuid4())
        qdrant.upsert(
            collection_name=qdrant_collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=metadata
                )
            ]
        )
    logging.debug(f"Extracted table metadata: {table_metadata}")
    return table_metadata

def extract_text(pages_content):
    """Extract text from PDF with metadata."""
    logging.debug("Extracting text from PDF")
    text_metadata = []
    for page_data in pages_content:
        page_num = page_data["page_number"]
        text = page_data["text"]
        paragraphs = text.split('\n\n')  # Split text into paragraphs
        for paragraph_num, paragraph in enumerate(paragraphs, start=1):
            metadata = {"type": "text", "text": paragraph, "page": page_num, "paragraph": paragraph_num}
            text_metadata.append(metadata)

            # Log the metadata
            logging.info(f"Text stored with metadata: {metadata}")

            # Generate embedding and store in Qdrant
            vector = model.encode(paragraph).tolist()
            point_id = str(uuid.uuid4())
            qdrant.upsert(
                collection_name=qdrant_collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=metadata
                    )
                ]
            )
    logging.debug(f"Extracted text metadata: {text_metadata}")
    return text_metadata 

def query_qdrant(query_text):
    """Query Qdrant to retrieve relevant data based on the query."""
    try:
        # Generate embedding for the query text
        query_vector = model.encode(query_text).tolist()

        # Query Qdrant to retrieve relevant points
        response = qdrant.search(
            collection_name=qdrant_collection_name,
            query_vector=query_vector,
            limit=10  # Adjust the limit as needed
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

def reconstruct_table(table_json):
    """Reconstruct the table from the JSON string."""
    table = json.loads(table_json)
    return table

def main():
    file_path = "algo.pdf"
    if not os.path.exists(file_path):
        logging.error(f"File {file_path} does not exist.")
        return

    logging.debug("Loading PDF pages")
    print("Loading PDF pages...")
    pages_content = load_pdf(file_path)

    logging.debug("Extracting images")
    print("Extracting images...")
    images = extract_images(pages_content)

    logging.debug("Extracting tables")
    print("Extracting tables...")
    tables = extract_tables(file_path)

    logging.debug("Extracting text")
    print("Extracting text...")
    texts = extract_text(pages_content)

    logging.debug("Images: %s", images)
    logging.debug("Tables: %s", tables)
    logging.debug("Texts: %s", texts)

    logging.info("Program executed successfully.")

if __name__ == "__main__":
    main()