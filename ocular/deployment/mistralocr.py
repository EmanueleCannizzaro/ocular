"""
Mistral AI OCR client implementation.
"""

import base64
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import json
import os
from pathlib import Path

from mistralai import Mistral #, DocumentFileChunk
from mistralai.extra import response_format_from_pydantic_model

from ocular.config import OcularConfig
from ocular.exceptions import OCRError, APIError, DocumentProcessingError
from ocular.models.services import RootModel


class MistralOCRClient:
    """Client for performing OCR using Mistral AI vision models."""

    def __init__(self, config: Optional[OcularConfig] = None):
        """Initialize the Mistral OCR client."""
        self.config = config or OcularConfig.from_env()

        root_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        # 1. Load environment variables
        env_filename = os.path.join(root_folder, '.env')
        load_dotenv()
        api_key = os.getenv("MISTRAL_API_KEY")

        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables. Please check your .env file.")

        # 4. Initialize the Mistral client
        self.client = Mistral(api_key=api_key)

        # 2. Read the JSON schema from a file (e.g., schema.json)
        try:
            json_filename = os.path.join(root_folder, 'data/schema.json')
            with open(json_filename, "r") as f:
                data = json.load(f)
                # print(json.dumps(data, indent=4))

            # Extract the JSON schema
            # Generate the JSON Schema from the top-level Pydantic model
            self.schema = RootModel.model_json_schema()

            # Print the resulting schema in a readable format
            print(json.dumps(self.schema, indent=2))

        except FileNotFoundError:
            raise FileNotFoundError("schema.json not found. Please ensure the file exists in the script's directory.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the schema: {e}")


    def _encode_file(self, file_content: bytes) -> str:
        """Encode image data as base64."""
        return base64.b64encode(file_content).decode('utf-8')

    def _validate_file_size(self, file_path: Path) -> None:
        """Validate file size doesn't exceed limit."""
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            raise DocumentProcessingError(
                f"File size ({file_size_mb:.2f}MB) exceeds limit of {self.config.max_file_size_mb}MB"
            )

    # def _pdf_to_images(self, pdf_path: Path) -> List[bytes]:
    #     """Convert PDF pages to images."""
    #     try:
    #         images = []
    #         with open(pdf_path, 'rb') as file:
    #             pdf_reader = pypdf.PdfReader(file)

    #             for page_num in range(len(pdf_reader.pages)):
    #                 page = pdf_reader.pages[page_num]

    #                 # Convert page to image using PDF rendering
    #                 # Note: This is a simplified approach. For production,
    #                 # consider using pdf2image or similar libraries

    #                 # For now, we'll extract text and create a simple image
    #                 # In a real implementation, you'd want proper PDF-to-image conversion
    #                 text = page.extract_text()

    #                 # Create a simple image with the text (placeholder)
    #                 img = Image.new('RGB', (800, 600), color='white')

    #                 # Convert to bytes
    #                 img_bytes = BytesIO()
    #                 img.save(img_bytes, format='PNG')
    #                 images.append(img_bytes.getvalue())

    #         return images
    #     except Exception as e:
    #         raise DocumentProcessingError(f"Failed to process PDF: {str(e)}")

    # async def extract_text_from_image(
    #     self,
    #     image_path: Path,
    #     prompt: Optional[str] = None
    # ) -> str:
    #     """Extract text from an image using Mistral AI vision model."""
    #     try:
    #         self._validate_file_size(image_path)

    #         with open(image_path, 'rb') as file:
    #             image_data = file.read()

    #         encoded_content = self._encode_file(image_data)

    #         default_prompt = (
    #             "Extract all text from this image. "
    #             "Maintain the original structure and formatting as much as possible. "
    #             "If the image contains tables, preserve the table structure. "
    #             "Return only the extracted text without any additional commentary."
    #         )

    #         messages = [
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {
    #                         "type": "text",
    #                         "text": prompt or default_prompt
    #                     },
    #                     {
    #                         "type": "image_url",
    #                         "image_url": f"data:image/jpeg;base64,{encoded_content}"
    #                     }
    #                 ]
    #             }
    #         ]

    #         response = await asyncio.to_thread(
    #             self.client.chat.complete,
    #             model=self.config.mistral_model,
    #             messages=messages,
    #             max_tokens=4000
    #         )

    #         return response.choices[0].message.content.strip()

    #     except Exception as e:
    #         if "api" in str(e).lower():
    #             raise APIError(f"Mistral API error: {str(e)}")
    #         else:
    #             raise OCRError(f"OCR processing failed: {str(e)}")

    # async def extract_text_from_pdf(
    #     self,
    #     pdf_path: Path,
    #     prompt: Optional[str] = None
    # ) -> List[str]:
    #     """Extract text from each page of a PDF."""
    #     try:
    #         self._validate_file_size(pdf_path)

    #         image_pages = self._pdf_to_images(pdf_path)
    #         results = []

    #         for i, image_data in enumerate(image_pages):
    #             encoded_content = self._encode_file(image_data)

    #             page_prompt = (
    #                 f"Extract all text from page {i+1} of this PDF. "
    #                 + (prompt or "Maintain original formatting and structure.")
    #             )

    #             messages = [
    #                 {
    #                     "role": "user",
    #                     "content": [
    #                         {
    #                             "type": "text",
    #                             "text": page_prompt
    #                         },
    #                         {
    #                             "type": "image_url",
    #                             "image_url": f"data:image/png;base64,{encoded_content}"
    #                         }
    #                     ]
    #                 }
    #             ]

    #             response = await asyncio.to_thread(
    #                 self.client.chat.complete,
    #                 model=self.config.mistral_model,
    #                 messages=messages,
    #                 max_tokens=4000
    #             )

    #             results.append(response.choices[0].message.content.strip())

    #         return results

    #     except Exception as e:
    #         if "api" in str(e).lower():
    #             raise APIError(f"Mistral API error: {str(e)}")
    #         else:
    #             raise OCRError(f"PDF OCR processing failed: {str(e)}")

    async def extract_structured_data(
        self,
        file_path: Path,
        schema: Dict[str, Any],
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract structured data from an image based on a provided schema."""
        self._validate_file_size(file_path)

        try:
            with open(file_path, 'rb') as file:
                file_content = file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found at {file_path}. Please check the path.")

        try:
            encoded_content = self._encode_file(file_content)

            # 5. Create a DocumentFileChunk from the base64 content
            # You must provide the correct MIME type for the file.
            # document_file_chunk = DocumentFileChunk(
            #     base64=encoded_content,
            #     mime_type="application/pdf"
            # )

            # 6. Process the document using the Mistral OCR API
            try:
                print("Processing local document with Mistral OCR...")
                ocr_response = self.client.ocr.process(
                    model="mistral-ocr-latest",
                    # document=document_file_chunk,
                    document=encoded_content,
                    response_format=response_format_from_pydantic_model(RootModel)
                )

                # 7. Access the structured data
                structured_data = ocr_response.parsed
                print("\n✅ Document processed successfully!")
                print("\nStructured JSON Output:")
                result_json = structured_data.model_dump_json(indent=2)
                print(result_json)
                return result_json

            except Exception as e:
                print(f"\n❌ An error occurred during OCR processing: {e}")
                raise APIError(f"Mistral API error: {str(e)}")

        except Exception as e:
            if "api" in str(e).lower():
                raise APIError(f"Mistral API error: {str(e)}")
            else:
                raise OCRError(f"Structured data extraction failed: {str(e)}")
