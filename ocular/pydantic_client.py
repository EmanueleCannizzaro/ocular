"""
PydanticAI-based OCR client for Mistral AI.
"""

import base64
from typing import Optional, Dict, Any, List
from pathlib import Path
from io import BytesIO

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel
from PIL import Image
import pypdf

from .config import OcularConfig
from .exceptions import OCRError, APIError, DocumentProcessingError


class OCRResult(BaseModel):
    """Structured OCR result using Pydantic."""
    extracted_text: str
    confidence_score: Optional[float] = None
    language: Optional[str] = None
    

class StructuredData(BaseModel):
    """Base class for structured data extraction."""
    pass


class InvoiceData(StructuredData):
    """Example structured data for invoice extraction."""
    invoice_number: Optional[str] = None
    date: Optional[str] = None
    total_amount: Optional[float] = None
    vendor_name: Optional[str] = None
    items: Optional[List[Dict[str, Any]]] = None


class PydanticOCRClient:
    """OCR client using PydanticAI with Mistral AI vision models."""
    
    def __init__(self, config: Optional[OcularConfig] = None):
        """Initialize the PydanticAI OCR client."""
        self.config = config or OcularConfig.from_env()
        
        # Initialize Mistral model
        self.model = MistralModel(
            model_name=self.config.mistral_model,
            api_key=self.config.mistral_api_key
        )
        
        # Create OCR agent
        self.ocr_agent = Agent(
            self.model,
            result_type=OCRResult,
            system_prompt=(
                "You are an expert OCR system. Extract text from images accurately, "
                "maintaining original formatting and structure. Provide confidence scores "
                "when possible and identify the primary language of the text."
            )
        )
        
        # Create structured data agent  
        self.structured_agent = Agent(
            self.model,
            system_prompt=(
                "You are an expert data extraction system. Extract structured data "
                "from documents according to the provided schema. Be precise and "
                "ensure all extracted data is accurate."
            )
        )
    
    def _encode_image(self, image_data: bytes) -> str:
        """Encode image data as base64."""
        return base64.b64encode(image_data).decode('utf-8')
    
    def _validate_file_size(self, file_path: Path) -> None:
        """Validate file size doesn't exceed limit."""
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            raise DocumentProcessingError(
                f"File size ({file_size_mb:.2f}MB) exceeds limit of {self.config.max_file_size_mb}MB"
            )
    
    def _pdf_to_images(self, pdf_path: Path) -> List[bytes]:
        """Convert PDF pages to images."""
        try:
            images = []
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                
                for page_num in range(len(pdf_reader.pages)):
                    # Create placeholder image for PDF page
                    # In production, use pdf2image library
                    img = Image.new('RGB', (800, 600), color='white')
                    
                    img_bytes = BytesIO()
                    img.save(img_bytes, format='PNG')
                    images.append(img_bytes.getvalue())
                    
            return images
        except Exception as e:
            raise DocumentProcessingError(f"Failed to process PDF: {str(e)}")
    
    async def extract_text_from_image(
        self, 
        image_path: Path, 
        prompt: Optional[str] = None
    ) -> OCRResult:
        """Extract text from an image using PydanticAI."""
        try:
            self._validate_file_size(image_path)
            
            with open(image_path, 'rb') as file:
                image_data = file.read()
            
            base64_image = self._encode_image(image_data)
            
            user_prompt = (
                prompt or 
                "Extract all text from this image. Maintain original structure and formatting."
            )
            
            # Create message with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    ]
                }
            ]
            
            result = await self.ocr_agent.run(
                user_prompt,
                message_history=messages
            )
            
            return result.data
            
        except Exception as e:
            if "api" in str(e).lower():
                raise APIError(f"Mistral API error: {str(e)}")
            else:
                raise OCRError(f"OCR processing failed: {str(e)}")
    
    async def extract_text_from_pdf(
        self, 
        pdf_path: Path, 
        prompt: Optional[str] = None
    ) -> List[OCRResult]:
        """Extract text from each page of a PDF."""
        try:
            self._validate_file_size(pdf_path)
            
            image_pages = self._pdf_to_images(pdf_path)
            results = []
            
            for i, image_data in enumerate(image_pages):
                base64_image = self._encode_image(image_data)
                
                page_prompt = (
                    f"Extract text from page {i+1} of this PDF. "
                    + (prompt or "Maintain original formatting.")
                )
                
                messages = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": page_prompt},
                            {
                                "type": "image_url",
                                "image_url": f"data:image/png;base64,{base64_image}"
                            }
                        ]
                    }
                ]
                
                result = await self.ocr_agent.run(
                    page_prompt,
                    message_history=messages
                )
                
                results.append(result.data)
                
            return results
            
        except Exception as e:
            if "api" in str(e).lower():
                raise APIError(f"Mistral API error: {str(e)}")
            else:
                raise OCRError(f"PDF OCR processing failed: {str(e)}")
    
    async def extract_structured_data(
        self, 
        image_path: Path, 
        result_type: type[BaseModel],
        prompt: Optional[str] = None
    ) -> BaseModel:
        """Extract structured data using PydanticAI with custom result type."""
        try:
            self._validate_file_size(image_path)
            
            # Create agent with custom result type
            structured_agent = Agent(
                self.model,
                result_type=result_type,
                system_prompt=(
                    "Extract structured data from this document according to the provided schema. "
                    "Be precise and ensure all data is accurate. Return structured data only."
                )
            )
            
            with open(image_path, 'rb') as file:
                image_data = file.read()
            
            base64_image = self._encode_image(image_data)
            
            user_prompt = (
                prompt or 
                f"Extract data according to the {result_type.__name__} schema from this image."
            )
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    ]
                }
            ]
            
            result = await structured_agent.run(
                user_prompt,
                message_history=messages
            )
            
            return result.data
                
        except Exception as e:
            if "api" in str(e).lower():
                raise APIError(f"Mistral API error: {str(e)}")
            else:
                raise OCRError(f"Structured data extraction failed: {str(e)}")
    
    async def extract_invoice_data(
        self, 
        image_path: Path,
        prompt: Optional[str] = None
    ) -> InvoiceData:
        """Convenience method for extracting invoice data."""
        return await self.extract_structured_data(
            image_path,
            InvoiceData,
            prompt or "Extract invoice information including number, date, amount, vendor, and line items."
        )