"""
Document processing module for Ocular.
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from enum import Enum

from .client import MistralOCRClient
from .config import OcularConfig
from .exceptions import DocumentProcessingError


class DocumentType(Enum):
    """Supported document types."""
    IMAGE = "image"
    PDF = "pdf"


class ProcessingResult:
    """Result of document processing."""
    
    def __init__(
        self,
        file_path: Path,
        document_type: DocumentType,
        extracted_text: Union[str, List[str]],
        structured_data: Optional[Dict[str, Any]] = None,
        processing_time: Optional[float] = None
    ):
        self.file_path = file_path
        self.document_type = document_type
        self.extracted_text = extracted_text
        self.structured_data = structured_data
        self.processing_time = processing_time
    
    def get_full_text(self) -> str:
        """Get all extracted text as a single string."""
        if isinstance(self.extracted_text, str):
            return self.extracted_text
        else:
            return "\n\n".join(self.extracted_text)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "file_path": str(self.file_path),
            "document_type": self.document_type.value,
            "extracted_text": self.extracted_text,
            "structured_data": self.structured_data,
            "processing_time": self.processing_time
        }


class DocumentProcessor:
    """Main document processor using Mistral AI OCR."""
    
    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    SUPPORTED_PDF_FORMATS = {'.pdf'}
    
    def __init__(self, config: Optional[OcularConfig] = None):
        """Initialize the document processor."""
        self.config = config or OcularConfig.from_env()
        self.ocr_client = MistralOCRClient(self.config)
    
    def _detect_document_type(self, file_path: Path) -> DocumentType:
        """Detect the type of document based on file extension."""
        suffix = file_path.suffix.lower()
        
        if suffix in self.SUPPORTED_IMAGE_FORMATS:
            return DocumentType.IMAGE
        elif suffix in self.SUPPORTED_PDF_FORMATS:
            return DocumentType.PDF
        else:
            raise DocumentProcessingError(
                f"Unsupported file format: {suffix}. "
                f"Supported formats: {self.SUPPORTED_IMAGE_FORMATS | self.SUPPORTED_PDF_FORMATS}"
            )
    
    async def process_document(
        self, 
        file_path: Union[str, Path],
        prompt: Optional[str] = None
    ) -> ProcessingResult:
        """Process a single document (image or PDF)."""
        import time
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise DocumentProcessingError(f"File not found: {file_path}")
        
        document_type = self._detect_document_type(file_path)
        start_time = time.time()
        
        try:
            if document_type == DocumentType.IMAGE:
                extracted_text = await self.ocr_client.extract_text_from_image(
                    file_path, prompt
                )
            else:  # PDF
                extracted_text = await self.ocr_client.extract_text_from_pdf(
                    file_path, prompt
                )
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                file_path=file_path,
                document_type=document_type,
                extracted_text=extracted_text,
                processing_time=processing_time
            )
            
        except Exception as e:
            raise DocumentProcessingError(f"Failed to process {file_path}: {str(e)}")
    
    async def process_batch(
        self, 
        file_paths: List[Union[str, Path]],
        prompt: Optional[str] = None,
        max_concurrent: int = 3
    ) -> List[ProcessingResult]:
        """Process multiple documents concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(file_path):
            async with semaphore:
                return await self.process_document(file_path, prompt)
        
        tasks = [process_with_semaphore(fp) for fp in file_paths]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def extract_structured_data(
        self,
        file_path: Union[str, Path],
        schema: Dict[str, Any],
        prompt: Optional[str] = None
    ) -> ProcessingResult:
        """Extract structured data from a document."""
        import time
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise DocumentProcessingError(f"File not found: {file_path}")
        
        document_type = self._detect_document_type(file_path)
        
        if document_type != DocumentType.IMAGE:
            raise DocumentProcessingError(
                "Structured data extraction currently only supports images"
            )
        
        start_time = time.time()
        
        try:
            structured_data = await self.ocr_client.extract_structured_data(
                file_path, schema, prompt
            )
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                file_path=file_path,
                document_type=document_type,
                extracted_text="",  # Not the focus for structured extraction
                structured_data=structured_data,
                processing_time=processing_time
            )
            
        except Exception as e:
            raise DocumentProcessingError(f"Failed to extract structured data from {file_path}: {str(e)}")
    
    async def process_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = False,
        prompt: Optional[str] = None,
        max_concurrent: int = 3
    ) -> List[ProcessingResult]:
        """Process all supported documents in a directory."""
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise DocumentProcessingError(f"Directory not found: {directory_path}")
        
        if not directory_path.is_dir():
            raise DocumentProcessingError(f"Path is not a directory: {directory_path}")
        
        # Find all supported files
        pattern = "**/*" if recursive else "*"
        all_files = directory_path.glob(pattern)
        
        supported_files = []
        for file_path in all_files:
            if file_path.is_file():
                try:
                    self._detect_document_type(file_path)
                    supported_files.append(file_path)
                except DocumentProcessingError:
                    continue  # Skip unsupported files
        
        if not supported_files:
            raise DocumentProcessingError(f"No supported files found in {directory_path}")
        
        return await self.process_batch(supported_files, prompt, max_concurrent)