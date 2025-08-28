"""
Document handling service for Ocular OCR system.
"""

import mimetypes
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from ..core.enums import DocumentType
from ..core.exceptions import ValidationError, DocumentProcessingError
from ..providers.settings import OcularSettings

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for document handling and validation."""
    
    def __init__(self, settings: OcularSettings):
        self.settings = settings
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.supported_pdf_formats = {'.pdf'}
    
    def detect_document_type(self, file_path: Path) -> DocumentType:
        """Detect document type from file path."""
        
        if not file_path.exists():
            raise ValidationError(
                "File does not exist",
                field="file_path",
                value=str(file_path)
            )
        
        suffix = file_path.suffix.lower()
        
        if suffix in self.supported_image_formats:
            return DocumentType.IMAGE
        elif suffix in self.supported_pdf_formats:
            return DocumentType.PDF
        else:
            # Try to detect by MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            if mime_type:
                if mime_type.startswith('image/'):
                    return DocumentType.IMAGE
                elif mime_type == 'application/pdf':
                    return DocumentType.PDF
            
            raise ValidationError(
                f"Unsupported file format: {suffix}",
                field="file_format",
                value=suffix
            )
    
    def validate_file(self, file_path: Path) -> bool:
        """Validate file against configured constraints."""
        
        if not file_path.exists():
            raise ValidationError(
                "File does not exist",
                field="file_path",
                value=str(file_path)
            )
        
        if not file_path.is_file():
            raise ValidationError(
                "Path is not a file",
                field="file_path",
                value=str(file_path)
            )
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        max_size = self.settings.files.max_file_size_mb
        
        if file_size_mb > max_size:
            raise ValidationError(
                f"File size ({file_size_mb:.2f}MB) exceeds limit of {max_size}MB",
                field="file_size",
                value=file_size_mb
            )
        
        # Check file extension
        suffix = file_path.suffix.lower()
        allowed_extensions = [ext.lower() for ext in self.settings.files.allowed_extensions]
        
        if suffix not in allowed_extensions:
            raise ValidationError(
                f"File extension {suffix} not allowed. Allowed: {allowed_extensions}",
                field="file_extension",
                value=suffix
            )
        
        return True
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get comprehensive file information."""
        
        if not file_path.exists():
            raise ValidationError("File does not exist", field="file_path", value=str(file_path))
        
        stat = file_path.stat()
        
        info = {
            "path": str(file_path),
            "name": file_path.name,
            "stem": file_path.stem,
            "suffix": file_path.suffix,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified_time": stat.st_mtime,
            "document_type": self.detect_document_type(file_path).value,
        }
        
        # Add MIME type
        mime_type, encoding = mimetypes.guess_type(str(file_path))
        if mime_type:
            info["mime_type"] = mime_type
        if encoding:
            info["encoding"] = encoding
        
        # Add image-specific info if applicable
        if info["document_type"] == DocumentType.IMAGE.value:
            try:
                from PIL import Image
                with Image.open(file_path) as img:
                    info["image_width"] = img.width
                    info["image_height"] = img.height
                    info["image_mode"] = img.mode
                    info["image_format"] = img.format
            except Exception as e:
                logger.warning(f"Could not read image info for {file_path}: {e}")
        
        # Add PDF-specific info if applicable
        elif info["document_type"] == DocumentType.PDF.value:
            try:
                import pypdf
                with open(file_path, 'rb') as f:
                    pdf_reader = pypdf.PdfReader(f)
                    info["pdf_pages"] = len(pdf_reader.pages)
                    info["pdf_encrypted"] = pdf_reader.is_encrypted
                    
                    # Try to get metadata
                    if pdf_reader.metadata:
                        info["pdf_title"] = pdf_reader.metadata.get("/Title")
                        info["pdf_author"] = pdf_reader.metadata.get("/Author")
                        info["pdf_creator"] = pdf_reader.metadata.get("/Creator")
                        
            except Exception as e:
                logger.warning(f"Could not read PDF info for {file_path}: {e}")
        
        return info
    
    def validate_batch_files(self, file_paths: List[Path]) -> Dict[str, List[str]]:
        """Validate a batch of files and return validation results."""
        
        results = {
            "valid": [],
            "invalid": [],
            "errors": []
        }
        
        for file_path in file_paths:
            try:
                if self.validate_file(file_path):
                    results["valid"].append(str(file_path))
            except ValidationError as e:
                results["invalid"].append(str(file_path))
                results["errors"].append(f"{file_path}: {e.message}")
            except Exception as e:
                results["invalid"].append(str(file_path))
                results["errors"].append(f"{file_path}: Unexpected error - {str(e)}")
        
        return results
    
    def find_documents_in_directory(
        self, 
        directory: Path, 
        recursive: bool = False
    ) -> List[Path]:
        """Find all supported documents in a directory."""
        
        if not directory.exists():
            raise ValidationError(
                "Directory does not exist",
                field="directory",
                value=str(directory)
            )
        
        if not directory.is_dir():
            raise ValidationError(
                "Path is not a directory",
                field="directory",
                value=str(directory)
            )
        
        pattern = "**/*" if recursive else "*"
        all_files = directory.glob(pattern)
        
        supported_files = []
        allowed_extensions = {ext.lower() for ext in self.settings.files.allowed_extensions}
        
        for file_path in all_files:
            if (file_path.is_file() and 
                file_path.suffix.lower() in allowed_extensions):
                
                try:
                    # Quick validation
                    self.validate_file(file_path)
                    supported_files.append(file_path)
                except ValidationError:
                    # Skip invalid files
                    continue
        
        return supported_files
    
    def create_temp_file(self, content: bytes, suffix: str = "") -> Path:
        """Create a temporary file with content."""
        
        import tempfile
        
        temp_dir = self.settings.files.temp_dir
        if temp_dir:
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
                dir=temp_dir
            ) as f:
                f.write(content)
                return Path(f.name)
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(content)
                return Path(f.name)
    
    def cleanup_temp_file(self, file_path: Path) -> None:
        """Clean up a temporary file."""
        
        try:
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {file_path}: {e}")
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get supported file formats by type."""
        
        return {
            "image": list(self.supported_image_formats),
            "pdf": list(self.supported_pdf_formats),
            "all": list(self.supported_image_formats | self.supported_pdf_formats)
        }
    
    def estimate_processing_time(self, file_path: Path) -> float:
        """Estimate processing time based on file characteristics."""
        
        info = self.get_file_info(file_path)
        base_time = 5.0  # Base processing time in seconds
        
        # Adjust based on file size
        size_mb = info["size_mb"]
        size_factor = min(size_mb / 10.0, 3.0)  # Cap at 3x for very large files
        
        # Adjust based on document type
        if info["document_type"] == DocumentType.PDF.value:
            # PDFs generally take longer, especially multi-page
            pdf_pages = info.get("pdf_pages", 1)
            page_factor = min(pdf_pages * 0.5, 5.0)  # Cap at 5x for many pages
            base_time += page_factor
        
        elif info["document_type"] == DocumentType.IMAGE.value:
            # Large images take longer
            if "image_width" in info and "image_height" in info:
                pixel_count = info["image_width"] * info["image_height"]
                pixel_factor = min(pixel_count / 1000000.0, 2.0)  # Cap at 2x
                base_time *= (1 + pixel_factor * 0.5)
        
        return base_time * (1 + size_factor)
    
    async def preprocess_file(self, file_path: Path) -> Path:
        """Preprocess file if needed (placeholder for future enhancement)."""
        
        # For now, just return the original file
        # Future enhancements could include:
        # - Image enhancement
        # - PDF optimization
        # - Format conversion
        
        return file_path