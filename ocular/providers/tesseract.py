"""
Tesseract OCR provider implementation for local processing.
"""

import asyncio
import json
import tempfile
import shutil
from typing import Optional, Dict, Any, List
from pathlib import Path

from .base import BaseOCRProvider
from ..core.models import OCRResult
from ..core.enums import ProviderType
from ..core.exceptions import (
    OCRError, ConfigurationError, ValidationError
)
from ..core.logging import log_async_function_call
from ..core.validation import text_validator, file_validator

try:
    import pytesseract
    from PIL import Image
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    Image = None


class TesseractProvider(BaseOCRProvider):
    """Tesseract OCR provider for local processing."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not HAS_TESSERACT:
            raise ConfigurationError(
                "Tesseract requires pytesseract and PIL. "
                "Install with: pip install pytesseract pillow",
                config_key="tesseract"
            )
        
        # Configuration
        self.tesseract_cmd = config.get("tesseract_cmd")  # Custom tesseract binary path
        self.language = config.get("language", "eng")
        self.config_params = config.get("config", "--psm 6 --oem 3")
        self.dpi = config.get("dpi", 300)
        self.timeout = config.get("timeout", 30)
        
        # Image preprocessing options
        self.preprocessing = config.get("preprocessing", {
            "enhance_contrast": True,
            "remove_noise": True,
            "correct_skew": False,
            "resize_factor": None
        })
        
        # Initialize tesseract
        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        
    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.TESSERACT
    
    @property
    def provider_name(self) -> str:
        return "Tesseract OCR"
    
    @log_async_function_call()
    async def _do_initialize(self) -> None:
        """Initialize Tesseract OCR."""
        try:
            # Test tesseract installation
            await self._test_tesseract()
            
            # Validate language data
            await self._validate_languages()
            
            self.logger.info(
                f"Initialized Tesseract OCR",
                language=self.language,
                config_params=self.config_params,
                dpi=self.dpi
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Tesseract OCR: {e}")
            raise ConfigurationError(
                f"Tesseract initialization failed: {e}",
                config_key="tesseract"
            )
    
    async def _test_tesseract(self) -> None:
        """Test Tesseract installation and accessibility."""
        try:
            def _test():
                # Create a simple test image
                test_image = Image.new('RGB', (100, 30), color='white')
                # Test basic functionality
                return pytesseract.get_tesseract_version()
            
            version = await asyncio.to_thread(_test)
            self.logger.info(f"Tesseract version: {version}")
            
        except pytesseract.TesseractNotFoundError:
            raise ConfigurationError(
                "Tesseract executable not found. Please install Tesseract OCR "
                "and ensure it's in your PATH or set tesseract_cmd config.",
                config_key="tesseract_cmd"
            )
        except Exception as e:
            raise ConfigurationError(f"Tesseract test failed: {e}")
    
    async def _validate_languages(self) -> None:
        """Validate that required language data is available."""
        try:
            def _get_languages():
                return pytesseract.get_languages(config='')
            
            available_languages = await asyncio.to_thread(_get_languages)
            
            # Parse language specification (e.g., "eng+fra" -> ["eng", "fra"])
            required_langs = self.language.split('+')
            missing_langs = [lang for lang in required_langs if lang not in available_languages]
            
            if missing_langs:
                raise ConfigurationError(
                    f"Missing Tesseract language data: {missing_langs}. "
                    f"Available languages: {available_languages}",
                    config_key="language"
                )
                
        except Exception as e:
            if "language data" in str(e).lower():
                raise
            else:
                self.logger.warning(f"Could not validate language data: {e}")
    
    @log_async_function_call()
    async def is_available(self) -> bool:
        """Check if Tesseract is available."""
        try:
            await self._test_tesseract()
            return True
        except Exception as e:
            self.logger.warning(f"Tesseract availability check failed: {e}")
            return False
    
    @log_async_function_call()
    async def _extract_text_impl(
        self, 
        file_path: Path, 
        prompt: Optional[str] = None,
        **kwargs
    ) -> OCRResult:
        """Extract text using Tesseract OCR."""
        
        self.logger.info(
            f"Starting Tesseract OCR for {file_path.name}",
            file_size=file_path.stat().st_size,
            language=self.language
        )
        
        try:
            # Handle PDF files
            if file_path.suffix.lower() == '.pdf':
                return await self._extract_from_pdf(file_path, prompt, **kwargs)
            else:
                return await self._extract_from_image(file_path, prompt, **kwargs)
                
        except Exception as e:
            self.logger.error(
                f"Tesseract OCR failed for {file_path.name}: {e}",
                exc_info=True
            )
            raise
    
    async def _extract_from_image(
        self, 
        file_path: Path, 
        prompt: Optional[str] = None,
        **kwargs
    ) -> OCRResult:
        """Extract text from image file."""
        
        def _process_image():
            # Load image
            image = Image.open(file_path)
            
            # Preprocess image if needed
            if self.preprocessing.get("enhance_contrast") or self.preprocessing.get("remove_noise"):
                image = self._preprocess_image(image)
            
            # Configure tesseract parameters
            config_params = self.config_params
            if kwargs.get("custom_config"):
                config_params = kwargs["custom_config"]
            
            # Extract text with confidence data
            ocr_data = pytesseract.image_to_data(
                image,
                lang=self.language,
                config=config_params,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract plain text
            text = pytesseract.image_to_string(
                image,
                lang=self.language,
                config=config_params
            )
            
            # Calculate confidence
            confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return text.strip(), avg_confidence / 100.0, ocr_data
        
        text, confidence, ocr_data = await asyncio.to_thread(_process_image)
        
        return OCRResult(
            text=text,
            confidence=confidence,
            language=self.language,
            provider=self.provider_name,
            metadata={
                "config_params": self.config_params,
                "dpi": self.dpi,
                "preprocessing": self.preprocessing,
                "word_count": len([w for w in ocr_data.get('text', []) if w.strip()]),
                "detected_words": sum(1 for conf in ocr_data.get('conf', []) if int(conf) > 0)
            }
        )
    
    async def _extract_from_pdf(
        self, 
        file_path: Path, 
        prompt: Optional[str] = None,
        **kwargs
    ) -> OCRResult:
        """Extract text from PDF file by converting to images."""
        try:
            # Try to import PDF processing library
            try:
                from pdf2image import convert_from_path
            except ImportError:
                raise ConfigurationError(
                    "PDF processing requires pdf2image. Install with: pip install pdf2image",
                    config_key="pdf2image"
                )
            
            def _process_pdf():
                # Convert PDF to images
                images = convert_from_path(
                    file_path, 
                    dpi=self.dpi,
                    first_page=kwargs.get('first_page'),
                    last_page=kwargs.get('last_page')
                )
                
                all_text = []
                all_confidences = []
                total_words = 0
                
                for page_num, image in enumerate(images):
                    # Preprocess image if needed
                    if self.preprocessing.get("enhance_contrast") or self.preprocessing.get("remove_noise"):
                        image = self._preprocess_image(image)
                    
                    # Extract text from page
                    page_text = pytesseract.image_to_string(
                        image,
                        lang=self.language,
                        config=self.config_params
                    )
                    
                    # Get confidence data
                    ocr_data = pytesseract.image_to_data(
                        image,
                        lang=self.language,
                        config=self.config_params,
                        output_type=pytesseract.Output.DICT
                    )
                    
                    page_confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
                    
                    if page_text.strip():
                        all_text.append(f"--- Page {page_num + 1} ---")
                        all_text.append(page_text.strip())
                        all_confidences.extend(page_confidences)
                        total_words += len([w for w in ocr_data.get('text', []) if w.strip()])
                
                combined_text = '\n\n'.join(all_text)
                avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
                
                return combined_text, avg_confidence / 100.0, total_words, len(images)
            
            text, confidence, word_count, page_count = await asyncio.to_thread(_process_pdf)
            
            return OCRResult(
                text=text,
                confidence=confidence,
                language=self.language,
                provider=self.provider_name,
                metadata={
                    "config_params": self.config_params,
                    "dpi": self.dpi,
                    "preprocessing": self.preprocessing,
                    "page_count": page_count,
                    "word_count": word_count,
                    "document_type": "pdf"
                }
            )
            
        except Exception as e:
            raise OCRError(f"PDF processing failed: {e}", provider=self.provider_name)
    
    def _preprocess_image(self, image):
        """Apply image preprocessing to improve OCR accuracy."""
        try:
            from PIL import ImageEnhance, ImageFilter
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image if specified
            if self.preprocessing.get("resize_factor"):
                factor = self.preprocessing["resize_factor"]
                new_size = (int(image.width * factor), int(image.height * factor))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Enhance contrast
            if self.preprocessing.get("enhance_contrast"):
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.5)
            
            # Remove noise
            if self.preprocessing.get("remove_noise"):
                image = image.filter(ImageFilter.MedianFilter(size=3))
            
            # Correct skew (basic implementation)
            if self.preprocessing.get("correct_skew"):
                # This would require more sophisticated skew detection
                # For now, just return the image as-is
                pass
            
            return image
            
        except Exception as e:
            self.logger.warning(f"Image preprocessing failed: {e}")
            return image
    
    async def _extract_structured_data_impl(
        self,
        file_path: Path,
        schema: Dict[str, Any],
        prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Extract structured data using Tesseract with layout analysis."""
        
        def _extract_with_layout():
            # Load image
            image = Image.open(file_path) if file_path.suffix.lower() != '.pdf' else None
            
            if image is None:
                # For PDFs, use first page
                from pdf2image import convert_from_path
                images = convert_from_path(file_path, dpi=self.dpi, last_page=1)
                image = images[0] if images else None
            
            if image is None:
                raise OCRError("Could not load image for structured extraction")
            
            # Get detailed OCR data
            ocr_data = pytesseract.image_to_data(
                image,
                lang=self.language,
                config=self.config_params,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text
            text = pytesseract.image_to_string(
                image,
                lang=self.language,
                config=self.config_params
            )
            
            # Group words by lines and blocks
            lines = self._group_words_by_lines(ocr_data)
            blocks = self._group_lines_by_blocks(lines)
            
            return {
                "text": text.strip(),
                "lines": lines,
                "blocks": blocks,
                "word_count": len([w for w in ocr_data.get('text', []) if w.strip()]),
                "confidence": sum(int(c) for c in ocr_data.get('conf', []) if int(c) > 0) / 
                            len([c for c in ocr_data.get('conf', []) if int(c) > 0]) / 100.0
            }
        
        return await asyncio.to_thread(_extract_with_layout)
    
    def _group_words_by_lines(self, ocr_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Group words into lines based on OCR data."""
        lines = {}
        
        for i, text in enumerate(ocr_data.get('text', [])):
            if text.strip():
                line_num = ocr_data['line_num'][i]
                block_num = ocr_data['block_num'][i]
                
                key = (block_num, line_num)
                if key not in lines:
                    lines[key] = {
                        'text': [],
                        'confidence': [],
                        'bbox': {
                            'left': ocr_data['left'][i],
                            'top': ocr_data['top'][i],
                            'width': ocr_data['width'][i],
                            'height': ocr_data['height'][i]
                        }
                    }
                
                lines[key]['text'].append(text)
                lines[key]['confidence'].append(int(ocr_data['conf'][i]))
        
        # Convert to list format
        result_lines = []
        for (block_num, line_num), line_data in sorted(lines.items()):
            result_lines.append({
                'text': ' '.join(line_data['text']),
                'confidence': sum(line_data['confidence']) / len(line_data['confidence']) / 100.0,
                'bbox': line_data['bbox'],
                'block_num': block_num,
                'line_num': line_num
            })
        
        return result_lines
    
    def _group_lines_by_blocks(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group lines into blocks."""
        blocks = {}
        
        for line in lines:
            block_num = line['block_num']
            if block_num not in blocks:
                blocks[block_num] = {
                    'lines': [],
                    'text': [],
                    'confidence': []
                }
            
            blocks[block_num]['lines'].append(line)
            blocks[block_num]['text'].append(line['text'])
            blocks[block_num]['confidence'].append(line['confidence'])
        
        # Convert to list format
        result_blocks = []
        for block_num, block_data in sorted(blocks.items()):
            result_blocks.append({
                'text': '\n'.join(block_data['text']),
                'confidence': sum(block_data['confidence']) / len(block_data['confidence']),
                'line_count': len(block_data['lines']),
                'block_num': block_num
            })
        
        return result_blocks
    
    def validate_config(self) -> bool:
        """Validate Tesseract provider configuration."""
        # Check language format
        if not self.language or not isinstance(self.language, str):
            return False
        
        # Check DPI
        if self.dpi < 72 or self.dpi > 600:
            return False
        
        # Check timeout
        if self.timeout <= 0 or self.timeout > 300:
            return False
        
        return True
    
    def get_supported_formats(self) -> List[str]:
        """Get supported formats for Tesseract OCR."""
        return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.pdf']
    
    async def cleanup(self) -> None:
        """Clean up Tesseract resources."""
        # No persistent resources to clean up for Tesseract
        pass

    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence based on text quality metrics."""
        if not text:
            return 0.0
        
        # Base confidence for Tesseract
        confidence = 0.8
        
        # Adjust based on text characteristics
        if len(text.split()) < 5:
            confidence -= 0.1
        
        if len(text) > 100:
            confidence += 0.05
        
        return max(0.0, min(1.0, confidence))