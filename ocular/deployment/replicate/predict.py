"""
Replicate Prediction Handler for OLMocr with vLLM
This file handles predictions for the Replicate platform deployment
"""

import base64
import io
from typing import Dict, List, Any
from PIL import Image
from cog import BasePredictor, Input, Path

# Import vLLM components
try:
    import vllm
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available, using fallback mode")


class Predictor(BasePredictor):
    """Replicate predictor for OLMocr model using vLLM"""

    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print("Setting up OLMocr model with vLLM...")

        self.model_name = "allenai/olmOCR-7B-0225-preview"

        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM is required but not available")

        # Initialize vLLM engine
        print(f"Loading model: {self.model_name}")

        try:
            self.model = vllm.LLM(
                model=self.model_name,
                gpu_memory_utilization=0.9,
                dtype="bfloat16",
                trust_remote_code=True,
                max_model_len=4096,
                tensor_parallel_size=1,
                disable_log_stats=False,
                download_dir="/src/.cache/models"
            )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir="/src/.cache/huggingface",
                trust_remote_code=True
            )

            print("Model loaded successfully")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def predict(
        self,
        image: Path = Input(
            description="Input image (PNG, JPEG) or PDF file to extract text from"
        ),
        prompt: str = Input(
            description="Text extraction prompt",
            default="Extract all text from this document and convert it to markdown format. Preserve the structure, formatting, tables, and reading order."
        ),
        max_tokens: int = Input(
            description="Maximum number of tokens to generate",
            default=4096,
            ge=1,
            le=8192
        ),
        temperature: float = Input(
            description="Temperature for text generation",
            default=0.1,
            ge=0.0,
            le=2.0
        ),
        top_p: float = Input(
            description="Top-p sampling parameter",
            default=0.9,
            ge=0.0,
            le=1.0
        ),
        output_format: str = Input(
            description="Output format for extracted text",
            default="markdown",
            choices=["markdown", "plain_text", "json"]
        )
    ) -> Dict[str, Any]:
        """Run a single prediction on the model"""

        try:
            # Process the input image/document
            print(f"Processing input file: {image}")

            # Load and process the image
            if str(image).lower().endswith('.pdf'):
                # Handle PDF files
                processed_image = self._process_pdf(image)
            else:
                # Handle image files
                processed_image = self._process_image(image)

            # Convert image to base64
            image_b64 = self._image_to_base64(processed_image)

            # Format the prompt based on output format
            formatted_prompt = self._format_prompt(prompt, output_format)

            # Create the chat message format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": formatted_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ]

            # Generate response using vLLM
            response = self._generate_with_vllm(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )

            # Format the output
            result = {
                "extracted_text": response.get("choices", [{}])[0].get("message", {}).get("content", ""),
                "metadata": {
                    "model": self.model_name,
                    "output_format": output_format,
                    "processing_time": response.get("processing_time", 0),
                    "token_usage": response.get("usage", {})
                },
                "success": True
            }

            return result

        except Exception as e:
            print(f"Error during prediction: {e}")
            return {
                "extracted_text": "",
                "metadata": {
                    "model": self.model_name,
                    "error": str(e)
                },
                "success": False,
                "error": str(e)
            }

    def _process_pdf(self, pdf_path: Path) -> Image.Image:
        """Convert PDF to image for processing"""
        try:
            import pdf2image

            # Convert first page of PDF to image
            pages = pdf2image.convert_from_path(
                pdf_path,
                first_page=1,
                last_page=1,
                dpi=200,
                fmt='RGB'
            )

            if not pages:
                raise ValueError("Could not extract any pages from PDF")

            return pages[0]

        except ImportError:
            raise RuntimeError("pdf2image is required for PDF processing")
        except Exception as e:
            raise RuntimeError(f"Error processing PDF: {e}")

    def _process_image(self, image_path: Path) -> Image.Image:
        """Load and process image file"""
        try:
            image = Image.open(image_path)

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize if too large (maintain aspect ratio)
            max_dimension = 1024
            if max(image.size) > max_dimension:
                image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

            return image

        except Exception as e:
            raise RuntimeError(f"Error processing image: {e}")

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        try:
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=95)
            image_bytes = buffer.getvalue()
            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            raise RuntimeError(f"Error converting image to base64: {e}")

    def _format_prompt(self, prompt: str, output_format: str) -> str:
        """Format the prompt based on desired output format"""
        if output_format == "markdown":
            return f"{prompt} Format the output as clean markdown with proper headers, lists, and table formatting."
        elif output_format == "json":
            return f"{prompt} Format the output as valid JSON with extracted text organized by sections."
        else:  # plain_text
            return f"{prompt} Extract as plain text, preserving line breaks and structure."

    def _generate_with_vllm(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        top_p: float
    ) -> Dict[str, Any]:
        """Generate response using vLLM engine"""

        import time
        start_time = time.time()

        try:
            # Process messages to extract text and images
            prompt_text = ""
            for message in messages:
                if isinstance(message.get("content"), list):
                    for content_item in message["content"]:
                        if content_item.get("type") == "text":
                            prompt_text += content_item.get("text", "")
                elif isinstance(message.get("content"), str):
                    prompt_text += message["content"]

            # Create sampling parameters
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=["<|im_end|>"]
            )

            # Format prompt for the model
            formatted_prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"

            # Generate response
            outputs = self.model.generate([formatted_prompt], sampling_params)

            processing_time = time.time() - start_time

            # Format response in OpenAI-compatible format
            response = {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": outputs[0].outputs[0].text.strip()
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(self.tokenizer.encode(formatted_prompt)),
                    "completion_tokens": len(outputs[0].outputs[0].token_ids),
                    "total_tokens": len(self.tokenizer.encode(formatted_prompt)) + len(outputs[0].outputs[0].token_ids)
                },
                "model": self.model_name,
                "processing_time": processing_time
            }

            return response

        except Exception as e:
            print(f"Error during vLLM generation: {e}")
            raise RuntimeError(f"Generation failed: {e}")
