"""
OCR Optimization Service

Provides token optimization techniques and vLLM-specific optimizations
to reduce costs and improve performance.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
import io


@dataclass
class OptimizationResult:
    """Result of optimization."""
    original_tokens: int
    optimized_tokens: int
    tokens_saved: int
    savings_percent: float
    techniques_applied: List[str]
    metadata: Dict[str, Any]


class TokenOptimizer:
    """
    Token optimization for vision models.

    Techniques:
    - Image resolution optimization
    - Compression without quality loss
    - Prompt engineering for fewer tokens
    - Batch processing optimization
    """

    def __init__(self):
        self.default_max_dimension = 2048
        self.compression_quality = 85
        self.token_estimation_rate = 0.75  # tokens per character estimate

    def optimize_image(
        self,
        image_path: str,
        max_dimension: int = None,
        quality: int = None,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Optimize image for vision model processing.

        Args:
            image_path: Path to image
            max_dimension: Maximum width/height
            quality: JPEG quality (1-100)

        Returns:
            Tuple of (optimized image bytes, metadata)
        """
        max_dim = max_dimension or self.default_max_dimension
        qual = quality or self.compression_quality

        with Image.open(image_path) as img:
            original_size = img.size
            original_bytes = len(open(image_path, 'rb').read())

            # Calculate new dimensions
            width, height = img.size
            if width > max_dim or height > max_dim:
                if width > height:
                    new_width = max_dim
                    new_height = int(height * (max_dim / width))
                else:
                    new_height = max_dim
                    new_width = int(width * (max_dim / height))

                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convert to RGB if needed
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')

            # Save optimized
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=qual, optimize=True)
            optimized_bytes = buffer.getvalue()

            metadata = {
                "original_size": original_size,
                "optimized_size": img.size,
                "original_bytes": original_bytes,
                "optimized_bytes": len(optimized_bytes),
                "compression_ratio": len(optimized_bytes) / original_bytes,
                "size_reduction_percent": (
                    (1 - len(optimized_bytes) / original_bytes) * 100
                ),
            }

            return optimized_bytes, metadata

    def optimize_prompt(
        self,
        original_prompt: str,
        optimization_level: str = "moderate"
    ) -> Tuple[str, OptimizationResult]:
        """
        Optimize prompt to reduce tokens while maintaining intent.

        Args:
            original_prompt: Original prompt text
            optimization_level: "light", "moderate", "aggressive"

        Returns:
            Tuple of (optimized prompt, optimization result)
        """
        original_tokens = self._estimate_tokens(original_prompt)
        optimized = original_prompt
        techniques = []

        if optimization_level in ["moderate", "aggressive"]:
            # Remove redundant words
            optimized = self._remove_redundancy(optimized)
            techniques.append("redundancy_removal")

            # Simplify sentence structure
            optimized = self._simplify_structure(optimized)
            techniques.append("structure_simplification")

        if optimization_level == "aggressive":
            # Use abbreviations
            optimized = self._use_abbreviations(optimized)
            techniques.append("abbreviations")

            # Remove filler words
            optimized = self._remove_fillers(optimized)
            techniques.append("filler_removal")

        optimized_tokens = self._estimate_tokens(optimized)
        saved_tokens = original_tokens - optimized_tokens

        result = OptimizationResult(
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            tokens_saved=saved_tokens,
            savings_percent=(saved_tokens / original_tokens * 100) if original_tokens > 0 else 0,
            techniques_applied=techniques,
            metadata={
                "original_prompt": original_prompt,
                "optimization_level": optimization_level,
            }
        )

        return optimized, result

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text."""
        # Simple estimation: ~0.75 tokens per character
        return int(len(text) * self.token_estimation_rate)

    def _remove_redundancy(self, text: str) -> str:
        """Remove redundant phrases."""
        # Remove common redundant patterns
        patterns = [
            (r'\s+please\s+', ' '),
            (r'\s+kindly\s+', ' '),
            (r'\s+just\s+', ' '),
            (r'\s+really\s+', ' '),
            (r'\s+actually\s+', ' '),
            (r'\s+basically\s+', ' '),
        ]

        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def _simplify_structure(self, text: str) -> str:
        """Simplify sentence structure."""
        # Replace verbose phrases with concise alternatives
        replacements = {
            r'in order to': 'to',
            r'due to the fact that': 'because',
            r'for the purpose of': 'for',
            r'at this point in time': 'now',
            r'has the ability to': 'can',
            r'is able to': 'can',
            r'make a decision': 'decide',
            r'make a determination': 'determine',
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def _use_abbreviations(self, text: str) -> str:
        """Use common abbreviations."""
        abbreviations = {
            r'\bextract all\b': 'extract',
            r'\bidentify and list\b': 'list',
            r'\ball of the\b': 'the',
            r'\beach and every\b': 'all',
        }

        for pattern, replacement in abbreviations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def _remove_fillers(self, text: str) -> str:
        """Remove filler words."""
        fillers = [
            r'\bvery\b', r'\breally\b', r'\bquite\b', r'\brather\b',
            r'\bperhaps\b', r'\bmaybe\b', r'\bsomewhat\b',
        ]

        for filler in fillers:
            text = re.sub(filler, '', text, flags=re.IGNORECASE)

        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text


class vLLMOptimizer:
    """
    vLLM-specific optimizations.

    Techniques:
    - Batch processing for throughput
    - Continuous batching
    - PagedAttention memory optimization
    - KV cache sharing
    """

    def __init__(self):
        self.batch_size = 32
        self.enable_chunked_prefill = True
        self.enable_prefix_caching = True

    def optimize_batch_inference(
        self,
        prompts: List[str],
        batch_size: Optional[int] = None,
    ) -> List[List[str]]:
        """
        Optimize prompts for batch inference.

        Args:
            prompts: List of prompts
            batch_size: Batch size (uses default if None)

        Returns:
            List of batched prompts
        """
        batch_size = batch_size or self.batch_size

        # Group similar-length prompts together for efficiency
        sorted_prompts = sorted(
            enumerate(prompts),
            key=lambda x: len(x[1])
        )

        batches = []
        for i in range(0, len(sorted_prompts), batch_size):
            batch = [p[1] for p in sorted_prompts[i:i + batch_size]]
            batches.append(batch)

        return batches

    def optimize_for_continuous_batching(
        self,
        prompts: List[str],
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """
        Prepare prompts for continuous batching in vLLM.

        Args:
            prompts: List of prompts
            max_tokens: Maximum tokens per request

        Returns:
            Optimized batch configuration
        """
        return {
            "prompts": prompts,
            "sampling_params": {
                "max_tokens": max_tokens,
                "temperature": 0.0,  # Deterministic for OCR
                "top_p": 1.0,
            },
            "use_beam_search": False,  # Faster for OCR
            "best_of": 1,
            "n": 1,
        }

    def enable_prefix_caching(
        self,
        system_prompt: str,
        user_prompts: List[str],
    ) -> List[str]:
        """
        Optimize prompts to leverage prefix caching.

        vLLM can cache common prefixes to reduce redundant computation.

        Args:
            system_prompt: Common system prompt
            user_prompts: List of user prompts

        Returns:
            Optimized prompts with shared prefix
        """
        # Add system prompt as prefix to leverage caching
        return [f"{system_prompt}\n\n{prompt}" for prompt in user_prompts]

    def optimize_kv_cache_usage(
        self,
        num_requests: int,
        avg_prompt_length: int,
        avg_output_length: int,
        gpu_memory_gb: int = 24,
    ) -> Dict[str, Any]:
        """
        Calculate optimal KV cache configuration.

        Args:
            num_requests: Expected concurrent requests
            avg_prompt_length: Average prompt length in tokens
            avg_output_length: Average output length in tokens
            gpu_memory_gb: Available GPU memory

        Returns:
            Optimal configuration dict
        """
        # Simplified calculation (actual vLLM does more complex allocation)
        bytes_per_token = 2  # FP16
        tokens_per_request = avg_prompt_length + avg_output_length

        total_tokens = num_requests * tokens_per_request
        kv_cache_size_gb = (total_tokens * bytes_per_token) / (1024**3)

        # Leave 30% for model weights and activations
        available_for_kv = gpu_memory_gb * 0.7

        if kv_cache_size_gb > available_for_kv:
            # Need to reduce batch size
            max_requests = int((available_for_kv * (1024**3)) / (tokens_per_request * bytes_per_token))
        else:
            max_requests = num_requests

        return {
            "max_num_seqs": max_requests,
            "max_num_batched_tokens": max_requests * tokens_per_request,
            "gpu_memory_utilization": 0.90,  # Use 90% of GPU memory
            "swap_space": 4,  # 4GB swap space
            "max_model_len": tokens_per_request,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
        }

    def create_vllm_config(
        self,
        model_name: str,
        gpu_type: str = "A40",
        tensor_parallel_size: int = 1,
    ) -> Dict[str, Any]:
        """
        Create optimized vLLM configuration.

        Args:
            model_name: Model identifier
            gpu_type: GPU type (A40, A100, H100)
            tensor_parallel_size: Number of GPUs for tensor parallelism

        Returns:
            vLLM configuration dict
        """
        gpu_memory_map = {
            "A40": 48,
            "A100": 80,
            "H100": 80,
            "A10": 24,
        }

        gpu_memory_gb = gpu_memory_map.get(gpu_type, 24)

        config = {
            "model": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": 0.90,
            "max_model_len": 4096,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 256,
            "swap_space": 4,
            "dtype": "float16",  # or "bfloat16" for H100
            "trust_remote_code": True,
        }

        # Adjust for GPU type
        if gpu_type == "H100":
            config["dtype"] = "bfloat16"
            config["max_num_seqs"] = 512
            config["max_num_batched_tokens"] = 16384

        return config


class BatchOptimizer:
    """
    Batch-level optimizations for large-scale OCR.
    """

    def __init__(self):
        self.token_optimizer = TokenOptimizer()
        self.vllm_optimizer = vLLMOptimizer()

    def optimize_batch(
        self,
        file_paths: List[str],
        provider: str = "mistral",
        optimization_level: str = "moderate",
    ) -> Dict[str, Any]:
        """
        Optimize entire batch for processing.

        Args:
            file_paths: List of file paths
            provider: OCR provider
            optimization_level: Optimization level

        Returns:
            Optimization plan
        """
        optimizations = {
            "image_optimization": [],
            "prompt_optimization": None,
            "batch_configuration": None,
            "estimated_cost_savings": 0.0,
        }

        # Optimize images
        for file_path in file_paths:
            try:
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    _, metadata = self.token_optimizer.optimize_image(file_path)
                    optimizations["image_optimization"].append({
                        "file": file_path,
                        "size_reduction": metadata["size_reduction_percent"],
                    })
            except Exception as e:
                print(f"Error optimizing {file_path}: {e}")

        # Optimize prompt
        default_prompt = "Extract all text from this document, maintaining original formatting and structure."
        optimized_prompt, prompt_result = self.token_optimizer.optimize_prompt(
            default_prompt,
            optimization_level
        )

        optimizations["prompt_optimization"] = {
            "original": default_prompt,
            "optimized": optimized_prompt,
            "tokens_saved": prompt_result.tokens_saved,
            "savings_percent": prompt_result.savings_percent,
        }

        # vLLM batch configuration
        if provider in ["vllm", "custom"]:
            batch_config = self.vllm_optimizer.optimize_for_continuous_batching(
                [optimized_prompt] * len(file_paths)
            )
            optimizations["batch_configuration"] = batch_config

        # Estimate cost savings
        avg_image_savings = sum(
            opt["size_reduction"]
            for opt in optimizations["image_optimization"]
        ) / max(len(optimizations["image_optimization"]), 1)

        prompt_savings = prompt_result.savings_percent

        total_savings = (avg_image_savings + prompt_savings) / 2
        optimizations["estimated_cost_savings"] = total_savings

        return optimizations


# Global optimizer instances
_token_optimizer: Optional[TokenOptimizer] = None
_vllm_optimizer: Optional[vLLMOptimizer] = None
_batch_optimizer: Optional[BatchOptimizer] = None


def get_token_optimizer() -> TokenOptimizer:
    """Get token optimizer instance."""
    global _token_optimizer
    if _token_optimizer is None:
        _token_optimizer = TokenOptimizer()
    return _token_optimizer


def get_vllm_optimizer() -> vLLMOptimizer:
    """Get vLLM optimizer instance."""
    global _vllm_optimizer
    if _vllm_optimizer is None:
        _vllm_optimizer = vLLMOptimizer()
    return _vllm_optimizer


def get_batch_optimizer() -> BatchOptimizer:
    """Get batch optimizer instance."""
    global _batch_optimizer
    if _batch_optimizer is None:
        _batch_optimizer = BatchOptimizer()
    return _batch_optimizer
