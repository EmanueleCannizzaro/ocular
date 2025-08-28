"""
Utility functions for Ocular package.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Union
from datetime import datetime

from .processor import ProcessingResult


def save_results_to_json(
    results: Union[ProcessingResult, List[ProcessingResult]], 
    output_path: Union[str, Path]
) -> None:
    """Save processing results to a JSON file."""
    output_path = Path(output_path)
    
    if isinstance(results, ProcessingResult):
        results = [results]
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "total_documents": len(results),
        "results": [result.to_dict() for result in results]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_text_to_file(
    results: Union[ProcessingResult, List[ProcessingResult]], 
    output_path: Union[str, Path],
    include_metadata: bool = True
) -> None:
    """Save extracted text to a plain text file."""
    output_path = Path(output_path)
    
    if isinstance(results, ProcessingResult):
        results = [results]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        if include_metadata:
            f.write(f"# OCR Results Generated on {datetime.now().isoformat()}\n")
            f.write(f"# Total Documents: {len(results)}\n\n")
        
        for i, result in enumerate(results, 1):
            if include_metadata:
                f.write(f"## Document {i}: {result.file_path.name}\n")
                f.write(f"Document Type: {result.document_type.value}\n")
                if result.processing_time:
                    f.write(f"Processing Time: {result.processing_time:.2f}s\n")
                f.write("\n")
            
            text = result.get_full_text()
            f.write(text)
            f.write("\n\n" + "="*50 + "\n\n")


def create_markdown_report(
    results: Union[ProcessingResult, List[ProcessingResult]], 
    output_path: Union[str, Path]
) -> None:
    """Create a markdown report of processing results."""
    output_path = Path(output_path)
    
    if isinstance(results, ProcessingResult):
        results = [results]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# OCR Processing Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Documents Processed:** {len(results)}\n\n")
        
        # Summary table
        f.write("## Summary\n\n")
        f.write("| File | Type | Processing Time | Text Length |\n")
        f.write("|------|------|----------------|-------------|\n")
        
        for result in results:
            text_length = len(result.get_full_text())
            processing_time = f"{result.processing_time:.2f}s" if result.processing_time else "N/A"
            f.write(f"| {result.file_path.name} | {result.document_type.value} | {processing_time} | {text_length} chars |\n")
        
        f.write("\n")
        
        # Detailed results
        f.write("## Detailed Results\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"### Document {i}: {result.file_path.name}\n\n")
            f.write(f"**File Path:** `{result.file_path}`\n\n")
            f.write(f"**Document Type:** {result.document_type.value}\n\n")
            
            if result.processing_time:
                f.write(f"**Processing Time:** {result.processing_time:.2f} seconds\n\n")
            
            if result.structured_data:
                f.write("**Structured Data:**\n\n")
                f.write("```json\n")
                f.write(json.dumps(result.structured_data, indent=2))
                f.write("\n```\n\n")
            
            f.write("**Extracted Text:**\n\n")
            f.write("```\n")
            f.write(result.get_full_text())
            f.write("\n```\n\n")
            f.write("---\n\n")


def validate_file_paths(file_paths: List[Union[str, Path]]) -> List[Path]:
    """Validate and convert file paths to Path objects."""
    validated_paths = []
    
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        validated_paths.append(path)
    
    return validated_paths


def calculate_processing_stats(results: List[ProcessingResult]) -> Dict[str, Any]:
    """Calculate processing statistics from results."""
    if not results:
        return {}
    
    processing_times = [r.processing_time for r in results if r.processing_time]
    text_lengths = [len(r.get_full_text()) for r in results]
    
    doc_types = {}
    for result in results:
        doc_type = result.document_type.value
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    
    stats = {
        "total_documents": len(results),
        "document_types": doc_types,
        "total_text_characters": sum(text_lengths),
        "average_text_length": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
    }
    
    if processing_times:
        stats.update({
            "total_processing_time": sum(processing_times),
            "average_processing_time": sum(processing_times) / len(processing_times),
            "fastest_processing_time": min(processing_times),
            "slowest_processing_time": max(processing_times),
        })
    
    return stats