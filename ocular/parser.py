import json
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class ParseResult:
    success: bool
    data: Optional[Dict[str, Any]]
    error: Optional[str]
    confidence: float

class RobustUnstructuredParser:
    def __init__(self, api_key: str):
        self.openai_client = openai.OpenAI(api_key=api_key)
        self.max_retries = 3
    
    def parse_with_validation(
        self, 
        unstructured_text: str, 
        expected_fields: Optional[List[str]] = None,
        schema_hint: str = ""
    ) -> ParseResult:
        """Parse with validation and retry logic"""
        
        for attempt in range(self.max_retries):
            try:
                result = self._attempt_parse(unstructured_text, schema_hint, attempt)
                
                if self._validate_result(result, expected_fields):
                    confidence = self._calculate_confidence(result, unstructured_text)
                    return ParseResult(True, result, None, confidence)
                    
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return ParseResult(False, None, str(e), 0.0)
                continue
        
        return ParseResult(False, None, "Max retries exceeded", 0.0)
    
    def _attempt_parse(self, text: str, schema_hint: str, attempt: int) -> Dict[str, Any]:
        """Single parsing attempt with adjusted prompt based on retry count"""
        
        temperature = 0.1 + (attempt * 0.1)  # Increase creativity on retries
        
        prompt = f"""
        Convert this unstructured text to valid JSON. 
        
        {'Be more creative and extract implicit information.' if attempt > 0 else ''}
        
        {schema_hint}
        
        Text: {text}
        
        Return only valid JSON:
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a precise data extraction expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1000
        )
        
        json_text = self._clean_json_response(response.choices[0].message.content)
        return json.loads(json_text)
    
    def _clean_json_response(self, response: str) -> str:
        """Clean up AI response to get pure JSON"""
        response = response.strip()
        
        # Remove code block markers
        response = re.sub(r'^```(?:json)?\n?', '', response)
        response = re.sub(r'\n?```$', '', response)
        
        return response
    
    def _validate_result(self, result: Dict[str, Any], expected_fields: Optional[List[str]]) -> bool:
        """Validate the parsed result"""
        if not isinstance(result, dict):
            return False
        
        if expected_fields:
            for field in expected_fields:
                if field not in result:
                    return False
        
        return True
    
    def _calculate_confidence(self, result: Dict[str, Any], original_text: str) -> float:
        """Calculate confidence score based on information extraction"""
        # Simple heuristic: ratio of structured data to original text length
        structured_info = json.dumps(result)
        return min(1.0, len(structured_info) / len(original_text))

# Usage
def example_usage():
    parser = RobustUnstructuredParser(api_key="your-openai-api-key")
    
    text = "John Smith, age 35, software developer at Google, lives in Mountain View, salary $150,000"
    
    result = parser.parse_with_validation(
        text,
        expected_fields=["name", "age", "occupation"],
        schema_hint="Extract: name, age, occupation, company, location, salary"
    )
    
    if result.success:
        print(f"Parsed successfully (confidence: {result.confidence:.2f}):")
        print(json.dumps(result.data, indent=2))
    else:
        print(f"Parsing failed: {result.error}")