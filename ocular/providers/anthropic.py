import anthropic
import json
import os

class ClaudeUnstructuredParser:
    def __init__(self, api_key: Optional[str] = None):
        self.client = anthropic.Anthropic(
            api_key=api_key or os.getenv('ANTHROPIC_API_KEY')
        )
    
    def parse_to_json(self, unstructured_text: str, schema_hint: str = "") -> Dict[str, Any]:
        """Convert unstructured text to JSON using Claude"""
        
        prompt = f"""
        Convert this unstructured text to valid JSON format:

        {unstructured_text}

        {schema_hint}

        Extract all relevant information and structure it logically. Return only the JSON, no explanations.
        """
        
        try:
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            json_string = response.content[0].text.strip()
            
            # Clean up the response
            if json_string.startswith('```json'):
                json_string = json_string[7:-3]
            elif json_string.startswith('```'):
                json_string = json_string[3:-3]
            
            return json.loads(json_string)
            
        except Exception as e:
            raise ValueError(f"Error with Claude API: {e}")