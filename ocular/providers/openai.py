import openai
import json
import os
from typing import Dict, Any, Optional

class UnstructuredToJSONParser:
    def __init__(self, api_key: Optional[str] = None):
        # Set API key from parameter or environment variable
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv('OPENAI_API_KEY')
        )
    
    def parse_to_json(self, unstructured_text: str, schema_hint: str = "") -> Dict[str, Any]:
        """Convert unstructured text to JSON using OpenAI"""
        
        # Create the prompt
        prompt = f"""
        Convert the following unstructured text into valid JSON format.
        Extract all relevant information and organize it logically.
        
        {schema_hint}
        
        Unstructured text:
        {unstructured_text}
        
        Requirements:
        1. Return ONLY valid JSON, no explanations
        2. Use appropriate data types (strings, numbers, booleans, arrays)
        3. Create nested objects when logical groupings exist
        4. Use snake_case for keys
        5. If dates are present, use ISO format (YYYY-MM-DD)
        
        JSON:
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",  # or "gpt-3.5-turbo" for cheaper option
                messages=[
                    {"role": "system", "content": "You are a data extraction expert. Convert unstructured text to clean JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent output
                max_tokens=1000
            )
            
            json_string = response.choices[0].message.content.strip()
            
            # Remove potential code block markers
            if json_string.startswith('```json'):
                json_string = json_string[7:]
            if json_string.startswith('```'):
                json_string = json_string[3:]
            if json_string.endswith('```'):
                json_string = json_string[:-3]
            
            # Parse and validate JSON
            parsed_json = json.loads(json_string.strip())
            return parsed_json
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse AI response as JSON: {e}")
        except Exception as e:
            raise ValueError(f"Error calling OpenAI API: {e}")

# Usage example
def main():
    # Initialize parser
    parser = UnstructuredToJSONParser()
    
    # Example 1: Personal information
    text1 = """
    Hi there! My name is Sarah Johnson and I'm 28 years old. 
    I work as a Software Engineer at TechCorp Inc. 
    You can reach me at sarah.j@email.com or call me at (555) 123-4567.
    I live in San Francisco, California. I graduated from Stanford University in 2018.
    My hobbies include hiking, photography, and playing guitar.
    """
    
    schema_hint1 = """
    Expected structure:
    - personal_info: name, age, occupation, company
    - contact: email, phone
    - location: city, state
    - education: university, graduation_year
    - hobbies: array of interests
    """
    
    result1 = parser.parse_to_json(text1, schema_hint1)
    print("Example 1 - Personal Info:")
    print(json.dumps(result1, indent=2))
    print("\n" + "="*50 + "\n")
    
    # Example 2: Product review
    text2 = """
    Product Review: iPhone 14 Pro
    Rating: 4 out of 5 stars
    Purchased on March 15, 2023 for $999.99
    Pros: Great camera quality, excellent build quality, fast performance
    Cons: Battery life could be better, expensive
    Would I recommend? Yes, but only for users who need the pro features
    Reviewer: Mike Chen, verified purchase
    """
    
    result2 = parser.parse_to_json(text2)
    print("Example 2 - Product Review:")
    print(json.dumps(result2, indent=2))

if __name__ == "__main__":
    main()