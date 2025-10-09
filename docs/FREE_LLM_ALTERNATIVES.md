# Free & Open-Source LLM Alternatives for Property Extraction

This guide covers **100% FREE** alternatives for the AI structure conversion step (Step 3) in the property extraction workflow.

## Overview

The property extraction workflow has 3 steps:
1. **OCR Extraction** - Extract text from PDF (using GPU endpoint)
2. **Text Processing** - Clean and prepare OCR text
3. **AI Structure Conversion** - Convert unstructured text to structured JSON ⭐ **This step now has FREE options!**

## FREE LLM Options

### 🌟 Recommended: Mistral-7B-Instruct (HuggingFace)

**Cost:** 100% FREE
**Quality:** ⭐⭐⭐⭐
**Speed:** Medium
**Setup:** No setup required!

```python
from ocular.services.property_extraction_service import PropertyExtractionService

# Use free Mistral-7B via HuggingFace (default)
service = PropertyExtractionService(llm_provider="mistral-7b")

result = await service.process_property_appraisal(
    pdf_path="appraisal.pdf",
    provider_name="olm_ocr"
)
```

**REST API:**
```bash
curl -X POST "http://localhost:8004/api/property-extraction/extract" \
  -F "file=@appraisal.pdf" \
  -F "provider=olm_ocr" \
  -F "llm_provider=mistral-7b"
```

**Features:**
- No API key required
- High quality for Italian documents
- HuggingFace Inference API (free tier)
- ~7B parameters

---

### 🚀 Phi-3-Mini (Microsoft)

**Cost:** 100% FREE
**Quality:** ⭐⭐⭐
**Speed:** Fast
**Setup:** No setup required!

```python
service = PropertyExtractionService(llm_provider="phi-3-mini")
```

**Features:**
- Lightweight and fast
- Microsoft's efficient model
- Good for simpler documents
- ~3.8B parameters

---

### 🦙 Llama-3-8B-Instruct (Meta)

**Cost:** 100% FREE
**Quality:** ⭐⭐⭐⭐
**Speed:** Medium
**Setup:** No setup required!

```python
service = PropertyExtractionService(llm_provider="llama-3-8b")
```

**Features:**
- Meta's latest open model
- Strong performance
- Good Italian language support
- ~8B parameters

---

### 💻 Local Models via Ollama

**Cost:** 100% FREE
**Quality:** ⭐⭐⭐⭐
**Speed:** Fast (depends on hardware)
**Setup:** Install Ollama once

#### Setup:

1. Install Ollama:
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Or download from https://ollama.ai
```

2. Pull a model:
```bash
# Mistral 7B (recommended)
ollama pull mistral:7b-instruct

# Or other models:
ollama pull phi3:mini
ollama pull llama3:8b-instruct
```

3. Start Ollama server:
```bash
ollama serve
```

4. Use in workflow:
```python
service = PropertyExtractionService(
    llm_provider="ollama",
    llm_model="mistral:7b-instruct"
)
```

**Features:**
- 100% local, no internet required
- Complete privacy
- No API limits
- Run on your hardware (GPU recommended but not required)

---

## Comparison Table

| Provider | Cost | Quality | Speed | API Key | Setup Required |
|---|---|---|---|---|---|
| **Mistral-7B (HF)** | FREE | ⭐⭐⭐⭐ | Medium | No | No |
| **Phi-3-Mini** | FREE | ⭐⭐⭐ | Fast | No | No |
| **Llama-3-8B** | FREE | ⭐⭐⭐⭐ | Medium | No | No |
| **Ollama (Local)** | FREE | ⭐⭐⭐⭐ | Fast* | No | Yes (easy) |
| Anthropic Claude | $3/1M | ⭐⭐⭐⭐⭐ | Fast | Yes | No |
| OpenAI GPT-4o-mini | $0.15/1M | ⭐⭐⭐⭐ | Fast | Yes | No |

\* Speed depends on local hardware

---

## Usage Examples

### Python API

#### Free HuggingFace (Recommended)
```python
from ocular.services.property_extraction_service import extract_property_from_pdf

# Uses Mistral-7B via HuggingFace (FREE!)
property_data = await extract_property_from_pdf(
    pdf_path="appraisal.pdf",
    provider="olm_ocr"  # OCR provider
    # llm_provider defaults to "mistral-7b" (FREE)
)
```

#### Local Ollama
```python
service = PropertyExtractionService(
    llm_provider="ollama",
    llm_model="mistral:7b-instruct"
)

result = await service.process_property_appraisal(
    pdf_path="appraisal.pdf",
    provider_name="olm_ocr"
)
```

#### Compare Multiple Models
```python
# Try different LLMs and compare results
llm_providers = ["mistral-7b", "phi-3-mini", "llama-3-8b"]

for llm in llm_providers:
    service = PropertyExtractionService(llm_provider=llm)
    result = await service.process_property_appraisal("appraisal.pdf")
    print(f"{llm}: {result['metadata']['processing_time_seconds']}s")
```

### REST API

```bash
# Mistral-7B (default - FREE)
curl -X POST "http://localhost:8004/api/property-extraction/extract" \
  -F "file=@appraisal.pdf" \
  -F "provider=olm_ocr" \
  -F "llm_provider=mistral-7b"

# Phi-3-Mini (FREE)
curl -X POST "http://localhost:8004/api/property-extraction/extract" \
  -F "file=@appraisal.pdf" \
  -F "provider=olm_ocr" \
  -F "llm_provider=phi-3-mini"

# Local Ollama (FREE)
curl -X POST "http://localhost:8004/api/property-extraction/extract" \
  -F "file=@appraisal.pdf" \
  -F "provider=olm_ocr" \
  -F "llm_provider=ollama"
```

### Command Line

```bash
# Use free Mistral-7B
python _examples/property_extraction_workflow.py appraisal.pdf \
  --provider olm_ocr \
  --llm-provider mistral-7b

# Use local Ollama
python _examples/property_extraction_workflow.py appraisal.pdf \
  --provider olm_ocr \
  --llm-provider ollama
```

---

## Getting Available LLM Providers

### API Endpoint
```bash
curl http://localhost:8004/api/property-extraction/llm-providers | jq
```

### Response:
```json
{
  "recommended": "mistral-7b",
  "recommendation_reason": "Free via HuggingFace, high quality for Italian documents",
  "models": {
    "mistral-7b-instruct": {
      "provider": "mistral-7b",
      "cost": "Free (HF Inference API)",
      "quality": "⭐⭐⭐⭐",
      "requires_api_key": false
    },
    "phi-3-mini": {
      "provider": "phi-3-mini",
      "cost": "Free (HF Inference API or local)",
      "quality": "⭐⭐⭐",
      "requires_api_key": false
    },
    ...
  }
}
```

---

## Environment Variables

### HuggingFace (Optional)

```bash
# Optional: For higher rate limits
HUGGINGFACE_API_KEY=your_hf_token_here
```

Get a free token at: https://huggingface.co/settings/tokens

### Ollama (Local)

```bash
# Optional: Custom Ollama URL
OLLAMA_BASE_URL=http://localhost:11434
```

---

## Performance Tips

### 1. Use HuggingFace for Cloud Deployments
- No setup required
- Works on any platform
- Free tier available
- API key optional (higher limits with key)

### 2. Use Ollama for Local/Private Deployments
- Complete privacy
- No internet required
- Unlimited requests
- Best with GPU

### 3. Use Anthropic/OpenAI for Best Quality
- If budget allows
- Highest accuracy
- Fastest processing
- Best for production

---

## Complete Workflow Example (100% FREE!)

```python
from ocular.services.property_extraction_service import PropertyExtractionService

# Initialize with free LLM
service = PropertyExtractionService(
    llm_provider="mistral-7b"  # FREE via HuggingFace!
)

# Complete workflow (PDF -> OCR -> Structure -> JSON)
result = await service.process_property_appraisal(
    pdf_path="appraisal.pdf",
    provider_name="olm_ocr",  # Or any OCR provider
    save_output=True
)

# Access structured data
property_profile = AuctionPropertyProfile(**result["structured_data"])
print(f"Sale ID: {property_profile.sale_id}")
print(f"Base Price: €{property_profile.auction_details.base_price:,.2f}")
```

**Total Cost:** 100% FREE! 🎉

---

## Troubleshooting

### HuggingFace Rate Limits

If you hit rate limits on the free HF Inference API:

1. Get a free HF token: https://huggingface.co/settings/tokens
2. Set environment variable:
```bash
export HUGGINGFACE_API_KEY=your_token_here
```

### Ollama Connection Error

If Ollama can't connect:

1. Check Ollama is running:
```bash
ollama serve
```

2. Test connection:
```bash
curl http://localhost:11434/api/tags
```

3. Try a simple request:
```bash
ollama run mistral:7b-instruct "Hello"
```

### Quality Issues

If extraction quality is poor:

1. Try different models:
   - Mistral-7B: Best balance
   - Llama-3-8B: Better reasoning
   - Anthropic Claude: Best quality (paid)

2. Improve OCR quality first:
   - Use OLM OCR or RoLM OCR
   - Ensure PDF is high quality

3. Add additional context:
```python
portal_data = {"idVendita": 12345, ...}
result = await service.process_property_appraisal(
    pdf_path="appraisal.pdf",
    additional_context=portal_data
)
```

---

## Summary

✅ **FREE Options Available:**
- Mistral-7B via HuggingFace (recommended)
- Phi-3-Mini via HuggingFace
- Llama-3-8B via HuggingFace
- Any model via Ollama (local)

✅ **No API Keys Required** (optional for higher limits)

✅ **High Quality** - Suitable for production use

✅ **Easy Integration** - Drop-in replacement

🚀 **Start using free LLMs today and enjoy unlimited property extraction!**
