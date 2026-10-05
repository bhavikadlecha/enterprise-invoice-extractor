"""
extractor.py
------------
Core extraction logic with:
  - Instructor-patched async VLM client (OpenAI, Anthropic, or Gemini)
  - Self-healing retry loop with error injection
  - Token usage + latency tracking (feeds into observability/tracker.py)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import instructor
from pydantic import ValidationError

from app.config import settings
from app.observability.tracker import ExtractionMetrics
from app.schemas import InvoiceSchema

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a precise, literal invoice data extraction engine.
Your job is to read the invoice image and extract all requested fields with
100% accuracy. Follow these rules STRICTLY:

1. VERBATIM EXTRACTION: Extract text fields (VendorName, LineItem descriptions, TaxID) EXACTLY as printed. 
   - DO NOT auto-correct typos or misspellings. 
   - DO NOT drop metadata, packaging specs, or symbols (e.g., preserve "(1X60)").
   - DO NOT hallucinate letters in numbers (e.g., do not mistake an '8' for a 'B').
2. NUMERIC FORMATTING: All monetary values must be in their pure numeric form (e.g., 1234.56, NOT "$1,234.56").
3. DATES: Dates must be in ISO 8601 format: YYYY-MM-DD.
4. ACCOUNTING MATH: The mathematical relationship MUST hold: 
   Subtotal + Tax + ShippingFee - Discount == TotalAmount.
   Double-check your arithmetic before responding. Ignore deceptive receipt labels like "Sub Total: 0.50" if the math implies otherwise.
5. If a field is not visible on the invoice, use null for optional fields (or 0.00 for monetary fields).
"""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ExtractionResult:
    """Returned by extract_invoice(). Bundles the invoice with its metrics."""
    invoice: InvoiceSchema
    metrics: ExtractionMetrics


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------
class ExtractionError(Exception):
    def __init__(self, message: str, last_validation_error: Optional[ValidationError] = None):
        super().__init__(message)
        self.last_validation_error = last_validation_error


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------
def _build_instructor_client():
    if settings.llm_provider == "openai":
        import openai
        raw_client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        client = instructor.from_openai(raw_client)
        return client, settings.openai_model, "openai"

    elif settings.llm_provider == "anthropic":
        import anthropic as anthropic_sdk
        raw_client = anthropic_sdk.AsyncAnthropic(api_key=settings.anthropic_api_key)
        client = instructor.from_anthropic(raw_client)
        return client, settings.anthropic_model, "anthropic"

    elif settings.llm_provider == "gemini":
        # Use litellm as a robust bridge to Gemini's native API
        import os
        from litellm import acompletion
        os.environ["GEMINI_API_KEY"] = settings.gemini_api_key
        client = instructor.from_litellm(acompletion)
        return client, "gemini/" + settings.gemini_model, "gemini"

    raise ValueError(f"Unsupported llm_provider: {settings.llm_provider!r}")


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------
async def extract_invoice(base64_image: str) -> ExtractionResult:
    """
    Extract structured invoice data from a base64-encoded JPEG image.
    Includes Auto-Fallback for Gemini to handle Google's frequent model availability changes.
    """
    client, base_model, provider = _build_instructor_client()

    # If Gemini, create a fallback cascade so the user never has to guess models again
    if provider == "gemini":
        models_to_try = [
            base_model,
            "gemini/gemini-3.8-flash",
            "gemini/gemini-flash-latest",
            "gemini/gemini-2.5-flash",
            "gemini/gemini-3.5-flash",
        ]
        # Remove duplicates while preserving order
        models_to_try = list(dict.fromkeys(models_to_try))
    else:
        models_to_try = [base_model]

    last_error: Optional[Exception] = None
    error_feedback: str = ""
    total_input_tokens = 0
    total_output_tokens = 0
    start_time = time.perf_counter()

    for current_model in models_to_try:
        retries_used = 0
        error_feedback = ""
        
        for attempt in range(1, settings.max_retries + 1):
            logger.info("Trying model [%s] - attempt %d/%d", current_model, attempt, settings.max_retries)
            messages = _build_messages(base64_image, error_feedback)

            try:
                if provider == "gemini":
                    # litellm: use `create`
                    response = await client.chat.completions.create(
                        model=current_model,
                        response_model=InvoiceSchema,
                        max_retries=1,
                        temperature=0.0,
                        messages=messages,
                    )
                    raw_response = None
                else:
                    # OpenAI / Anthropic
                    response, raw_response = await client.chat.completions.create_with_completion(
                        model=current_model,
                        response_model=InvoiceSchema,
                        max_retries=1,
                        temperature=0.0,
                        messages=messages,
                    )

                # --- Token usage (OpenAI/Anthropic only) ---
                if raw_response and hasattr(raw_response, "usage") and raw_response.usage:
                    usage = raw_response.usage
                    total_input_tokens += getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0)
                    total_output_tokens += getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0)

                latency = round(time.perf_counter() - start_time, 3)

                metrics = ExtractionMetrics(
                    model=current_model,
                    provider=provider,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    latency_seconds=latency,
                    retries=retries_used,
                    success=True,
                    vendor_name=response.vendor_name,
                )

                logger.info("Extraction succeeded using %s (%.2fs).", current_model, latency)
                return ExtractionResult(invoice=response, metrics=metrics)

            except ValidationError as exc:
                last_error = exc
                retries_used += 1
                error_feedback = _format_validation_errors(exc)
                logger.warning("Attempt %d failed validation:\n%s", attempt, error_feedback)
            except Exception as exc:
                # Catch 404s, 429s, 503s from Google and immediately break to the next model in fallback list
                err_str = str(exc)
                if "404" in err_str or "429" in err_str or "503" in err_str or "High demand" in err_str:
                    logger.warning("Model %s rejected by Google (Quota/Unavailable). Auto-falling back...", current_model)
                    last_error = exc
                    break # Break the retry loop and try the next model
                
                # If it's a completely unknown error, still try the next model just in case
                logger.warning("Unexpected error with %s: %s", current_model, err_str)
                last_error = exc
                break

    # All models and retries exhausted
    raise ExtractionError(
        f"Extraction completely failed after trying all fallback models. Last error: {last_error}",
        last_validation_error=last_error if isinstance(last_error, ValidationError) else None,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_messages(base64_image: str, error_feedback: str) -> list[dict]:
    """Build the messages list for the LLM call."""
    text_block = (
        "Please extract all invoice data from the image below and return it "
        "as a structured JSON object matching the required schema."
    )
    if error_feedback:
        text_block += (
            "\n\nCORRECTION REQUIRED — your previous response had the following errors:\n"
            f"{error_feedback}\n"
            "Please carefully re-read the invoice image and fix these specific issues."
        )

    return [
        {
            "role": "system",
            "content": _SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_block},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        },
    ]


def _format_validation_errors(exc: ValidationError) -> str:
    lines = []
    for error in exc.errors():
        field_path = " → ".join(str(loc) for loc in error["loc"]) if error["loc"] else "model"
        lines.append(f"  • Field [{field_path}]: {error['msg']} (type: {error['type']})")
    return "\n".join(lines)
