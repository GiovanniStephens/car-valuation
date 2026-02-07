"""
LLM-based extraction of car parameters from Facebook Marketplace listings.

Uses Groq API with Llama 3.3 70B for structured extraction.
"""

import json
import re
from typing import Literal, Optional

import keyring
from pydantic import BaseModel, field_validator


class ExtractedCarListing(BaseModel):
    """Structured representation of extracted car listing data."""

    make: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    odometer: Optional[int] = None
    engine_size_cc: Optional[int] = None
    fuel_type: Literal["Petrol", "Diesel", "Electric", "Hybrid"] = "Petrol"
    transmission: Literal["Automatic", "Manual"] = "Automatic"
    cylinders: int = 4
    is_4wd: bool = False
    exterior_colour: str = "Unknown"
    region: Optional[str] = None
    asking_price: Optional[int] = None
    confidence: Literal["high", "medium", "low"] = "medium"
    extraction_notes: Optional[str] = None

    @field_validator("make", "model", mode="before")
    @classmethod
    def strip_strings(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("fuel_type", mode="before")
    @classmethod
    def normalize_fuel_type(cls, v):
        if not isinstance(v, str):
            return "Petrol"
        v = v.strip().lower()
        mapping = {
            "petrol": "Petrol",
            "gas": "Petrol",
            "gasoline": "Petrol",
            "diesel": "Diesel",
            "electric": "Electric",
            "ev": "Electric",
            "hybrid": "Hybrid",
            "phev": "Hybrid",
        }
        return mapping.get(v, "Petrol")

    @field_validator("transmission", mode="before")
    @classmethod
    def normalize_transmission(cls, v):
        if not isinstance(v, str):
            return "Automatic"
        v = v.strip().lower()
        if "manual" in v or v == "mt":
            return "Manual"
        return "Automatic"


EXTRACTION_PROMPT = """Extract car listing details from Facebook Marketplace posts.

Guidelines:
- Convert engine sizes to cc (e.g., 2.4L = 2400, 1.5L = 1500)
- Convert miles to km if needed (miles * 1.609)
- NZ regions: Auckland, Bay of Plenty, Canterbury, Gisborne, Hawke's Bay, Manawatu, Marlborough, Nelson Bays, Northland, Otago, Southland, Taranaki, Timaru - Oamaru, Waikato, Wairarapa, Wellington, West Coast, Whanganui
- For Japanese imports, make educated guesses about fuel type if not specified
- If year is ambiguous (e.g., "15 model"), interpret as 2015
- Set confidence to "low" if critical fields (make, model, year, odometer) are missing or uncertain"""


def get_groq_client():
    """Initialize Groq client with keyring credentials.

    Returns:
        Groq client instance.

    Raises:
        ValueError: If API key is not found in keyring.
    """
    from groq import Groq

    api_key = keyring.get_password("Groq", "key")
    if not api_key:
        raise ValueError(
            "Groq API key not found. Set it with: "
            "keyring.set_password('Groq', 'key', 'your-api-key')"
        )
    return Groq(api_key=api_key)


def _fix_common_issues(data: dict) -> dict:
    """Fix common extraction issues and normalize data types.

    Args:
        data: Raw extracted dictionary from LLM.

    Returns:
        Cleaned dictionary with proper types.
    """
    result = data.copy()

    # Convert string numbers to integers
    for field in ["year", "odometer", "engine_size_cc", "cylinders", "asking_price"]:
        if field in result and result[field] is not None:
            if isinstance(result[field], str):
                # Remove commas, $, spaces
                cleaned = re.sub(r"[$,\s]", "", result[field])
                try:
                    result[field] = int(float(cleaned))
                except (ValueError, TypeError):
                    result[field] = None

    # Handle engine size conversion from litres string
    if "engine_size_cc" in result and result["engine_size_cc"] is None:
        # Check if there's a string like "2.4L" somewhere
        raw = data.get("engine_size_cc")
        if isinstance(raw, str):
            match = re.search(r"(\d+\.?\d*)\s*[lL]", raw)
            if match:
                result["engine_size_cc"] = int(float(match.group(1)) * 1000)

    # Normalize boolean
    if "is_4wd" in result:
        if isinstance(result["is_4wd"], str):
            result["is_4wd"] = result["is_4wd"].lower() in ["true", "yes", "1"]

    # Ensure make/model are strings
    for field in ["make", "model"]:
        if field in result and result[field] is not None:
            result[field] = str(result[field]).strip()

    return result


def extract_car_params(listing_text: str) -> ExtractedCarListing:
    """Extract car parameters from Facebook Marketplace listing text.

    Uses Groq tool calling for structured extraction - the schema is defined
    as function parameters, ensuring arguments match the expected types.

    Args:
        listing_text: Raw text from a Facebook Marketplace listing.

    Returns:
        ExtractedCarListing with parsed car data.

    Raises:
        ValueError: If extraction fails or tool call is not returned.
    """
    client = get_groq_client()

    # Generate JSON schema from Pydantic model
    schema = ExtractedCarListing.model_json_schema()

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": listing_text},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "extract_car_listing",
                    "description": "Extract structured car details from a listing",
                    "parameters": schema,
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "extract_car_listing"}},
        temperature=0.1,
    )

    # Extract from tool call arguments
    message = response.choices[0].message
    if not message.tool_calls:
        raise ValueError("LLM did not return a tool call")

    tool_call = message.tool_calls[0]
    try:
        data = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse tool call arguments as JSON: {e}")

    # Fix common issues as safety net
    data = _fix_common_issues(data)

    return ExtractedCarListing(**data)
