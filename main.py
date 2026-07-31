"""
Render FastAPI backend for the Gemma Nigeria Diagnosis app.

Single endpoint (POST /diagnose) that serves both the Kotlin app and the
Telegram bot -- they send the same request shape, get the same response shape.

Flow per request:
  1. Receive voice/image/text input from client (app or Telegram)
  2. Call Gemma (multimodal) to transcribe/understand the input and extract
     a clean symptom description
  3. Embed that symptom description, query Chroma Cloud for the closest
     matching disease record
  4. Call Gemma again with the retrieved record as grounding context, asking
     it to produce the final structured guidance in the user's language
  5. Return a structured JSON response matching the shared schema

Environment variables required (set these in Render's dashboard):
  GEMINI_API_KEY   - Google AI Studio API key
  HF_TOKEN         - Hugging Face token (only needed if the dataset repo is
                     private; harmless to set even if public)
  CHROMA_API_KEY   - Chroma Cloud API key
  CHROMA_TENANT    - Chroma Cloud tenant id (optional if key is single-DB scoped)
  CHROMA_DATABASE  - Chroma Cloud database name (optional if key is single-DB scoped)
"""

import base64
import json
import os
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from google import genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors

# ---- Config ----
# Fallback chain: tried in order for every Gemma/Gemini call. Gemma models
# are tried FIRST since this is a "Build with Gemma" hackathon submission --
# Gemini models are only a backup if Gemma is unavailable/rate-limited.
MODEL_CHAIN = [
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it",
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
]
EMBEDDING_MODEL = "gemini-embedding-001"   # must match embed_and_upsert.py
CHROMA_COLLECTION = "disease_kb"           # must match embed_and_upsert.py
TOP_K = 1  # only need the single best match for this use case

app = FastAPI(title="Gemma Nigeria Diagnosis API")

# ---- Clients (created once at startup, reused across requests) ----
_gemini_client: Optional[genai.Client] = None
_chroma_collection = None


def get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is None:
        api_key = os.environ.get("CHROMA_API_KEY")
        if not api_key:
            raise RuntimeError("CHROMA_API_KEY environment variable not set")
        tenant = os.environ.get("CHROMA_TENANT")
        database = os.environ.get("CHROMA_DATABASE")
        if tenant and database:
            client = chromadb.CloudClient(tenant=tenant, database=database, api_key=api_key)
        else:
            client = chromadb.CloudClient(api_key=api_key)
        _chroma_collection = client.get_or_create_collection(name=CHROMA_COLLECTION)
    return _chroma_collection


# ---- Request / response schema ----

class DiagnoseRequest(BaseModel):
    domain: Literal["crop", "animal", "human"]
    input_type: Literal["voice", "image", "text"]
    language_hint: Literal["yoruba", "hausa", "igbo", "pidgin", "english", "auto"] = "auto"
    audio_base64: Optional[str] = None
    image_base64: Optional[str] = None
    text: Optional[str] = None
    source: Literal["app", "telegram"] = "app"


class DiagnoseResponse(BaseModel):
    matched_disease: str
    domain: str
    confidence: Literal["high", "medium", "low"]
    symptoms_detected: str
    cause: str
    prevention: str
    guidance: str
    severity: str
    disclaimer: str
    language_response: str
    transcribed_input: str


def generate_with_fallback(client: genai.Client, contents, system_instruction: str):
    """
    Tries each model in MODEL_CHAIN in order until one succeeds. Raises the
    last error if every model fails, so the caller can return a clean 503
    instead of the request crashing with an unhandled 500.
    """
    last_error: Optional[Exception] = None
    for model in MODEL_CHAIN:
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=genai_types.GenerateContentConfig(system_instruction=system_instruction),
            )
        except genai_errors.ClientError as e:
            last_error = e
            continue  # quota/rate/not-found on this model -- try the next one
        except Exception as e:
            last_error = e
            continue
    raise HTTPException(
        status_code=503,
        detail="All AI models are currently unavailable (rate limits or outages). Please try again shortly."
    ) from last_error


# ---- Step 1: transcribe/understand the raw input via Gemma ----

def extract_symptoms(req: DiagnoseRequest) -> tuple[str, str]:
    """
    Returns (transcribed_input, symptom_description).
    Uses Gemma's multimodal input for voice/image, or passes text straight
    through for text input.
    """
    client = get_gemini_client()

    system_instruction = (
        "You are helping a diagnostic assistant understand a farmer, animal "
        "owner, or patient in Nigeria. Transcribe or describe what they said "
        "or what the image shows, then extract a clean, concise symptom "
        "description in English, regardless of the input language. "
        "Return ONLY valid JSON, no markdown fences, in this exact format: "
        '{"transcribed_input": "...", "symptom_description": "..."}'
    )

    parts = []
    if req.input_type == "voice" and req.audio_base64:
        audio_bytes = base64.b64decode(req.audio_base64)
        parts.append(genai_types.Part.from_bytes(data=audio_bytes, mime_type="audio/ogg"))
        parts.append("Transcribe this voice message and extract the symptoms described.")
    elif req.input_type == "image" and req.image_base64:
        image_bytes = base64.b64decode(req.image_base64)
        parts.append(genai_types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
        parts.append(f"Describe what you see in this {req.domain} image and extract visible symptoms.")
    elif req.input_type == "text" and req.text:
        parts.append(f"Extract the symptom description from this message: {req.text}")
    else:
        raise HTTPException(status_code=400, detail="Missing input data for the given input_type")

    response = generate_with_fallback(client, parts, system_instruction)

    text = response.text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]

    try:
        parsed = json.loads(text)
        return parsed.get("transcribed_input", ""), parsed.get("symptom_description", "")
    except json.JSONDecodeError:
        # fall back to using the raw model output as the symptom description
        return text, text


# ---- Step 2: embed the symptom description and query Chroma ----

def retrieve_disease_record(symptom_description: str, domain: str) -> tuple[dict, str]:
    """Returns (metadata_dict, confidence_level)."""
    client = get_gemini_client()
    collection = get_chroma_collection()

    embed_result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=symptom_description,
    )
    query_vector = embed_result.embeddings[0].values

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=TOP_K,
        where={"domain": domain},
    )

    if not results["ids"] or not results["ids"][0]:
        raise HTTPException(status_code=404, detail="No matching disease record found")

    metadata = results["metadatas"][0][0]
    distance = results["distances"][0][0] if results.get("distances") else None

    # Chroma cosine distance: lower = more similar. Thresholds are a starting
    # point -- tune these once you've reviewed real query results.
    if distance is None:
        confidence = "medium"
    elif distance < 0.15:
        confidence = "high"
    elif distance < 0.30:
        confidence = "medium"
    else:
        confidence = "low"

    return metadata, confidence


# ---- Step 3: generate the final structured guidance via Gemma ----

def generate_guidance(metadata: dict, symptom_description: str, language_hint: str) -> dict:
    client = get_gemini_client()

    disclaimer_required = str(metadata.get("disclaimer_required", "")).lower() == "true"
    requires_professional = str(metadata.get("requires_professional", "")).lower() == "true"

    system_instruction = (
        "You are a diagnostic assistant helping users in Nigeria understand "
        "crop, animal, or human health issues. You are given a VERIFIED "
        "disease record. Do not invent facts beyond what is given. Respond "
        f"in {language_hint if language_hint != 'auto' else 'the same language as the input'}, "
        "in a warm, clear, simple tone suitable for someone without medical/technical "
        "background. Return ONLY valid JSON, no markdown fences, in this exact format: "
        '{"guidance_message": "...", "language_response": "..."}'
    )

    prompt = (
        f"User's reported symptoms: {symptom_description}\n\n"
        f"Matched disease record (verified, source: {metadata.get('source')}):\n"
        f"Name: {metadata.get('name')}\n"
        f"Cause: {metadata.get('cause')}\n"
        f"Prevention: {metadata.get('prevention')}\n"
        f"Guidance: {metadata.get('guidance')}\n"
        f"Severity: {metadata.get('severity')}\n\n"
        "Write a short, clear message combining this into actionable advice for the user."
    )

    response = generate_with_fallback(client, prompt, system_instruction)

    text = response.text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {"guidance_message": text, "language_response": language_hint}

    disclaimer = ""
    if disclaimer_required or requires_professional:
        if metadata.get("domain") == "human":
            disclaimer = "This is not a medical diagnosis. Please see a doctor or health worker for proper care."
        elif metadata.get("domain") == "animal":
            disclaimer = "Please consult a veterinary or livestock extension officer to confirm this and get treatment."
        else:
            disclaimer = "Please consult an agricultural extension officer to confirm this before taking action."

    return parsed, disclaimer


# ---- Main endpoint ----

@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose(req: DiagnoseRequest):
    transcribed_input, symptom_description = extract_symptoms(req)

    if not symptom_description:
        raise HTTPException(status_code=422, detail="Could not extract symptoms from input")

    metadata, confidence = retrieve_disease_record(symptom_description, req.domain)
    guidance_parsed, disclaimer = generate_guidance(metadata, symptom_description, req.language_hint)

    return DiagnoseResponse(
        matched_disease=metadata.get("name", "Unknown"),
        domain=metadata.get("domain", req.domain),
        confidence=confidence,
        symptoms_detected=symptom_description,
        cause=metadata.get("cause", ""),
        prevention=metadata.get("prevention", ""),
        guidance=guidance_parsed.get("guidance_message", metadata.get("guidance", "")),
        severity=metadata.get("severity", "unknown"),
        disclaimer=disclaimer,
        language_response=guidance_parsed.get("language_response", req.language_hint),
        transcribed_input=transcribed_input,
    )


@app.get("/health")
def health():
    """Cheap endpoint to keep Render's free tier warm via an external pinger."""
    return {"status": "ok"}


# ---- Chat endpoint (free-form conversation, separate from /diagnose) ----

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]  # full conversation history, oldest first
    domain: Optional[Literal["crop", "animal", "human"]] = None  # optional context hint
    source: Literal["app", "telegram"] = "app"


class ChatResponse(BaseModel):
    reply: str


CHAT_SYSTEM_INSTRUCTION = (
    "You are a friendly, knowledgeable assistant helping people in Nigeria with "
    "general questions about crop health, animal health, and human health. "
    "This is an open conversation, not a formal diagnosis -- if someone describes "
    "specific symptoms and wants an actual diagnosis, gently suggest they use the "
    "app's dedicated Diagnosis feature instead, since that one is grounded in a "
    "verified knowledge base. For general questions, explanations, or casual "
    "conversation, answer helpfully and warmly in plain, simple language. Keep "
    "responses reasonably short and conversational. If a question touches on "
    "human health, remind the user (briefly, not repetitively) to see a real "
    "doctor for anything serious."
)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    client = get_gemini_client()

    # Convert the conversation history into the format generate_content expects:
    # a list of Content parts alternating user/model turns.
    contents = []
    for msg in req.messages:
        role = "model" if msg.role == "assistant" else "user"
        contents.append(genai_types.Content(role=role, parts=[genai_types.Part(text=msg.content)]))

    system_instruction = CHAT_SYSTEM_INSTRUCTION
    if req.domain:
        system_instruction += f" The user has been browsing the '{req.domain}' section of the app."

    response = generate_with_fallback(client, contents, system_instruction)

    return ChatResponse(reply=response.text.strip())