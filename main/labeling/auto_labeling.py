import os
import time
import json
import re

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from google import genai
from google.api_core.exceptions import GoogleAPICallError, RetryError, InvalidArgument
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Dict, Any, Literal, Annotated
from dotenv import load_dotenv, find_dotenv  # For loading .env file

from loguru import logger  # Replaces logging
from main.config import INTERIM_DATA_DIR
from main.utils import *


# --- Configuration Loading ---


# Load environment variables from .env file
if load_dotenv(find_dotenv()):
    logger.info("Successfully loaded environment variables from .env file.")
else:
    logger.warning("Could not find .env file. Attempting to use environment variables directly.")



class SarcasmAnnotation(BaseModel):
    caption: str
    label: Literal["sarcasm", "non-sarcasm"]
    confident_score: Annotated[float, Field(ge=0.0, le=1.0)]
    reason: str


def call_gemini_api(client: genai, model_name: str, prompt: str) -> Optional[str]:
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": SarcasmAnnotation,
            },
        )
        return response.parsed

    except InvalidArgument as e:
        logger.error(f"Invalid input: {e.message}")
        return None

    except RetryError as e:
        logger.error(f"Retryable error (possibly rate limit): {e}")
        return None

    except GoogleAPICallError as e:
        logger.error(f"API call failed: {e.message}")
        return None

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None
    

def get_data():
    input_path: Path = INTERIM_DATA_DIR / "Round_1/Label/text.json"
    data = load_json(input_path)
    
    captions = []
    for caption in data:
        captions.append(caption['caption'])
    return captions

def main():

    key = os.getenv('API_KEY')
    model = 'gemini-2.5-pro-preview-05-06'
    try:
        # Pass timeout directly to the client for broader application
        #genai.configure(api_key=key)
        client = genai.Client(api_key=key)
        logger.info(f"OpenAI client initialized. Model: {model}")
    except Exception as e:
        logger.critical(f"Failed to initialize client: {e}. Exiting.")
        return
    
    data = get_data()
    updated_count = 0
    failed_rows = []

    output = []

    try:
        for ith, caption in enumerate(data[:1]):
            logger.info(f"Processing {ith}")
            api_output = call_gemini_api(
                client=client,
                model_name=model,
                prompt=f"Classify the following text as sarcasm or non-sarcasm: {caption}"
            )
            if api_output:
                updated_count += 1
            else:
                logger.warning(f"Skipping update for {ith}th caption due to API call failure or invalid response.")
                failed_rows.append(ith)

            output.append ({
                'caption': api_output.caption,
                'label': api_output.label,
                'confident_score': api_output.confident_score,
                'reason': api_output.reason
            })
    except Exception as e:
        logger.critical(f"An unexpected error occurred during the main processing loop: {e}")

    save_to_json(output, INTERIM_DATA_DIR  / "text_labeled.json")
    logger.info('Output saved')

if __name__ == "__main__":
    main()