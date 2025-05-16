import os
import time
import json
import re
from pathlib import Path
import numpy as np
from google import genai
from google.genai.types import HttpOptions
from google.api_core.exceptions import GoogleAPICallError, RetryError, InvalidArgument
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Dict, Any, Literal, Annotated
from dotenv import load_dotenv, find_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

from loguru import logger
from main.config import INTERIM_DATA_DIR
from main.utils import *

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

def call_gemini_api(client: genai, model_name: str, prompt: str) -> Optional[SarcasmAnnotation]:
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
    except (InvalidArgument, RetryError, GoogleAPICallError) as e:
        logger.error(f"API error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None

def get_data():
    input_path: Path = INTERIM_DATA_DIR / "Round_1/Label/text.json"
    data = load_json(input_path)
    return [item['caption'] for item in data]

def process_caption(client, model, ith_caption: Tuple[int, str]) -> Optional[Dict[str, Any]]:
    ith, caption = ith_caption
    logger.info(f"Processing caption {ith}")
    prompt = f"Classify the following text as sarcasm or non-sarcasm: {caption}"
    result = call_gemini_api(client, model, prompt)

    time.sleep(0.4)

    if result:
        return {
            'index': ith,
            'caption': result.caption,
            'label': result.label,
            'confident_score': result.confident_score,
            'reason': result.reason
        }
    else:
        return None

def main():
    key = os.getenv('API_KEY')
    model = 'gemini-2.5-pro-preview-05-06'
    
    try:
        client = genai.Client(api_key=key, http_options=HttpOptions(timeout=2*60000))
        logger.info(f"Gemini client initialized. Model: {model}")
    except Exception as e:
        logger.critical(f"Failed to initialize client: {e}. Exiting.")
        return
    
    data = get_data()
    output = []
    failed_rows = []

    max_workers = 10  # Tùy chỉnh theo CPU và quota

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_caption, client, model, (ith, caption)): ith
            for ith, caption in enumerate(data)
        }

        for future in as_completed(futures):
            ith = futures[future]
            try:
                result = future.result(timeout= 30)
                if result:
                    output.append(result)
                    logger.info(f"Complete caption {ith}")
                else:
                    failed_rows.append(ith)
                    logger.warning(f"API call failed for item {ith}")

            except TimeoutError:
                logger.error(f"Timeout while processing item {ith}")
                failed_rows.append(ith)
            except Exception as e:
                logger.error(f"Unhandled error in thread for item {ith}: {e}")
                failed_rows.append(ith)

            logger.info(f"Total processed: {len(output)} | Failed: {len(failed_rows)}")


    #logger.info(f"Total processed: {len(output)} | Failed: {len(failed_rows)}")
    output = [output + failed_rows]
    save_to_json(output, INTERIM_DATA_DIR / "text_labeled.json")
    logger.info('Output saved')

if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
    finally:
        elapsed = time.time() - start_time
        logger.info(f"Total execution time: {elapsed:.2f} seconds")