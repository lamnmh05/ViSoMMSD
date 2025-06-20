import os
import time
import json
from google import genai
from google.genai import types
from google.genai.types import HttpOptions
from google.api_core.exceptions import GoogleAPICallError, RetryError, InvalidArgument
from dotenv import load_dotenv, find_dotenv
from loguru import logger
from main.utils import get_config, remove_json_markdown

def label_sample(client, model, ith_sample, prompt):
    ith, sample = ith_sample
    logger.info(f"Processing sample {ith}:")
    try:
        with open(sample['image'], 'rb') as f:
            img_bytes = f.read()
    except Exception as e:
        logger.error(f"Error loading image {sample.get('image')}: {e}")
        return None

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(mime_type="image/png", data=img_bytes),
                types.Part.from_text(text=f"{prompt.user_prompt} {sample['caption']}"),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(response_mime_type="text/plain")

    try:
        result = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        logger.info(f"Completed sample {ith}")
        
        return result.candidates[0].content.parts[0].text

    except (InvalidArgument, RetryError, GoogleAPICallError) as e:
        logger.error(f"API error for item {ith}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error for item {ith}: {e}")
        return None

def main(config):
    if load_dotenv(find_dotenv()):
        logger.info("Successfully loaded environment variables from .env file.")
    else:
        logger.warning("Could not find .env file. Attempting to use environment variables directly.")

    key = os.getenv('API_KEY')
    model = config.model_name
    prompt = get_config(config.prompt_path)
    data = json.load(open(config.input_file, 'r', encoding='utf-8'))

    for sample in data:
        if not os.path.isabs(sample['image']):
            sample['image'] = os.path.join(config.image_folder, sample['image'])

    client = genai.Client(api_key=key, http_options=HttpOptions(timeout=config.http_timeout * 1000))
    output = []
    failed_rows = []
    total = len(data)

    print(total)
    
    for ith, sample in enumerate(data):
        try:
            result = label_sample(client, model, (ith, sample), prompt)
            
            try:
                parsed_result = remove_json_markdown(result)
                sample['image_llm_label'] = parsed_result['label']
            except:
                logger.warning(f"remove_json_markdown failed for item {ith}: {e}")
                sample['image_llm_label'] = result

            output.append(sample)

        except Exception as e:
            logger.error(f"Unhandled error for item {ith}: {e}")
            failed_rows.append(sample)
        time.sleep(3)  # RPM = 30

    with open(config.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    with open(config.failed_file, "w", encoding="utf-8") as f:
        json.dump(failed_rows, f, ensure_ascii=False, indent=2)
    logger.info('Output saved')

if __name__ == "__main__":
    start_time = time.time()
    config = get_config(r"D:\Git_repo\ViSoMMSD\config\CoT_4-shot_gemma-3-4b-it.yaml")
    try:
        main(config)
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
    finally:
        elapsed = time.time() - start_time
        logger.info(f"Prompt technique {config.name}")
        logger.info(f"Total execution time: {elapsed:.2f} seconds")
