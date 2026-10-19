import pandas as pd
import os, re, shutil, httpx, threading

from glob import glob
from typing import Tuple
from openai import OpenAI
from tqdm.auto import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from omegaconf import DictConfig
from concurrent.futures import ThreadPoolExecutor, as_completed

from Tools.Base import ensure_dir, escape_special_characters, read_prompts_csv, save_responses_csv

general_lock = threading.Lock()   # Screen Lock

####################################################### Chunk Files #######################################################

def split_csv(input_file, chunk_size, tmp_dir):
    chunked_files, last_chunk_rows = [], 0

    os.makedirs(tmp_dir, exist_ok=True)
    chunk_id = 0
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        chunk_file = os.path.join(tmp_dir, f'chunk_{chunk_id}.csv')
        chunk.to_csv(chunk_file, index=False)
        row_count = chunk.shape[0]
        chunked_files.append((chunk_file, row_count))
        chunk_id += 1

    print(f"✅ Split completed, {chunk_id} subfiles generated in total.")
    return chunked_files


def merge_chunked_csvs(tmp_dir, output_file):
    ensure_dir(os.path.dirname(output_file))

    files = glob(os.path.join(tmp_dir, 'chunk_*.csv'))
    files = sorted(files, key=lambda x: int(re.search(r'chunk_(\d+)', x).group(1))) # Sort by number in file name
    dfs = [pd.read_csv(f) for f in files]
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(output_file, index=False)

    # Delete the processing directory and all its contents
    shutil.rmtree(tmp_dir)
    print(f"✅ Merge completed, save result as: {output_file}")
    print(f"✅ Cleaned temporary directory: {tmp_dir}")


# Example
# chunked_files = split_csv(
#     input_file = "./data/JailbreakStudy-M2-Modified.csv",
#     chunk_size = 3,
#     tmp_dir = "./tmp"
# )
# print(chunked_files)
# merge_chunked_csvs(
#     tmp_dir = "./tmp",
#     output_file = "./merged.csv"
# )

####################################################### Get Responses #######################################################

def setup_proxy_model(base_url: str, api_key: str):
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=httpx.Timeout(100.0, connect=5.0, read=60.0)
    )


class ResponseGeneratorByProxy:
    def __init__(self, model_name: str, base_url: str, api_key: str):
        self.model_name = model_name

        # Setup API Client
        self.client = setup_proxy_model(base_url, api_key)


    def generate_description(self, prompt: str, developer_message: str) -> str:
        return self._generate_proxy_model(prompt, developer_message, self.model_name)


    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(2))
    def _generate_proxy_model(self, prompt: str, developer_message: str, model_name: str) -> str:
        if developer_message is not None:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "developer",
                        "content": developer_message
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
                # max_tokens=512,
            )
        else:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
                # max_tokens=512,
            )

        # Check whether the victim model is valid
        if response.model is not None:
            return response.choices[0].message.content
        else:
            raise ValueError(f"Unsupported victim model: {model_name}")


def get_responses(print_error: bool = True, config: DictConfig = None):
    responses = []

    # Initialize description generator
    generator = ResponseGeneratorByProxy(model_name=config.tmp.model_name, base_url=config.base_url, api_key=config.api_key)

    # Read prompts
    text_prompts = read_prompts_csv(config.tmp.input_file)

    # Generate descriptions
    for (dataset, category_id, task_id, category_name, question, instruction) in tqdm(text_prompts):
        prompt = config.tmp.user_template.format(question = question)
        try:
            response = generator.generate_description(prompt, config.developer_message)
            if response is None:
                response = ""
            responses.append(
                (category_id, task_id, prompt, question, response)
            )

        except Exception as e:
            responses.append(
                (category_id, task_id, prompt, question, "")
            )
            if print_error:
                print(f"Error processing question {category_id}_{task_id}: {e}")

    # Save responses
    save_responses_csv(responses, config.tmp.output_file, config.tmp.with_HCoT, config.tmp.HCoT_dataset)


####################################################### Multithread #######################################################

def get_responses_thread(input_file: str, output_file: str, bar: tqdm, print_error: bool = False, config: DictConfig = None):
    responses = []

    # Initialize description generator
    generator = ResponseGeneratorByProxy(model_name=config.tmp.model_name, base_url=config.base_url, api_key=config.api_key)

    # Read prompts
    text_prompts = read_prompts_csv(input_file)

    # Generate descriptions
    for (dataset, category_id, task_id, category_name, question, instruction) in text_prompts:
        prompt = config.tmp.user_template.format(question = question)
        try:
            response = generator.generate_description(prompt, config.developer_message)
            if response is None:
                response = ""
            responses.append(
                (category_id, task_id, prompt, question, response)
            )
        except Exception as e:
            responses.append(
                (category_id, task_id, prompt, question, "")
            )
            if print_error:
                print(f"Error processing question {category_id}_{task_id}: {e}")

        with general_lock:
            bar.update(1)

    # Save responses
    save_responses_csv(responses, output_file, config.tmp.with_HCoT, config.tmp.HCoT_dataset)


def get_responses_worker(job_info, bars: list[str]):
    idx = job_info["idx"]
    bar = bars[idx]

    try:
        get_responses_thread(
            input_file = job_info["input_file"],
            output_file = job_info["output_file"],
            config = job_info["config"],
            bar = bar
        )
    finally:
        bar.close()

    return job_info["label"]


def get_jobs(chunked_files: list[Tuple], config: DictConfig = None):
    jobs, idx = [], 0

    for (chunked_file, row_count) in chunked_files:
        jobs.append(
            {
                "idx": idx,
                "label": os.path.splitext(os.path.basename(chunked_file))[0],
                "input_file": chunked_file,
                "output_file": chunked_file,
                "row_count": row_count,
                "config": config
            }
        )
        idx += 1

    return jobs

####################################################### Filal Function #######################################################

def generate_responses_one_file(config: DictConfig = None):
    
    # Get directory and file form output paths and name
    print(f"Using victim model: {config.tmp.model_name}")
    ensure_dir(config.tmp.output_dir)
    config.tmp.output_file = os.path.join(config.tmp.output_dir, f"responses_{config.tmp.model_name}.csv" if  config.tmp.template_name is None else f"{ config.tmp.template_name}/responses_{config.tmp.model_name}.csv")
    
    # Process input questions
    print("Processing questions...")
    try:
        # If it is necessary to split the dataset
        if config.chunk_size != -1:
            # Split datasets
            chunked_files = split_csv(input_file = config.tmp.input_file, chunk_size = config.chunk_size, tmp_dir = config.tmp.tmp_dir)

            # 1. Define tasks
            jobs = get_jobs(chunked_files = chunked_files, config = config)

            # 2. Fixed line number progress bar
            bars = {j["idx"]: tqdm(total=j["row_count"], desc=j["label"], position=j["idx"], leave=True) for j in jobs}

            # 3. Start thread
            with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                futures = [pool.submit(get_responses_worker, j, bars) for j in jobs]
                for f in as_completed(futures):
                    label = f.result()
                    print(f"✅ job-{label} done")

            # Merge chunked responses
            if config.tmp.with_HCoT:
                base, ext = os.path.splitext(config.tmp.output_file)
                config.tmp.output_file = f"{base}_formatted{ext}"
            merge_chunked_csvs(tmp_dir = config.tmp.tmp_dir, output_file = config.tmp.output_file)
        else:
            get_responses(print_error = True, config = config)
            print(f"Responses saved to {config.tmp.output_dir}")

    except (FileNotFoundError, KeyError) as e:
        print(f"Error: {e}")
        return


# responses = []
# if __name__ == "__main__":
#     generate_response_one_file(
#         input_file = input_file,
#         output_dir = output_dir,
#         model_name = model_name,
#         base_url = base_url,
#         api_key = api_key,
#         prompt_template = escape_special_characters(User_TEMPLATES["Car_Answer"], skip_patten = "{question}"),
#         developer_message = DEVELOPER_MESSAGES["D1"],
#         template_name = "",
#         chunk_size = 6, # Split dataset into chunks and processed with multithreading if chunk_size is not -1
#         tmp_dir = "./tmp"
#     )