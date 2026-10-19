import torch, argparse, os
from tqdm import tqdm
from typing import Tuple
from openai import OpenAI
from omegaconf import OmegaConf
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from Tools.Judgers import GPTScorer
from Templates.Judger_Templates import JUDGER_TEMPLATES_PROMPT
from Tools.Base import JudgementType, PromptType, read_prompts_csv, save_prompt_judgements_csv

# Read config
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./config/cfg_Scoring_Prompt_by_Commercial_LLM.yaml")
args = parser.parse_args()

OmegaConf.register_new_resolver("tuple", lambda *x: tuple(x))
config_API = OmegaConf.load("./config/cfg_API.yaml")
config = OmegaConf.load(args.config)
config = OmegaConf.merge(config_API, config)


# Setings
base_url = config.base_url # Set your openai base_url
api_key = config.api_key # Set your api_key
input_file = config.input_file # File to collect the responses from the victim model
output_dir = config.output_dir # File to save the jailbreak scores of the responses
judger_name = config.judger_name # Judge model
prompt_template = JUDGER_TEMPLATES_PROMPT[config.prompt_template]
min_max_score = config.min_max_score # Min and max value the judger would clip
error_value = config.error_value # Score would be setted if judger provide non-integer value
success_threshold = config.success_threshold # Threshold of the score to be classified as "harmful"


def judge_one_file(base_url: str, api_key: str, input_file: str, output_dir: str, judger_name: str, prompt_template: str, min_max_score: Tuple, error_value: int, success_threshold: int):
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize scorer
    judger = GPTScorer(base_url = base_url, api_key = api_key, prompt_template=prompt_template, model=judger_name, min_max_score=min_max_score)
    print(f"Using judge model: {judger.model}")

    # Read prompts
    text_prompts = read_prompts_csv(input_file)

    # Set judgements file path
    input_file_stem = os.path.splitext(os.path.basename(input_file))[0]
    judges_file = os.path.join(output_dir, f"prompt_judges_{input_file_stem}_({judger_name}).csv")

    # Compute jailbreak scores
    judgements = []
    success_count = 0
    skip_count = 0

    print("Computing jailbreak judgements...")
    for item in tqdm(text_prompts):
        judgement = judger.compute_jailbreak_score(question_text = item[4], error_value = error_value, prompt_type = PromptType.Question) # Scoring with Question

        # Check the score
        if judgement[0] >= success_threshold:
            success_count += 1
        elif judgement[0] == error_value:
            # Transform error_value to normal value
            if judger_name == "abab6.5s-chat-pro":
                judgement = (min_max_score[0], judgement[1])
            elif judger_name == "grok-3":
                judgement = (min_max_score[1], judgement[1])
                success_count += 1

            print(f"{item[1]}_{item[2]}: The response of is not a valid integer, due to the judger is {judger_name}, set score to be {judgement[0]}. \n {judgement[1]}")
            skip_count += 1

        judgements.append(item + [judgement[0]])

    # Compute and log final metrics
    success_rate = success_count / len(judgements) if judgements else 0

    # Save scores and compute statistics
    save_prompt_judgements_csv(judgements, judges_file, judger_name, JudgementType.Origin, f"{success_rate:.2%}, {success_count}/{len(judgements)}")

    print(f"Evaluation complete:")
    print(f"Skip count: {skip_count}")
    print(f"Success rate: {success_rate:.2%} ({success_count}/{len(judgements)})")
    print(f"Results saved to: {judges_file}")


if __name__ == "__main__":
    judge_one_file(
        base_url = base_url,
        api_key = api_key,
        input_file = input_file,
        output_dir = output_dir,
        judger_name = judger_name,
        prompt_template = prompt_template,
        min_max_score = min_max_score,
        error_value = error_value,
        success_threshold = success_threshold
    )