import os, torch, argparse

from tqdm import tqdm
from typing import Tuple
from omegaconf import OmegaConf

from Tools.Judgers import GPTScorer
from Templates.Judger_Templates import JUDGER_TEMPLATES_RESPONSE
from Tools.Base import JudgementType, PromptType, read_responses_csv, save_judgements_csv


# Read config
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./config/cfg_Scoring_Response_by_Commercial_LLM.yaml")
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
prompt_template = JUDGER_TEMPLATES_RESPONSE[config.prompt_template]
min_max_score = config.min_max_score # Min and max value the judger would clip
error_value = config.error_value # Score would be setted if judger provide non-integer value
prompt_type = PromptType[config.prompt_type] # The type of the content to be checked: Answer, Conversation (Question can not be used here)
success_threshold = config.success_threshold # Threshold of the score to be classified as "harmful"


def judge_one_file(base_url: str, api_key: str, input_file: str, output_dir: str, prompt_template: str, judger_name: str, model_name: str, min_max_score: Tuple, error_value: int, prompt_type: PromptType, success_threshold: int, make_victim_dir: bool = False):
    # Initialize scorer
    scorer = GPTScorer(base_url = base_url, api_key = api_key, prompt_template=prompt_template, model=judger_name, min_max_score=min_max_score)
    print(f"Using judge model: {scorer.model}")

    # Read descriptions
    text_response = read_responses_csv(input_file)

    # Set score file path
    if make_victim_dir:
        score_file = os.path.join(output_dir, model_name, f"judges_{model_name}_({scorer.model}).csv")
    else:
        score_file = os.path.join(output_dir, f"judges_{model_name}_({scorer.model}).csv")

    # Compute jailbreak scores
    scores = []
    success_count = 0
    skip_count = 0

    print("Computing jailbreak scores...")
    for item in tqdm(text_response):
        if prompt_type == PromptType.Conversation:
            score = scorer.compute_jailbreak_score(question_text = item[3], answer_text = item[4], error_value = error_value, prompt_type = prompt_type) # Scoring under conversation type
        elif prompt_type == PromptType.Answer:
            score = scorer.compute_jailbreak_score(answer_text = item[4], error_value = error_value, prompt_type = prompt_type) # Scoring with only responses
        else:
            raise ValueError(f"Unsupported prompt type ({prompt_type}) in judge responses.")

        # Check the score
        if score[0] >= success_threshold:
            success_count += 1
        elif score[0] == error_value:
            # Transform error_value to normal value
            if judger_name == "abab6.5s-chat-pro":
                score = (min_max_score[0], score[1])
            elif judger_name == "grok-3":
                score = (min_max_score[1], score[1])
                success_count += 1

            print(f"{item[0]}_{item[1]}: The response of is not a valid integer, due to the judger is {judger_name}, set score to be {score[0]}. \n {score[1]}")
            skip_count += 1

        scores.append(item + [score[0]])

    # Compute and log final metrics
    total_count = len(scores)
    success_rate = success_count / total_count if scores else 0
    avg_score = sum(s[-1] for s in scores) / total_count if scores else 0

    # Save scores and compute statistics
    save_judgements_csv(scores, score_file, judger_name, JudgementType.Origin, f"{success_rate:.2%}, {success_count}/{total_count}")

    print(f"Evaluation complete:")
    print(f"Reset count: {skip_count}")
    print(f"Success rate: {success_rate:.2%} ({success_count}/{total_count})")
    print(f"Average jailbreak score: {avg_score:.4f}")
    print(f"Results saved to: {score_file}")

    return success_rate


if __name__ == "__main__":
    judge_one_file(
        base_url = base_url,
        api_key = api_key,
        input_file = input_file,
        output_dir = output_dir,
        prompt_template = prompt_template,
        judger_name = judger_name,
        model_name = os.path.split(input_file)[1][10:-4], # Model generating the descriptions
        min_max_score = min_max_score,
        error_value = error_value,
        prompt_type = prompt_type,
        success_threshold = success_threshold,
        make_victim_dir = False
    )