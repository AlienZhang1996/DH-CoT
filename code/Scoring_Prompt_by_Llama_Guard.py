import os, torch, argparse

from tqdm import tqdm
from omegaconf import OmegaConf

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoProcessor, MllamaForConditionalGeneration, MllamaProcessor

from Tools.Judgers import LlamaGuardJudger
from Tools.Base import ConversationType, JudgementType, read_prompts_csv, save_prompt_judgements_csv


# Read config
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./config/cfg_Scoring_Prompt_by_Llama_Guard.yaml")
args = parser.parse_args()
config = OmegaConf.load(args.config)

# Setings
input_file = config.input_file # File to collect the responses from the victim model
output_dir = config.output_dir # File to save the jailbreak judgements of the responses
judger_name = config.judger_name # Can be Llama-Guard-3-1B, Llama-Guard-3-8B, Llama-Guard-3-8B-INT8, Llama-Guard-3-11B-Vision, Llama-Guard-4-12B
judger_dir_guard = config.judger_dir_guard # Judge Model directory
conversation_type = ConversationType[config.conversation_type] # Can be User, Assistant, User_Image. Text Llama can not use *_Image.
judgement_type = JudgementType[config.judgement_type] # Can be Origin, Num_1, Num_10, Bool


def judge_one_file(input_file: str, output_dir: str, judger_name: str, judger_dir_guard: str, conversation_type: ConversationType, judgement_type: JudgementType):
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize scorer
    judger = LlamaGuardJudger(judger_name = judger_name, judger_dir_guard = judger_dir_guard, device = device)
    print(f"Using judge model: {judger.judger_name}")

    # Read descriptions, Set judgements file path
    text_prompts = read_prompts_csv(input_file)
    input_file_stem = os.path.splitext(os.path.basename(input_file))[0]
    judges_file = os.path.join(output_dir, f"prompt_judges_{input_file_stem}_({judger_name}).csv")

    # Compute jailbreak scores
    judgements = []
    success_count = 0

    print("Computing jailbreak judgements...")
    for item in tqdm(text_prompts):
        if conversation_type in judger.conversation_imgset:
            judgement = judger.judge(
                user_content = item[4],
                assistant_content = "",
                image_path = None,
                conversation_type = conversation_type,
                print_input_prompt = False
            )
        else:
            judgement = judger.judge(
                user_content = item[4],
                assistant_content = "",
                conversation_type = conversation_type,
                print_input_prompt = False
            )

        judgements.append(item + [judgement])

        if judgement.startswith("unsafe"):
            success_count += 1


    # Compute and log final metrics
    success_rate = success_count / len(judgements) if judgements else 0

    # Save scores and compute statistics
    save_prompt_judgements_csv(judgements, judges_file, judger_name, judgement_type, f"{success_rate:.2%}, {success_count}/{len(judgements)}")


    print(f"Evaluation complete:")
    print(f"Success rate: {success_rate:.2%} ({success_count}/{len(judgements)})")
    print(f"Results saved to: {judges_file}")


if __name__ == "__main__":
    judge_one_file(
        input_file = input_file,
        output_dir = output_dir,
        judger_name = judger_name,
        judger_dir_guard = judger_dir_guard,
        conversation_type = conversation_type,
        judgement_type = judgement_type
    )