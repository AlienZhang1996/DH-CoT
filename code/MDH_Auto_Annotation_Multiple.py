import torch
import argparse

from pathlib import Path
from tqdm.auto import tqdm
from omegaconf import OmegaConf

from Tools.Judgers import PromptType
from Tools.MDH_Tools import compute_votes_one_file
from Tools.Base import ConversationType, JudgementType
from Templates.Judger_Templates import JUDGER_TEMPLATES_RESPONSE as JUDGER_TEMPLATES


if __name__ == "__main__":
    # Read config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/cfg_MDH_JRD_Auto-Annotation_Multiple.yaml")
    args = parser.parse_args()
    
    OmegaConf.register_new_resolver("tuple", lambda *x: tuple(x))
    config_API = OmegaConf.load("./config/cfg_API.yaml")
    config_API = OmegaConf.create({"API_Judgers": config_API})
    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config, config_API)

    config.API_Judgers.judger_templates_api = [JUDGER_TEMPLATES[k] for k in config.API_Judgers.judger_templates_api]
    config.API_Judgers.prompt_type_api = PromptType[config.API_Judgers.prompt_type_api]
    config.LLama_Guards.judgement_type_guards = JudgementType[config.LLama_Guards.judgement_type_guards]
    config.LLama_Guards.conversation_type_guards = ConversationType[config.LLama_Guards.conversation_type_guards]

    input_dir = Path(config.Base.input_dir)
    output_dir = Path(config.Base.output_dir)

    # Make sure the output root directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Recursive traversal
    for csv_file in input_dir.rglob('*.csv'):
        if 'checkpoint' in csv_file.name:
            print(f"Skip the python checkpoint file: {csv_file}")
            continue

        # Relative paths, used to maintain subdirectory structure
        rel_path = csv_file.relative_to(input_dir)

        # Output path
        output_path = (output_dir / rel_path).parent
        output_path.mkdir(parents=True, exist_ok=True)
        config.Base.output_dir = output_path

        # Voting
        print("-----------------")
        print(f"Processing file: {csv_file}")
        print("-----------------")
        # Voting
        compute_votes_one_file(
            input_file = csv_file,
            config = config
        )
        print("Processing done!", end="\n\n")