import os, argparse
from pathlib import Path
from omegaconf import OmegaConf

from Templates.User_Templates import USER_TEMPLATES
from Templates.Developer_Messages import DEVELOPER_MESSAGES

from Tools.Base import escape_special_characters, select_files_from_dir
from Tools.Response_Generation import generate_responses_one_file

if __name__ == "__main__":
    
    # Read config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/cfg_Generate_Response.yaml")
    args = parser.parse_args()
    
    config_API = OmegaConf.load("./config/cfg_API.yaml")
    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config_API, config)
    config = OmegaConf.merge(config, {
        "tmp": {
            "input_file": None,
            "output_dir": None,
            "output_file": None,
            "model_name": None,
            "prompt_template": None,
            "user_template": None,
            "tmp_dir": None,
            "with_HCoT": None,
            "HCoT_dataset": None
        }
    })
    config.user_templates = {k: USER_TEMPLATES[k] for k in config.user_templates}
    config.developer_message = DEVELOPER_MESSAGES[config.developer_message]
    
    # Input, output location
    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Scan the dataset and print it
    datasets = select_files_from_dir(input_dir = input_dir, file_pattern = "*.csv", filter_patterns = ["checkpoint"])
    print(f"Scanning datasets from f{input_dir}:", end="\n\n")
    print(*datasets, sep='\n')
    print()

    # Recursive traversal
    for input_file in datasets:
        input_file = Path(input_file)
        
        # Relative paths, used to maintain subdirectory structure
        rel_path = input_file.relative_to(input_dir)
        config.tmp.input_file = input_file
        
        # Check if there is a dataset directory
        stem_dir = input_file.stem
        if rel_path.parent.name != stem_dir:
            rel_path = rel_path.parent / stem_dir / input_file.name
        
        # Output path
        output_path = (output_dir / rel_path).parent
        output_path.mkdir(parents=True, exist_ok=True)
        config.tmp.output_dir = output_path
        config.tmp.tmp_dir = os.path.join(output_path, 'tmp')

        # Did the dataset use H-CoT
        if "Hcot" in input_file.name:
            config.tmp.with_HCoT = True
            config.tmp.HCoT_dataset = input_file
        else:
            config.tmp.with_HCoT = False
            config.tmp.HCoT_dataset = None
        
        # Generating Responses
        print("---------------------------------------------------")
        print(f"Processing file: {input_file}")
        print("---------------------------------------------------")

        for template_name, user_template in config.user_templates.items():
            print(f"Using template: {template_name}")
            print(f"Using developer_message: \n{config.developer_message}")
            
            config.tmp.template_name = template_name
            config.tmp.user_template = escape_special_characters(user_template, skip_patten = "{question}")
            
            for model_name in config.model_names:
                config.tmp.model_name = model_name
                generate_responses_one_file(config = config)
                print("-----------------")
            print("-----------------")
        
        print("Processing done!", end="\n\n")
