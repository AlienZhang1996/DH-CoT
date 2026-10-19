import os, csv, base64
import pandas as pd
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Any, Iterable

############################################# Data type #############################################

class ConversationType(Enum):
    User = 1
    Assistant = 2
    User_Assistant = 3
    User_Image = 4
    Assistant_Image = 5
    User_Assistant_Image = 6


class LlamaGuardType(Enum):
    One_B = 1
    Eight_B = 2
    Eleven_B_Vision = 3
    Ewelve_B_IV = 4


class JudgementType(Enum):
    Origin = 1
    Num_1 = 2
    Num_10 = 3
    Bool = 4


class PromptType(Enum):
    Question = 1
    Answer = 2
    Conversation = 3

############################################# Base Function #############################################

def get_media_type(image_path: str) -> str:
    """Get the correct media type based on file extension."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".jpg", ".jpeg", ".jpeg"]:
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    else:
        raise ValueError(f"Unsupported image extension: {ext}")


def encode_image(image_path: str) -> str:
    """Encode image file to base64 string.

    Args:
        image_path: Path to image file

    Returns:
        str: Base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path to ensure exists
    """
    os.makedirs(path, exist_ok=True)


def escape_special_characters(s: str, skip_patten: str = "{question}"):
    """转义字符串中的特殊字符，同时保留 skip_patten 不变"""

    # Replace skip_patten with
    s = s.replace(skip_patten, "_PLACEHOLDER_")

    # Escaping curly braces
    s = s.replace("{", "{{").replace("}", "}}")
    # Escaping backslashes
    s = s.replace("\\", "\\\\")
    # Escape percent signs (if necessary)
    s = s.replace("%", "%%")

    # Replace skip_patten back
    s = s.replace("_PLACEHOLDER_", skip_patten)
    return s


def find_file_path(folder_path, target_filename):
    """
    Search for file paths in a specified folder and its subfolders
    :param folder_path: Folder paths to search
    :param target_filename: Filename to look for
    :return: List of found file paths
    """
    folder = Path(folder_path)  # Creating a Path Object
    found_paths = list(folder.rglob(target_filename))  # Recursive search for files
    return [str(path) for path in found_paths]  # Converts a Path object to a string path


def select_files_from_dir(input_dir: str, file_pattern: str = '*.csv', filter_patterns: Iterable[str] = ["checkpoint"]) -> List[str]:
    """Select files from the input directory that meet the criteria"""
    
    filter_set = set(filter_patterns)
    path_obj = Path(input_dir).expanduser()

    # 1. Recursive matching of file_cattern
    matched_files = path_obj.rglob(file_pattern)

    # 2. Filter out files containing any blacklist substrings
    def _keep(file_path: Path) -> bool:
        name = file_path.name # Only make judgments on file names
        return not any(blk in name for blk in filter_set)

    kept = filter(_keep, matched_files)

    # 3. Return a list of path strings
    return [str(p) for p in kept]


############################################# File read and save #############################################

def read_responses_csv(file_path: str) -> List[Tuple[int, int, str, str, str]]:
    """Read responses from csv file, returns list of (category_id, task_id, prompt, question, response) tuples."""
    responses = []
    with open(file_path, mode="r", newline="", encoding="utf-8") as file:
      reader = csv.reader(file)
      responses = list(reader)[1:]
    return [[s[0], s[1], s[2].replace("\\n", "\n"), s[3], s[4].replace("\\n", "\n")] for s in responses]


def save_responses_csv(responses: List[Tuple[int, int, str, str, str]], output_file: str, with_HCoT: bool = False, HCoT_dataset: str = None):
    """Save responses to CSV file."""
    ensure_dir(os.path.dirname(output_file))
    responses = [[s[0], s[1], s[2].replace("\n", "\\n"), s[3], s[4].replace("\n", "\\n")] for s in responses]
    responses = [["category_id", "task_id", "Prompt", "question", "response"]] + responses
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(responses)

    # If the H-cot dataset is used
    if with_HCoT:
        save_format_descriptions_HCoT(HCoT_dataset, output_file)
        os.remove(output_file)


def read_formatted_descriptions_csv(file_path: str) -> List[Tuple[str, str]]:
    """Read descriptions from csv file, returns list of (category_id, task_id, Image, Prompt, question, Description) tuples."""
    descriptions = []
    with open(file_path, mode="r", newline="", encoding="utf-8") as file:
      reader = csv.reader(file)
      descriptions = list(reader)[1:]
    return [[s[0], s[1], s[2], s[3], s[4], s[5].replace("\\n", "\n")] for s in descriptions]


def save_judgements_csv(judgements: List[Tuple[int, int, str, str, str, Any]], output_file: str, judger_name: str = None, judgement_type: JudgementType = JudgementType.Origin, malicious_rate_str: str = None):
    """Save jailbreak judgements to csv file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    new_judgements = []
    for judgement in judgements:
        decision = list(judgement)
        decision[2] = judgement[2].replace("\n", "\\n")
        decision[-2] = judgement[-2].replace("\n", "\\n")
        if judgement_type == JudgementType.Origin:
            decision[-1] = judgement[-1]
        elif judgement_type == JudgementType.Num_1:
            decision[-1] = 1 if judgement[-1].startswith("unsafe") else 0
        elif judgement_type == JudgementType.Num_10:
            decision[-1] = 10 if judgement[-1].startswith("unsafe") else 0
        elif judgement_type == JudgementType.Bool:
            decision[-1] = True if judgement[-1].startswith("unsafe") else False
        else:
            raise ValueError("Invalid judgement type!")
        new_judgements.append(decision)

    # Construct header
    if malicious_rate_str is None:
      judge_title = f"Judgement - {judger_name}"
    else:
      judge_title = f"Judgement - {judger_name} ({malicious_rate_str})"
    judgements = [("category_id", "task_id", "Prompt", "question", "Response", judge_title)] + new_judgements

    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
      writer = csv.writer(f)
      writer.writerows(judgements)


def read_prompts_csv(file_path: str) -> List[Tuple[str, int, int, str, str, str]]:
    """Read prompt datas from csv file, returns list of (dataset, category_id, task_id, category_name, question, instruction) tuples."""
    datas = []
    with open(file_path, mode="r", newline="", encoding="utf-8") as file:
      reader = csv.reader(file)
      datas = list(reader)[1:]
    return [[s[0], s[1], s[2], s[3], s[4].replace("\\n", "\n"), s[5].replace("\\n", "\n")] for s in datas]


def save_prompt_judgements_csv(judgements: List[Tuple[str, int, int, str, str, str, any]], output_file: str, judger_name: str = None, judgement_type: JudgementType = JudgementType.Origin, malicious_rate_str: str = None):
    """Save prompt jailbreak judgements to csv file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    new_judgements = []
    for s in judgements:
        if judgement_type == JudgementType.Origin:
            j = s[6]
        elif judgement_type == JudgementType.Num_1:
            j = 1 if s[6].startswith("unsafe") else 0
        elif judgement_type == JudgementType.Num_10:
            j = 10 if s[6].startswith("unsafe") else 0
        elif judgement_type == JudgementType.Bool:
            j = True if s[6].startswith("unsafe") else False
        else:
            raise ValueError("Invalid judgement type!")

        new_judgements.append([s[0], s[1], s[2], s[3], s[4].replace("\n", "\\n"), s[5].replace("\n", "\\n"), j])

    # Construct header
    if malicious_rate_str is None:
      judge_title = f"Jailbreak Judgements - {judger_name}"
    else:
      judge_title = f"Jailbreak Judgements - {judger_name} ({malicious_rate_str})"
    new_judgements = [("dataset", "category_id", "task_id", "category_name", "question", "instruction", judge_title)] + new_judgements

    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
      writer = csv.writer(f)
      writer.writerows(new_judgements)

############################################# Format response file #############################################

def save_format_descriptions(dataset_csv: str, description_csv: str):
    # Read Table 1 and Table 2
    table1 = pd.read_csv(dataset_csv)
    table2 = pd.read_csv(description_csv)

    # Remove the ".xx" suffix from the index string and split the index string into category_id and task_id
    table2[['category_id', 'task_id']] = table2['Image'].str.replace(r'\.[^.]+$', '', regex=True).str.split('_', expand=True)

    # Convert category_id and task_id to numeric types
    table2['category_id'] = pd.to_numeric(table2['category_id'], errors='coerce')
    table2['task_id'] = pd.to_numeric(table2['task_id'], errors='coerce')

    # Sort table 2 in ascending order by category_id and task_id
    table2.sort_values(by=['category_id', 'task_id'], inplace=True)
    # table1.sort_values(by=['category_id', 'task_id'], inplace=True)

    # Ensure that the data types of the category_id and task_id columns in Table 1 are consistent with those in Table 2.
    table1['category_id'] = pd.to_numeric(table1['category_id'], errors='coerce')
    table1['task_id'] = pd.to_numeric(table1['task_id'], errors='coerce')

    # Merge Table 1 and Table 2 based on category_id and task_id
    # Assume that the column name for the question in Table 1 is 'question'
    result = pd.merge(table2, table1[['category_id', 'task_id', 'question']],
                      on=['category_id', 'task_id'], how='left')

    result = result[['category_id', 'task_id', 'Image', 'Prompt', 'question', 'Description']]

    # Save the results to a new Excel file
    result.to_csv(description_csv, index=False)

# Example
# dataset_csv = './SafeBench-Car-2.0.csv'
# description_csv = './description_Janus-1.3B.csv'
# save_format_descriptions(dataset_csv, description_csv)


def save_format_descriptions_HCoT(dataset_csv: str, description_csv: str):
    # Read two CSV files
    df1 = pd.read_csv(description_csv)
    df2 = pd.read_csv(dataset_csv)

    # 1. Remove the line break in the prompt
    df1['Prompt'] = df1['Prompt'].str.replace('\\n', '\n', regex=False)

    # 2. Extract the required columns from df2 and remove duplicates (to prevent duplicate matches)
    df2_subset = df2[['category_id', 'task_id', 'instruction']].drop_duplicates()

    # 3. Merge df1 and df2_subset based on category_id and task_id
    df_merged = pd.merge(df1, df2_subset, on=['category_id', 'task_id'], how='left')

    # 4. Replace the question column with the value of the instruction column
    df_merged['question'] = df_merged['instruction']

    # 5. Delete the temporary instruction column (optional)
    df_merged.drop(columns=['instruction'], inplace=True)

    # Construct output name
    base, ext = os.path.splitext(description_csv)
    output_csv = f"{base}_formatted{ext}"

    # 6. Save the results to a new file
    df_merged.to_csv(output_csv, index=False)

    # print(f"Save the results to: {output_csv}")

# Example
# dataset_csv = './data/MaliciousEducator-M2-Filtered-Hcot-O3.csv'
# description_csv = './data/responses_o3-2025-04-16_D3_1.csv'
# save_format_descriptions_HCoT(dataset_csv, description_csv)