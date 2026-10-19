# Code

## Environment

The experiment environment uses Python 3.10.13, PyTorch 2.3.0, and CUDA 12.1. Python dependencies are listed in `requirements.txt`. Local Llama Guard scoring requires model weights and a suitable GPU.

From the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements.txt
```

Set `DH_COT_API_BASE_URL` to your OpenAI-compatible provider endpoint and `DH_COT_API_KEY` to its credential. Set `DH_COT_MODELS_DIR` to an absolute path containing the Llama Guard model directories when using local scoring. `.env.example` documents these names; the scripts read the process environment and do not automatically load a `.env` file.

Set the input, output, and model settings in the YAML configuration before running an experiment. Model availability depends on the configured provider.

## Entry points

Run research entry points with `code/` as the working directory.

| Script | Purpose | Default configuration |
| --- | --- | --- |
| `Generate_Response.py` | Generate model responses | `config/cfg_Generate_Response.yaml` |
| `MDH_Auto_Annotation_Multiple.py` | Score response CSVs and apply hierarchical voting | `config/cfg_MDH_JRD_Auto-Annotation_Multiple.yaml` |
| `MDH_Auto_Annotation_Multithread.py` | Threaded MDH scoring and voting | `config/cfg_MDH_JRD_Auto-Annotation_Multiple_Multithread.yaml` |
| `Scoring_Prompt_by_Commercial_LLM.py` | Commercial-model prompt scoring | `config/cfg_Scoring_Prompt_by_Commercial_LLM.yaml` |
| `Scoring_Prompt_by_Llama_Guard.py` | Local Llama Guard prompt scoring | `config/cfg_Scoring_Prompt_by_Llama_Guard.yaml` |
| `Scoring_Response_by_Commercial_LLM.py` | Commercial-model response scoring | `config/cfg_Scoring_Response_by_Commercial_LLM.yaml` |
| `Scoring_Response_by_Llama_Guard.py` | Local Llama Guard response scoring | `config/cfg_Scoring_Response_by_Llama_Guard.yaml` |
| `Format_Response.py` | Restore base questions in H-CoT response files | `config/cfg_Format_Response.yaml` |
| `summarize_examples.py` | Summarize saved example results without model calls | No configuration required |

For example, after configuring the environment and YAML:

```bash
cd code
python Generate_Response.py --config config/cfg_Generate_Response.yaml
```

Relative paths in YAML files are interpreted from `code/`. The default generator scans all CSVs in `../datasets/2_RTA`, including the full subsets, H-CoT variants, and 50-question inputs. To run one experimental condition, copy the required input files into a separate working directory and set `input_dir` to that directory. The generator recognizes `Hcot` in input filenames and restores the base question from `instruction` after response generation.

Generation outputs are grouped by dataset and user-template name, for example `outputs/responses/1_RTA-SafeBench/Vanilla/responses_gpt-4o.csv`. Set response-scoring input paths to the generated CSVs you wish to score. `Format_Response.py` edits matching response files in place; use a working copy. MDH scoring marks ambiguous cases for human review. Saved manual corrections must be included when calculating final results.

## Templates

- `Templates/Developer_Messages.py`: D1-D12 developer-message templates and the `None` and `Normal` baselines.
- `Templates/User_Templates.py`: the existing `Vanilla`, `Car_Answer`, and `Car_Modify` user templates.
- `Templates/Judger_Templates.py`: prompt and response scoring templates.

Partial evaluation results are provided in [examples/](../examples/). Hosted models can change over time, so generated responses may vary across runs.
