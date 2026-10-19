# Jailbreaking Commercial Black-Box LLMs with Explicitly Harmful Prompts

Code, datasets, and partial evaluation results for [Jailbreaking Commercial Black-Box LLMs with Explicitly Harmful Prompts](https://arxiv.org/abs/2508.10390).

**Chiyu Zhang, Lu Zhou, Xiaogang Xu, Jiafei Wu, Liming Fang, and Zhe Liu**

This work studies jailbreak attacks against commercial black-box LLMs using explicitly harmful prompts. It introduces **MDH**, a malicious-content detection framework combining model-based annotation with human assistance, and uses it to construct the **RTA** red-teaming dataset suite.

The repository also provides the **D-Attack** and **DH-CoT** experimental implementations, including developer-message templates, H-CoT inputs, and model-response scoring tools.

## Method overview

### MDH

MDH combines model-based judgment with human assistance for malicious-content detection and red-teaming dataset cleaning.

| <a href="figures/prompt_taxonomy.png"><img src="figures/prompt_taxonomy.png" alt="Prompt taxonomy" width="390"></a> | <a href="figures/mdh_pipeline.png"><img src="figures/mdh_pipeline.png" alt="MDH workflow" width="390"></a> |
| :---: | :---: |
| Prompt taxonomy. SG denotes safeguards. | MDH workflow and dataset cleaning. |

### D-Attack and DH-CoT

The method overview illustrates the template components and adversarial context alignment used by D-Attack and DH-CoT.

<p align="center"><a href="figures/attack_pipeline.png"><img src="figures/attack_pipeline.png" alt="D-Attack and DH-CoT method overview" width="90%"></a></p>

## RTA dataset series

The five cleaned RTA subsets contain 1,205 questions. Their harmful-category distributions are shown below.

<p align="center"><a href="figures/rta_distribution.png"><img src="figures/rta_distribution.png" alt="Harmful-category distributions in the RTA dataset suite" width="90%"></a></p>

Dataset sizes, file descriptions, and source terms are provided in the [dataset guide](datasets/README.md).

## Evaluation results

### Vanilla queries

The following table shows vanilla-query results on the original and RTA datasets. Values are presented as 1 minus the rejection rate. AC denotes adult content.

<p align="center"><a href="figures/vanilla_attack_results.png"><img src="figures/vanilla_attack_results.png" alt="Vanilla-query results before and after dataset cleaning" width="85%"></a></p>

### D-Attack

Attack success rates of D-Attack on the five RTA subsets:

<p align="center"><a href="figures/d_attack_results.png"><img src="figures/d_attack_results.png" alt="D-Attack success rates on the RTA datasets" width="65%"></a></p>

### Attack comparison

The table below compares attack success rates across methods and victim models on RTA-MaliciousEducator. Blue shading marks reasoning-model columns. Sys and Dep denote the system and developer prompt roles.

<p align="center"><a href="figures/attack_comparison.png"><img src="figures/attack_comparison.png" alt="Attack success rates across methods and models on RTA-MaliciousEducator" width="100%"></a></p>

Partial evaluation result files, including model responses, judge scores, and annotations, are provided in [examples/](examples/).

## Repository contents

| Directory | Contents |
| --- | --- |
| [code/](code/) | Response generation, MDH annotation, model scoring, and D1-D12 templates |
| [datasets/1_Original_Datasets_Formatted/](datasets/1_Original_Datasets_Formatted/) | Five formatted source subsets |
| [datasets/2_RTA/](datasets/2_RTA/) | Five cleaned RTA subsets, three H-CoT variants, and four 50-question input files |
| [examples/](examples/) | Partial evaluation results: 18 response-scoring, prompt-scoring, and annotation files |
| [figures/](figures/) | Method diagrams, RTA data distributions, and evaluation-result tables |

Result filenames identify the dataset, method or judge, model, and, where applicable, template ID and prompt role. The [evaluation file guide](examples/README.md) describes the experimental conditions, counts, and rates.

## Read the data and results

All dataset and example files are UTF-8 CSVs. The five cleaned RTA subsets contain **1,205 questions** in total. Source-subset sizes and column definitions are in the [dataset guide](datasets/README.md).

To summarize the stored example labels and scores locally, from the repository root:

```bash
python3 code/summarize_examples.py
```

This command uses only the Python standard library and does not call a model. It reports counts and rates directly from the saved columns. Original question, response, score, and manual-annotation values are retained in the CSVs.

## Use the experiment code

See [code/README.md](code/README.md) for the environment, entry points, and configuration. Experiments require model-provider credentials and, for local Llama Guard scoring, model weights and a suitable GPU. Credentials are read from environment variables; [.env.example](.env.example) lists their names.

[Developer_Messages.py](code/Templates/Developer_Messages.py) defines the D1-D12 templates. The standalone response-generation entry point uses the **developer** role; **System** and **Developer** in result filenames identify their respective experimental conditions.

## Citation and data terms

```bibtex
@misc{zhang2025jailbreaking,
  title={Jailbreaking Commercial Black-Box LLMs with Explicitly Harmful Prompts},
  author={Chiyu Zhang and Lu Zhou and Xiaogang Xu and Jiafei Wu and Liming Fang and Zhe Liu},
  year={2025},
  eprint={2508.10390},
  archivePrefix={arXiv}
}
```

The [MIT license](LICENSE) applies to the project software. Datasets and embedded third-party text retain their source terms, including the noncommercial conditions for BeaverTails and H-CoT. Attribution and source-specific terms are listed in [datasets/README.md](datasets/README.md) and [datasets/licenses/](datasets/licenses/). The collection contains harmful prompts and model responses used for safety evaluation.
