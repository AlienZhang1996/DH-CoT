# Datasets

This directory contains formatted source data, cleaned RTA subsets, and prepared H-CoT inputs. All CSVs are encoded in UTF-8, and sample counts exclude the header row.

- **`1_Original_Datasets_Formatted/`**: the five formatted source subsets, totaling 1,680 questions.
- **`2_RTA/`**: the five cleaned RTA subsets, the three H-CoT input variants, and four 50-question experiment input files.

## Base datasets

| Dataset | Formatted source subset | Cleaned RTA subset |
| --- | --- | --- |
| SafeBench | [500 rows](1_Original_Datasets_Formatted/1_SafeBench_Formatted.csv) | [350 rows](2_RTA/1_RTA-SafeBench.csv) |
| QuestionSet | [390 rows](1_Original_Datasets_Formatted/2_QuestionSet_Formatted.csv) | [270 rows](2_RTA/2_RTA-QuestionSet.csv) |
| JailbreakStudy | [40 rows](1_Original_Datasets_Formatted/3_JailbreakStudy_Formatted.csv) | [35 rows](2_RTA/3_RTA-JailbreakStudy.csv) |
| BeaverTails | [700 rows](1_Original_Datasets_Formatted/4_BeaverTails_Formatted.csv) | [500 rows](2_RTA/4_RTA-BeaverTails.csv) |
| MaliciousEducator | [50 rows](1_Original_Datasets_Formatted/5_MaliciousEducator_Formatted.csv) | [50 rows](2_RTA/5_RTA-MaliciousEducator.csv) |
| **Total** | **1,680 rows** | **1,205 rows** |

The formatted source files contain the study's input subsets. The first five files in `2_RTA/` contain the cleaned RTA data. These sizes correspond to Table 7, and the cleaned RTA data are used in the evaluations in Tables 8, 9, and S8.

## H-CoT variants and 50-question inputs

Files **6-8** in `2_RTA/` contain the OpenAI, Gemini, and DeepSeek H-CoT input variants. Each variant covers the same 50 MaliciousEducator questions and is associated with the experiments in Tables 9 and 12.

Files **9-12** provide the 50-question subsets and prepared inputs for the cross-dataset experiments in Table 10:

| File | Contents |
| --- | --- |
| [9_RTA-SafeBench-50.csv](2_RTA/9_RTA-SafeBench-50.csv) | 50 questions from RTA-SafeBench |
| [10_RTA-QuestionSet-50.csv](2_RTA/10_RTA-QuestionSet-50.csv) | 50 questions from RTA-QuestionSet |
| [11_RTA-SafeBench-Hcot-Full-Request-50.csv](2_RTA/11_RTA-SafeBench-Hcot-Full-Request-50.csv) | Prepared H-CoT inputs for the same 50 SafeBench questions |
| [12_RTA-QuestionSet-Hcot-o3-mini-2.0-Full-Request-50.csv](2_RTA/12_RTA-QuestionSet-Hcot-o3-mini-2.0-Full-Request-50.csv) | Prepared H-CoT inputs for the same 50 QuestionSet questions |

These variants and subsets do not add new unique questions to the 1,205-question base RTA collection. Partial evaluation results and their experimental conditions are described in [examples/README.md](../examples/README.md).

## CSV fields

The six columns are `dataset`, `category_id`, `task_id`, `category_name` (or `category` for BeaverTails), `question`, and `instruction`. In prepared H-CoT files, `question` is the full request and `instruction` contains the underlying base question. Identifiers are local to each file and preprocessing stage; `task_id` alone is not a globally unique identifier.

## Sources and licenses

The root MIT license covers project software, not the complete data collection. These source terms also apply where source text appears in example results.

| Source | Upstream project | Data terms |
| --- | --- | --- |
| SafeBench | [FigStep](https://github.com/CryptoAILab/FigStep) | MIT; [license copy](licenses/FigStep_MIT.txt) |
| QuestionSet | [JailbreakLLMs](https://github.com/TrustAIRLab/JailbreakLLMs) | MIT; [license copy](licenses/JailbreakLLMs_MIT.txt) |
| BeaverTails | [BeaverTails](https://github.com/PKU-Alignment/beavertails) | CC BY-NC 4.0 for data; [license copy](licenses/BeaverTails_DATA_CC-BY-NC-4.0.txt) |
| MaliciousEducator / H-CoT | [H-CoT](https://github.com/dukeceicenter/jailbreak-o1o3-deepseek-r1) | CC BY-NC-SA 4.0; [license copy](licenses/HCoT_CC-BY-NC-SA-4.0.txt) |
| JailbreakStudy | [Author project: prohibited scenarios](https://sites.google.com/view/llm-jailbreak-study/prohibited-scenario) | No explicit independent dataset redistribution license was identified; consult the source authors for reuse permission. |

RTA records were filtered and, for some sources, edited from these materials. Prepared H-CoT inputs and model-generated text remain subject to applicable source and provider terms. The project does not grant additional rights to third-party material.
