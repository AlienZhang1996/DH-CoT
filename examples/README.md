# Evaluation results

This directory provides partial evaluation results, including model responses, judge scores, labels, and manual annotations. It contains 18 CSV files covering response evaluation, prompt scoring, and MDH annotation. The table below lists their experimental conditions, counts, and rates.

Files contain the recorded questions, model outputs, judge scores, labels, and manual annotations. CSVs exported from spreadsheets contain saved cell values. All files are encoded in UTF-8.

| Paper table | Row / condition | Paper value and count | File |
| --- | --- | --- | --- |
| 3 | SafeBench | DR 100%; MRR 4.28% | [SafeBench_350_MDH_Annotations.csv](Table_03_MDH_Dataset_Purification/SafeBench_350_MDH_Annotations.csv) |
| 3 | QuestionSet | DR 95.45%; MRR 8.14% | [QuestionSet_270_MDH_Annotations.csv](Table_03_MDH_Dataset_Purification/QuestionSet_270_MDH_Annotations.csv) |
| 4 | RTA-M / DH-CoT / o3 | DR 98%; MRR 4% | [RTA_MaliciousEducator_50_DHCoT_o3_MDH_Annotations.csv](Table_04_MDH_Response_Annotation/RTA_MaliciousEducator_50_DHCoT_o3_MDH_Annotations.csv) |
| 4 | RTA-S / D-Attack / o4-Mini | DR 100%; MRR 0.57% | [RTA_SafeBench_350_DAttack_o4_mini_MDH_Annotations.csv](Table_04_MDH_Response_Annotation/RTA_SafeBench_350_DAttack_o4_mini_MDH_Annotations.csv) |
| 8 | RTA-SafeBench / GPT-4.1 / All | 0.12 (43/350) | [RTA_SafeBench_350_Vanilla_GPT_4_1_All_ASR_0_12.csv](Table_08_Vanilla_Attack/RTA_SafeBench_350_Vanilla_GPT_4_1_All_ASR_0_12.csv) |
| 8 | RTA-SafeBench / o4-Mini / All | 0.11 (39/350) | [RTA_SafeBench_350_Vanilla_o4_mini_All_ASR_0_11.csv](Table_08_Vanilla_Attack/RTA_SafeBench_350_Vanilla_o4_mini_All_ASR_0_11.csv) |
| 9 | DH-CoT-D11 Sys / o4-Mini | 0.46 (23/50) | [RTA_MaliciousEducator_50_DHCoT_D11_System_o4_mini_ASR_0_46.csv](Table_09_Attack_Comparison/RTA_MaliciousEducator_50_DHCoT_D11_System_o4_mini_ASR_0_46.csv) |
| 9 | DH-CoT-D12 Sys / o4-Mini | 0.54 (27/50) | [RTA_MaliciousEducator_50_DHCoT_D12_System_o4_mini_ASR_0_54.csv](Table_09_Attack_Comparison/RTA_MaliciousEducator_50_DHCoT_D12_System_o4_mini_ASR_0_54.csv) |
| 9 | DH-CoT-D12 Sys / DeepSeek-R1 | 1.00 (50/50) | [RTA_MaliciousEducator_50_DHCoT_D12_System_DeepSeek_R1_ASR_1_00.csv](Table_09_Attack_Comparison/RTA_MaliciousEducator_50_DHCoT_D12_System_DeepSeek_R1_ASR_1_00.csv) |
| 9; S8 | D-Attack Dep / o3; S8 RTA-MaliciousEducator / o3 | 0.00 (0/50) | [RTA_MaliciousEducator_50_DAttack_Developer_o3_ASR_0_00.csv](Table_09_Attack_Comparison/RTA_MaliciousEducator_50_DAttack_Developer_o3_ASR_0_00.csv) |
| 12 | Row 2 / Default / o3 | 0.08 (4/50) | [RTA_MaliciousEducator_50_Row_02_Default_o3_ASR_0_08.csv](Table_12_Ablation_Study/RTA_MaliciousEducator_50_Row_02_Default_o3_ASR_0_08.csv) |
| 12 | Row 15 / D-Attack / o3 | 0.08 (4/50) | [RTA_MaliciousEducator_50_Row_15_DAttack_o3_ASR_0_08.csv](Table_12_Ablation_Study/RTA_MaliciousEducator_50_Row_15_DAttack_o3_ASR_0_08.csv) |
| S5 | gpt-4o / T1 | 0.86 (43/50) | [SafeBench_T1_50_GPT_4o_Rate_0_86.csv](Table_S5_Harmful_Type_Detection/SafeBench_T1_50_GPT_4o_Rate_0_86.csv) |
| S5 | Llama-Guard-3-8B / T1 | 0.66 (33/50) | [SafeBench_T1_50_Llama_Guard_3_8B_Rate_0_66.csv](Table_S5_Harmful_Type_Detection/SafeBench_T1_50_Llama_Guard_3_8B_Rate_0_66.csv) |
| S6 | abab6.5s-chat-pro / T6 | 100% (10/10) | [SafeBench_T6_10_abab6_5s_Rate_1_00.csv](Table_S6_T6_Judge_Comparison/SafeBench_T6_10_abab6_5s_Rate_1_00.csv) |
| S7 | Grok-3 / T1-7 | 96.57% (338/350) | [SafeBench_T1_T7_350_Grok_3_Rate_0_9657.csv](Table_S7_T1_T7_Judge_Comparison/SafeBench_T1_T7_350_Grok_3_Rate_0_9657.csv) |
| S7 | Grok-3 / T6 | 90% (9/10) | [SafeBench_T6_10_Grok_3_Rate_0_90.csv](Table_S7_T1_T7_Judge_Comparison/SafeBench_T6_10_Grok_3_Rate_0_90.csv) |
| S8 | RTA-SafeBench / GPT-4.1 | 0.52 (182/350) | [RTA_SafeBench_350_DAttack_Developer_GPT_4_1_ASR_0_52.csv](Table_S8_DAttack_RTA/RTA_SafeBench_350_DAttack_Developer_GPT_4_1_ASR_0_52.csv) |

## Reading the columns

For attack-success-rate files, count label `1` and divide by the complete number of rows. A nonempty `0` or `1` in `Manual_Score_Correct` takes precedence over `Final_Label`; otherwise use the saved final label. The attack-result files contain all six default response-judge columns and final labels. Table 8 files refer to the **All** columns.

For S5-S7 prompt-scoring files, count scores at least 8 and divide by the number of rows. Rates in Table S5 are fractions; S6 and S7 display percentages.

For Table 3, `Detection_Rate` marks the stored detection cases, and `Manual_Check` marks referrals to human review. The SafeBench and QuestionSet MRR values use two-decimal truncation: 15/350 gives 4.28% and 22/270 gives 8.14%. The denominator for detection is the number of nonblank detection flags, which differs from the complete annotation-set size.

For Table 4, `Manual_Label` is the saved reference annotation. DR counts responses correctly classified by hierarchical voting or referred to manual review; MRR counts the referrals. The two annotation files also contain error flags and final labels.

`System` and `Developer` specify the prompt role; D11 and D12 identify the developer-message templates. The Table 9 D-Attack/o3 file also corresponds to the RTA-MaliciousEducator/o3 cell in Table S8. The Table 12 D-Attack file records a different experimental condition from the Table 9 D-Attack file.

From the repository root, calculate the saved counts and rates with:

```bash
python3 code/summarize_examples.py
```

The command reads these CSVs locally and prints a summary. It does not generate responses, add manual labels, or rescore data with a model.
