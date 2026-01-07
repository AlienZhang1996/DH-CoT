# Jailbreaking Commercial Black-Box LLMs with Explicitly Harmful Prompts

Official [PyTorch](https://pytorch.org/) implementation of the paper:

[Jailbreaking Commercial Black-Box LLMs with Explicitly Harmful Prompts](http://arxiv.org/abs/2508.10390)

[Chiyu Zhang](https://alienzhang1996.github.io/), [Lu Zhou](https://faculty.nuaa.edu.cn/zhoulu2020/zh_CN/index.htm), [Xiaogang Xu](https://scholar.google.com/citations?user=R65xDQwAAAAJ), [Jiafei Wu](https://dblp.org/pid/227/7227.html), [Liming Fang](https://scholar.google.com/citations?user=8p2FacYAAAAJ), [Zhe Liu](https://scholar.google.com/citations?user=Em0jNiUAAAAJ)



Existing black-box jailbreak attacks achieve certain success on non-reasoning models but degrade significantly on recent SOTA reasoning models. To improve attack ability, inspired by adversarial aggregation strategies, we integrate multiple jailbreak tricks into a single developer template. Especially, we apply **Adversarial Context Alignment** to purge semantic inconsistencies and use NTP (a type of harmful prompt) -based few-shot examples to guide malicious outputs, lastly forming **DH-CoT** attack with a fake chain of thought. In experiments, we further observe that existing red-teaming datasets include samples unsuitable for evaluating attack gains, such as BPs, NHPs, and NTPs. Such data hinders accurate evaluation of true attack effect lifts. To address this, we introduce **MDH**, a **M**alicious content **D**etection framework integrating LLM-based annotation with **H**uman assistance, with which we clean data and build **RTA** dataset suite. Experiments show that MDH reliably filters low-quality samples and that DH-CoT effectively jailbreaks models including GPT-5 and Claude-4, notably outperforming SOTA methods like H-CoT and TAP.



### MDH

|       ![子图1](./figures/fig1_unsuitable_samples.png)        | ![子图2](./figures/fig2_MDH_pipeline.png)  |
| :----------------------------------------------------------: | :----------------------------------------: |
| A Taxonomy of prompts (samples) in red teaming datasets. SG denotes Safeguards. | MDH workflow and its use in data cleaning. |

### D-Attack & DH-CoT

|         ![子图1](./figures/fig3_attacks_pipline.png)         |
| :----------------------------------------------------------: |
| Flowchart of D-Attack and DH-CoT, using examples from GPT-4o and o4-Mini. |



**Todo List**

- Upload experimental code
- Upload RTA datasets
- Upload MDH's jailbreak response detection judgment files



## RTA Dataset Series

### Data Distribution

<p align="center">
  <img src="./figures/fig4_RTA_construction.png" alt="子图1" width="100%">
</p>



### Dataset Cleaning Summary


|      Dataset      | Original Size | Current Size | Types | Removed | Modified | Edit-Removal Ratio (%) |
| :---------------: | :-----------: | :----------: | :---: | :-----: | :------: | :--------------------: |
|     SafeBench     |      500      |     350      | 7/10  |   150   |    38    |          37.6          |
|    QuestionSet    |      390      |     270      | 9/13  |   120   |    49    |         43.34          |
|  JailbreakStudy   |      40       |      35      |  7/8  |    5    |    8     |          32.5          |
|    BeaverTails    |      700      |     500      | 9/14  |   200   |   190    |         55.71          |
| MaliciousEducator |      50       |      50      | 8/10  |    0    |    0     |           0            |

The *Type* column shows the number of types after cleaning (removal / merging) and the original count. The *Removed* and *Modified* columns indicate samples removed and rewrote, respectively.



### Rejection Rate on Vanilla Attack

<p align="center">
    <img src='./figures/fig5_vallia_attack.png' alt='fig1_title.png' title='fig1_title.png' style="width:85%;" />
</p>

Rejection rates (reported as complements, which is 1 − rejection rate) for each dataset, to facilitate comparison with results in  section of *Jailbreak Results*. *All* includes all malicious types of samples; *w/o AC* excludes samples of Adult Content. Vanilla Attack refers to the situation where the prompt in the dataset is used to directly query the LLM.




## Jailbreak Results

### D-Attack

<p align="center">
    <img src='./figures/fig6_D-Atk_attack.png' alt='fig1_title.png' title='fig1_title.png' style="width:65%;" />
</p>

<p align="center">
    ASR of D-Attack on the RTA-series datasets.
</p>




### DH-CoT

<p align="center">
    <img src='./figures/fig7_DH-CoT_attack.png' alt='fig1_title.png' title='fig1_title.png' style="width:100%;" />
</p>

This table compares ASR of DH-CoT and D-Attack with current state-of-the-art jailbreak attacks on the RTA-MaliciousEducator dataset.. The compared methods fall into three categories: **1) query-based gray-box methods (top section)**, **2) template-based black-box methods (middle section)**, and **3) CoT-based black-box methods (bottom section)**. *Vanilla* denotes the attack success rate without any attack applied. PAIR and TAP are query-based gray-box methods, while DeepInception and SelfCipher are template-based black-box methods. H-CoT is a CoT-based black-box method. White-box methods require gradients or other internal model information and therefore cannot be evaluated on commercial black-box models. The column labels *3.5, 4o, 4.1, 5, 5.1, o1-m, o1, o3-m, o3, o4-m, 2.5-pro, 2.5-f-t, c35-s, c37-s, c4-s, c37-s-t, c4-s-t, d-v3, d-r1-0528*, and *d-r1* denote the victim models G*PT-3.5, GPT-4o, GPT-4.1, GPT-5, GPT-5.1, o1-Mini, o1, o3-Mini, o3, o4-Mini, Gemini-2.5-pro, Gemini-2.5-Flash-Thinking, Claude-3-5-Sonnet, Claude-3-7-Sonnet, Claude-3-7-Sonnet-Thinking, Claude-Sonnet-4, Claude-Sonnet-4-Thinking, Deepseek-V3, Deepseek-R1-250528*, and *Deepseek-R1*, respectively. *Sys* and *Dep* abbreviate the System and Developer prompt roles. A short dash (“-”) indicates the victim does not support the developer role. Blue-shaded columns mark victims that are reasoning models. All experimental results are computed using MDH. Each experiment is run three times and the best value is reported. The best result for each victim model is highlighted in bold.





