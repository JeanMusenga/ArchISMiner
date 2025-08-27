# Replication for the Paper: *ArchISMiner: Automatic Mining of Architectural Issue–Solution Pairs from Online Developer Communities*

This is the replication package for the paper: *ArchISMiner: Automatic Mining of Architectural Issue–Solution Pairs from Online Developer Communities*.  
It provides an overview of the ArchISMiner framework, the source code implementing its components and baselines, and the dataset used in our study.

## 🚨 Introduction and Research Methodology Overview

Stack Overflow (SO), a leading online community forum, is a rich source of software development knowledge. However, locating architectural knowledge (e.g., architectural solutions) remains challenging due to the overwhelming volume of unstructured content and fragmented discussions. Developers must manually sift through posts to find relevant architectural insights, which is both time-consuming and error-prone. To address this, we introduce **ArchISMiner**, a framework for automatically mining architectural knowledge from SO. The research methodology employed is illustrated below:

![Overview of Research Methodology](images/OverviewOftheReseachMethod.png)

## 🏗️ ArchISMiner Framework

**ArchISMiner** is a framework for automatically mining architectural knowledge from Stack Overflow (SO) posts. It comprises two complementary components:

- **ArchPI** – Identifies Architecture-Related Posts (ARPs) from the broader set of SO posts. Specifically, it leverages a diverse set of models, including traditional machine learning (ML), deep learning (DL), state-of-the-art pre-trained language models (PLMs), and large language models (LLMs), to select the optimal model for this task.

  ![ARPs Identification Component](images/ARPs_Indetification_Component.png)

- **ArchISPE** – Extracts architectural issue–solution pairs from the identified ARPs to capture task-specific architectural knowledge. Given an ARP (i.e., a Question_body and its corresponding Answer_body) consisting of 𝑛 sentences, ArchISPE extracts a small set of key sentences that explicitly express architectural issues and solutions, and generates a concise, self-contained issue-solution pair for the post.

  ![ArchISPE Component](images/ArchISPE_Component.png)

Together, these components enable software engineers, particularly architects and developers, to efficiently gather relevant and useful architectural knowledge from online developer communities.

## 📁 Repository Structure

├── data/ # Dataset used in the study

├── archispe/ # Source code of the ArchISPE framework

├── baselines/ # Baseline models for comparison

├── evaluation/ # Evaluation scripts and metrics

├── results/ # Output results from experiments

└── README.md # This file


## Dataset Description

The `data/` directory includes:
- `questions.14932_ARPs_and_Programming_Posts. xlsx` – contains posts labeled as APRs and programming posts (i.e., 7,466 ARPs and 7,466 Programming Posts) used to train and test the classifiers. 
- `366_ARPs_for_extracting_Issue_Solution_Pairs.xlsx` – contains randomly selected 366 ARPs from 7,466 ARPs for extracting architectural issue-solution pairs.
- `ArchIssueSolutionBench.xlsx` – Manually annotated architectural issue–solution pairs. contains our proposed architectural benchmark used to evaluate the potential techniques for extracting architectural issue-solution pairs from online developer community forums. 

---


## 📝 Citation

```bibtex
@article{Musenga2025ArchISMiner,
  author = {Musengamana Jean de Dieu and Ruiyin Li and Peng Liang and Mojtaba Shahin and Muhammad Waseem and Zengyang Li and Bangchao Wang and Arif Ali Khan and Mst Shamima Aktar},
  title = {{ArchISMiner: Automatic Mining of Architectural Issue–Solution Pairs from Online Developer Communities}},
  journal={arXiv preprint arXiv:xxx.xxxx},
  year={2025}
}
```
