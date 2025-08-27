# Replication Package for *ArchISMiner: Automatic Mining of Architectural Issue–Solution Pairs from Online Developer Communities*

This replication package accompanies the paper *ArchISMiner: Automatic Mining of Architectural Issue–Solution Pairs from Online Developer Communities*.  
It includes:
- An overview of the ArchISMiner framework.
- Source code implementing its components and baselines.
- The dataset used in our study.

## 1. Introduction and Research Methodology Overview

Stack Overflow (SO), a widely used platform for software development discussions, contains abundant technical knowledge. However, extracting architectural knowledge (e.g., architectural solutions) is challenging due to the unstructured nature of posts and fragmented discussions. This requires considerable manual effort, making the process inefficient and error-prone.  

To address this, we introduce **ArchISMiner**, a framework for automatically mining architectural knowledge from SO. The research methodology employed is illustrated below:

![Overview of Research Methodology](images/OverviewOftheReseachMethod.png)

## 2. ArchISMiner Framework

**ArchISMiner** consists of two main components:

- **ArchPI** – Identifies Architecture-Related Posts (ARPs) from the full set of SO posts. It leverages a combination of traditional machine learning (ML), deep learning (DL), pre-trained language models (PLMs), and large language models (LLMs) to determine the most effective model for this task.  
  ![ARPs Identification Component](images/ARPs_Indetification_Component.png)

- **ArchISPE** – Extracts architectural issue–solution pairs from ARPs. Given a Question-Answer pair with *n* sentences, ArchISPE selects a subset of key sentences explicitly expressing architectural issues and solutions, forming a concise, self-contained issue–solution pair.  
  ![ArchISPE Component](images/ArchISPE_Component.png)

Together, these components enable software engineers to efficiently retrieve relevant architectural insights from large-scale developer discussions.

## 3. Repository Structure

├── data/ # Dataset used in the study
├── archispe/ # Source code for ArchISPE framework
├── baselines/ # Baseline models for comparison
├── evaluation/ # Scripts for evaluation and metrics computation
├── results/ # Experimental results
└── README.md # This file


## 4. Dataset Description

The `data/` directory includes:
- `questions.csv` – Stack Overflow questions.
- `answers.csv` – Corresponding answers.
- `annotations.json` – Manually annotated architectural issue–solution pairs.

**Dataset Size**: [Insert number of questions, answers, and annotated pairs].  
**Annotation Process**: [Briefly describe, e.g., conducted by two experts following guidelines].  

---

