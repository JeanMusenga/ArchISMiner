# Replication for the Paper: *ArchISMiner: Automatic Mining of Architectural Issue–Solution Pairs from Online Developer Communities*

This is the replication package for the paper: *ArchISMiner: Automatic Mining of Architectural Issue–Solution Pairs from Online Developer Communities*. 
It provides an overview of the ArchISMiner framework, the source code implementing its components and baselines, and the dataset used in our study.

## 🚨 Introduction and  an overview of the research methodology

Stack Overflow (SO), a leading online community forum, is a rich source of software development knowledge. However, locating architectural knowledge (e.g., architectural solutions) remains challenging due to the overwhelming volume of unstructured content and fragmented discussions. Developers must manually sift through posts to find relevant architectural insights, which is both time-consuming and error-prone. IN this study, we follow a rigorous research method and introduce ArchISMiner, a novel framework for mining architectural knowledge from SO.

![Alt text](images/OverviewOftheReseachMethod.png)

## 🚨 ArchISMiner

**ArchISMiner** is a framework for automatically mining architectural knowledge from Stack Overflow (SO) posts. It comprises two complementary components:

- **ArchPI** – Identifies Architecture-Related Posts (ARPs) from the broader set of SO posts. Specifically, it leverages a diverse set of models, including traditional machine learning (ML), deep learning (DL), state-of-the-art pre-trained language models (PLMs), and large language models (LLMs), to select the optimal model for this task.

  ![Alt text](images/ARPs_Indetification_Component.png)

- **ArchISPE** – Extracts architectural issue–solution pairs from the identified ARPs to capture task-specific architectural knowledge. Given an ARP (i.e., a Question_body and its corresponding Answer_body) consisting of 𝑛 sentences, ArchISPE extracts a small set of key sentences that explicitly express architectural issues and solutions, and generates a concise, self-contained issue-solution pair for the post.

   ![Alt text](images/ArchISPE_Component.png)
  
Together, these components enable software engineers—particularly architects and developers—to efficiently gather relevant architectural information from online developer communities.




This repository provides:
- The implementation of our proposed framework (**ArchISPE**)
- Baseline models for comparative evaluation
- The dataset used in our study

---

## 📁 Repository Structure

├── data/ # Dataset used in the study

├── archispe/ # Source code of the ArchISPE framework

├── baselines/ # Baseline models for comparison

├── evaluation/ # Evaluation scripts and metrics

├── results/ # Output results from experiments

└── README.md # This file


---

## 📊 Dataset Description

The `data/` folder contains:

- `questions.csv`: Stack Overflow questions  
- `answers.csv`: Corresponding answers  
- `annotations.json`: Manually annotated architectural issue–solution pairs  

---

## 📄 File Organization

More details about each subfolder and how to run the experiments are provided in their respective README files (if available). Please refer to the usage instructions in the `archispe/` and `evaluation/` folders.

---



