# Replication for the Paper: *ArchISMiner: Automatic Mining of Architectural Issue–Solution Pairs from Online Developer Communities*

This replication package accompanies the paper *ArchISMiner: Automatic Mining of Architectural Issue–Solution Pairs from Online Developer Communities*. 
It provides an overview of the ArchISMiner framework, the source code implementing its components, and the dataset used in our study.

## 🚨 Introduction

**ArchISMiner** is a framework for automatically mining architectural knowledge from Stack Overflow (SO) posts. It comprises two complementary components:

- **ArchPI** – Identifies architecture-related posts (ARPs) from the broader set of SO posts.  
- **ArchISPE** – Extracts architectural issue–solution pairs from the identified ARPs to capture task-specific architectural knowledge.

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



