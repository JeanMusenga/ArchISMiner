# Replication for the Paper: *ArchISMiner: Automatic Mining of Architectural Issue–Solution Pairs from Online Developer Communities*

This is the Replication Package for the Paper:  “Automatic Mining of Architectural Issue–Solution Pairs from Online Developer Communities. This repository contains an introduction to the ArchISMiner framework, source code for the implementations of each component, the dataset, and ArchISMiner.

## 🚨 Introduction

We developed a framework to mine architectural knowledge by analyzing ARPs from SO. Our framework comprises two complementary learning approaches, ArchPI and ArchISPE, which operate
in two main phases: a) Automatic Identification of Architecture-Related Posts and b) Automatic Extraction of Architectural Issue-Solution Pairs.

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



