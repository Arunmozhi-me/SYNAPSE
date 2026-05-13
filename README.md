# SYNAPSE
A synthetic realism-constrained academic multiplex network dataset that captures researcher relationships across co-authorship, projects, grants, publications, and interactions.
# Academic Multiplex Network Dataset

## Overview
This repository contains a synthetic, realism-constrained **Academic Multiplex Network Dataset** developed to model researcher relationships across multiple academic contexts. 

## Dataset Files
- `researchers_profile.csv` – Researcher attributes, topic vectors, and interdisciplinarity scores
- `projects.csv` – Research projects, domains, timelines, PIs, and Co-Investigators
- `grants.csv` – Grant details, funding agencies, amounts, domains, and investigator teams
- `publications.csv` – Publications, authors, keywords, journals, citations, and linked projects/grants
- `interactions.csv` – Temporal researcher interactions arising from projects, grants, and organic academic activities

## Network Layers
The dataset can be used to construct:
1. **Co-authorship network**
2. **Project collaboration network**
3. **Grant collaboration network**
4. **Interaction network**

A **unified multiplex graph** can also be created by combining all layers.

## Key Features
- Common researcher nodes across all layers
- Domain-aware and workload-aware project allocation
- Temporal consistency among careers, projects, grants, publications, and interactions
- Support for isolated and low-participation researchers
- Semantic features such as `topic_vector` and `inter_score`
- Suitable for structural and learning-based network analysis

## **Intended Uses
- Community detection in multiplex networks
- Overlapping and dynamic community analysis
- Graph neural networks and embedding methods
- Link prediction
- Dataset validation using small-world, modularity, assortativity, and degree-distribution analysis

## Reproducibility
The dataset is generated using rules and fixed randomness where applicable, enabling reproducible experiments and validation.

## Citation
If you use this dataset or code, please cite the associated repository and DOI once available.

## Author
**Arunmozhi Mourougappane**  
PhD Scholar, Department of Computer Science  
Pondicherry University
