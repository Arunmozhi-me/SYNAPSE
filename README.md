# SYNAPSE
SYNAPSE is a synthetic multiplex academic social network dataset with co-authorship, project, grant, and interaction layers. It includes researcher attributes and temporal domain evolution to support studies on collaboration patterns, community detection, link prediction, and network analysis.

# Dataset Overview

SYNAPSE contains a synthetic academic network of 3,000 researchers. Each researcher is associated with profile-level attributes such as research domain, department, academic role, and career-related information. The dataset also includes temporal domain evolution, which represents changes in researchers’ academic domains over time.  The network can be organized as a multiplex graph with multiple relationship layers. Each layer captures a different type of academic relationship. These layers can be analyzed separately or combined into a unified multiplex graph. Each edge in the dataset is enriched with attributes spanning four dimensions:
  **Structural** — network position and connection strength
  **Semantic**   — topic and field alignment
  **Contextual** — institutional and role-based information
  **Temporal**   — time-stamped evolution of relationships

**Network Layers**

SYNAPSE generates four main network layers and one unified graph.

Layer	Description
Co-authorship Layer	- Generated from publication records. An edge is created between two researchers if they are co-authors of at least one publication.
Project Collaboration Layer - 	Generated from project participation records. Researchers are connected if they are part of the same research project.
Grant Collaboration Layer	- Generated from grant records. Researchers are connected if they are involved in the same funded grant.
Interaction Layer - 	Generated from interaction records such as meetings, discussions, seminars, or informal exchanges.
**How to Run**

**Clone the repository:**

git clone https://github.com/your-username/SYNAPSE.git
cd SYNAPSE

**Install required Python packages:**

pip install pandas numpy networkx

**Run the data generation scripts in order:**

python scripts/researchers_profile.py
python scripts/projects.py
python scripts/grants.py
python scripts/publications.py
python scripts/nteractions.py

After execution, the generated CSV files will be available in the data/ directory.

ossible Research Applications

T**he generated SYNAPSE data can be used for:**

1. Synthetic academic social network generation
2. Multiplex network analysis
3. Community detection in academic networks
4. Link prediction in multiplex networks
5. Layer-wise network comparison
6. Temporal collaboration analysis
7. Research domain evolution analysis
8. Domain drift analysis
   
**Reproducibility**

The generated dataset can be reproduced using a fixed random seed. The default version uses:
SEED = 42
NUMBER_OF_RESEARCHERS = 3000

Users may modify the seed, number of researchers, and generation constraints to create different versions of the synthetic academic network.
Citation

**If you use this repository or generated dataset in your research, please cite:**

Arunmozhi Mourougappane. SYNAPSE: 

License

This repository is released for academic and research purposes.

**Recommended license:**

Creative Commons Attribution 4.0 International License (CC BY 4.0)

**Disclaimer**

SYNAPSE generates synthetic academic network data. It does not contain real researcher identities, real academic relationships, or real institutional records. The generated data is intended for research, experimentation, benchmarking, and educational use.
