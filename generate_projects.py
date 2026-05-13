"""
generate_projects.py
────────────────────
How titles are generated:
  - Real titles are analysed to extract structural patterns
    (e.g. "<Method> of <Topic> in <Domain>", "<Topic> for <Application>").
  - Those patterns become templates with domain-specific slot banks.
  - Every project title — whether the target is 500 or 5000 — is generated
    from these templates, so the corpus is uniform and extensible.
  - Domain is auto-detected from each generated title using keyword rules,
    ensuring title content always matches the assigned domain.
  - Cross-domain is capped at exactly 2 domains.

Run:
  python generate_projects.py

Here, the grant_id is not directly linked with the corresponding project_id; a separate file is executed to establish this mapping.
"""

import pandas as pd
import numpy as np
import random
import re

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                        # 1) PATHS & SETTINGS
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
RESEARCHERS_CSV  = "Researchers_Profile.csv"
OUT_PROJECTS_CSV = "generate_projects.csv"

RANDOM_SEED     = 42
TARGET_PROJECTS = 1300

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

PROJECT_TYPES = ["Internal", "External", "Consultancy"]
TYPE_WEIGHTS  = [0.30, 0.55, 0.15]

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                    # 2) DOMAIN KEYWORD MAP  (for auto-detection)
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
DOMAIN_KEYWORDS = {
    "Computer Science": [
        "machine learning", "deep learning", "neural network", "nlp",
        "natural language", "computer vision", "artificial intelligence",
        "ai-", " ai ", "algorithm", "graph neural", "reinforcement learning",
        "federated", "transformer", "language model", "data science",
        "recommender", "clustering", "classification", "knowledge graph",
        "semantic parsing", "program synthesis", "code generation",
        "cryptography", "blockchain", "cybersecurity", "cloud computing",
        "iot", "embedded system", "rtos", "stream processing",
        "query optimization", "database", "distributed learning",
        "contrastive learning", "self-supervised", "explainability",
        "algorithmic fairness", "autonomous vehicle", "lidar", "sensor fusion",
        "robot path", "slam", "edge ai", "data labeling", "weak supervision",
        "hyperparameter", "continual learning", "online learning",
        "anomaly detection", "intrusion detection", "network traffic",
        "adversarial", "differential privacy", "homomorphic",
        "secure computation", "formal verification", "model compression",
        "subgraph matching", "information extraction", "entity linking",
        "topic model", "streaming analytics",
    ],
    "Engineering": [
        "engineering", "turbine", "hvac", "structural", "seismic", "fatigue",
        "composite", "manufacturing", "additive manufacturing", "3d printing",
        "heat exchanger", "thermal management", "fluid mechanics", "cfd",
        "aerodynamics", "aerospace", "jet noise", "aeroacoustics", "vibration",
        "gearbox", "mems", "microfluidic", "lab-on-chip", "fiber optic",
        "structural health monitoring", "corrosion", "offshore", "civil engineering",
        "concrete", "steel frame", "geotechnical", "landslide", "pavement",
        "fire engineering", "power engineering", "smart grid", "microgrid",
        "power grid", "renewable energy", "wind turbine", "battery pack",
        "electric vehicle", "wireless power", "robotics", "manipulator",
        "uav", "drone", "mechatronics", "control system", "predictive maintenance",
        "industrial automation", "iiot", "smart manufacturing",
        "semiconductor packaging", "lithography", "nanofabrication",
        "signal integrity", "avionics", "spacecraft", "thermal barrier",
        "anti-icing", "water treatment", "filtration", "desalination",
        "multi-rotor", "quadrotor", "load balancing", "fault detection",
    ],
    "Mathematics": [
        "mathematics", "mathematical", "algebraic", "topology", "topological",
        "stochastic", "probabilistic", "probability", "statistics", "statistical",
        "optimization", "optimisation", "numerical", "differential equation",
        "partial differential", "linear algebra", "sparse matrix",
        "graph theory", "combinatorics", "discrete math", "number theory",
        "cryptanalysis", "elliptic curve", "prime field", "modular arithmetic",
        "bayesian", "mcmc", "monte carlo", "markov chain", "ergodic",
        "wavelet", "spectral method", "hilbert space", "lie algebra",
        "representation theory", "category theory", "persistent homology",
        "tda", "riemannian", "game theory", "optimal transport", "wasserstein",
        "convex", "nonlinear dynamics", "dynamical system", "chaos",
        "extreme value", "functional data", "density estimation",
        "complexity theory", "approximation algorithm", "stochastic calculus",
        "causal inference", "econometrics",
    ],
    "Physics": [
        "quantum", "physics", "condensed matter", "superconductiv",
        "topological insulator", "spin ", "magnetism", "photonic",
        "plasmon", "laser", "optics", "spectroscopy", "photoemission",
        "neutron", "nuclear reactor", "particle physics", "neutrino",
        "scintillator", "astrophysics", "black hole", "neutron star",
        "supernova", "galactic", "cosmology", "dark matter",
        "gravitational wave", "plasma", "magnetohydrodynamics",
        "solar flare", "ultracold", "bose-einstein", "optical lattice",
        "phonon", "thermal conductivity", "perovskite", "graphene",
        "weyl semimetal", "topological phase", "x-ray diffraction",
        "electron microscopy", "muon", "geophysics", "seismology",
        "string theory", "standard model", "electroweak", "lattice gauge",
        "quantum field theory", "hamiltonian", "high-energy density",
        "shock wave", "dense plasma", "binary star", "accretion disk",
        "astrophysical jet", "rotation curve", "mhd",
    ],
    "Chemistry": [
        "chemistry", "chemical", "synthesis", "catalysis", "catalyst",
        "organic synthesis", "organometallic", "cobalt catalysis",
        "cross-coupling", "hydrogenation", "oxidation", "polymerization",
        "electrochemistry", "redox", "electrocatalysis", "fuel cell",
        "electrolyte", "supercapacitor", "photocatalysis", "voc degradation",
        "dft", "ab initio", "computational chemistry", "qm/mm",
        "molecular dynamics", "halogen bonding", "radical scavenging",
        "antioxidant", "nmr", "raman", "nanoparticle", "nanomaterial",
        "metal-organic framework", "mof", "cof", "zeolite",
        "polymer", "hydrogel", "self-assembly", "drug delivery",
        "nanocarrier", "nanomedicine", "heavy metal", "pollutant",
        "water purification", "pesticide residue", "agrochemical",
        "metabolite", "natural product", "biochar", "carbon material",
        "porous carbon", "kinetics of", "thermodynamic", "phase stability",
        "organofluorine", "chirality", "asymmetric", "stereochemistry",
        "green chemistry", "biocatalysis", "enzyme mimic",
    ],
    "Biology": [
        "biology", "biological", "genomics", "genome", "gene regulation",
        "genetic", "crispr", "transcriptomics", "proteomics", "metabolomics",
        "metagenomics", "epigenetic", "chromatin", "microbiome", "microbial",
        "bacteria", "microbiology", "ecology", "ecological", "biodiversity",
        "species distribution", "evolution", "phylogenomics", "phylogenetics",
        "phylogeography", "molecular biology", "cell biology",
        "developmental biology", "stem cell", "tissue engineering",
        "protein folding", "enzyme engineering", "rna structure",
        "immunology", "immune response", "pathogen", "virus", "viral",
        "marine biology", "marine ecology", "coral reef", "phytoplankton",
        "plant biology", "plant pathogen", "crop genomics",
        "insect immunity", "amphibian", "vertebrate", "invertebrate",
        "gut microbiota", "symbiosis", "mutualism", "biofilm",
        "antibiotic resistance", "structural biology", "ribosome",
        "ribonucleoprotein", "exosome", "cell signaling",
        "autophagy", "apoptosis", "regeneration", "planarian",
        "metabolic engineering", "synthetic biology",
    ],
    "Medical Sciences": [
        "medical", "clinical", "disease", "patient", "hospital",
        "diagnosis", "diagnostic", "biomarker", "imaging",
        "cancer", "oncology", "tumor", "immunotherapy",
        "checkpoint inhibitor", "car-t", "chemotherapy", "radiotherapy",
        "diabetes", "glucose", "insulin", "cardiovascular", "cardiac",
        "stroke", "sepsis", "icu", "neurodegeneration", "alzheimer",
        "parkinson", "neurology", "eeg", "seizure", "neurosurgery",
        "retinopathy", "ophthalmology", "colonoscopy", "radiology",
        "pet imaging", "mri", "ultrasound", "ct imaging", "ecg",
        "vaccine", "epidemiology", "public health", "global health",
        "pharmacology", "pharmacokinetics", "antibiotic", "antiviral",
        "surgery", "surgical", "minimally invasive", "robotic surgery",
        "mental health", "depression", "anxiety", "psychiatric",
        "maternal health", "paediatric", "neonatal", "kidney",
        "nephrology", "rare disease", "gene therapy", "telemedicine",
        "point-of-care", "wearable sensor", "triage", "emergency care",
        "drug resistance", "drug discovery",
    ],
    "Agriculture & Environment": [
        "agriculture", "agricultural", "crop", "soil health", "farm",
        "irrigation", "drought tolerance", "water resource", "watershed",
        "agroecology", "agroforestry", "agronomy", "precision agriculture",
        "pest management", "biological control", "livestock", "grazing",
        "rangeland", "food security", "food system", "food supply",
        "carbon sequestration", "greenhouse gas", "methane emission",
        "land use", "deforestation", "desertification", "wetland",
        "ecosystem service", "biodiversity conservation",
        "satellite drought", "soil erosion", "fertilizer", "nutrient runoff",
        "pollinator", "vertical farming", "aquaponic", "greenhouse farming",
        "circular agriculture", "biomass", "bioenergy", "hydrological",
        "hydrology", "wildfire impact", "watershed protection",
        "regenerative farming", "organic farming", "sustainable farming",
        "mycorrhizal", "rhizosphere", "nitrogen fixation", "plant immunity",
        "crop rotation", "post-harvest", "soil microbiome",
    ],
    "Arts & Humanities": [
        "arts", "humanities", "literature", "literary",
        "poetry", "poetic", "narrative", "postcolonial", "diaspora",
        "cultural heritage", "heritage", "archival", "manuscript",
        "oral tradition", "ethnomusicology", "music archaeology",
        "historiography", "collective memory", "philosophy", "philosophical",
        "aesthetics", "metaphysics", "consciousness", "linguistics",
        "endangered language", "semiotics", "rhetoric", "discourse analysis",
        "visual culture", "street art", "protest art", "cinema",
        "film studies", "media studies", "satire", "archaeology",
        "ancient civilization", "symbolism", "ritual", "cultural identity",
        "museum studies", "digital heritage", "repatriation",
        "classical literature", "cultural critique", "feminism literature",
        "translation studies", "language revitalization",
    ],
    "Social Sciences": [
        "social science", "sociology", "sociological", "political science",
        "policy", "governance", "democracy", "voting behavior",
        "economics", "economic inequality", "poverty", "income mobility",
        "public policy", "welfare", "urban studies", "urban migration",
        "housing inequality", "criminology", "crime", "recidivism",
        "health disparities", "education policy", "educational equity",
        "student performance", "behavioral economics", "risk perception",
        "decision theory", "cognitive psychology", "social psychology",
        "social behavior", "conformity", "deviance", "social media addiction",
        "social network", "social capital", "social movement",
        "misinformation", "media framing", "organizational behavior",
        "management science", "innovation diffusion", "trade economics",
        "globalization", "comparative politics", "ethnography",
        "anthropology", "cross-cultural", "gender studies", "feminist",
        "caste", "rural development", "community development",
        "digital inequality", "technology adoption", "conflict",
        "displacement", "climate vulnerability",
    ],
}

def detect_domains(title: str) -> str:
    """
    Score each domain by keyword hits in the title.
    Return single domain or 2-domain cross.
    """
    t = title.lower()
    scores = {d: sum(1 for kw in kws if kw in t)
              for d, kws in DOMAIN_KEYWORDS.items()}
    ranked = sorted([(d, s) for d, s in scores.items() if s > 0],
                    key=lambda x: -x[1])
    if not ranked:
        return "Unknown"

    if len(ranked) == 1:
        return ranked[0][0]
    top_d, top_s = ranked[0]
    sec_d, sec_s = ranked[1]
    if sec_s >= 1 and sec_s >= top_s * 0.4:
        return "+".join(sorted([top_d, sec_d]))
    return top_d

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                          # 3) SLOT BANKS
                          #    Derived by abstracting noun phrases that appear repeatedly across real titles.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
SLOTS = {
    # ── Computer Science ───────────────────────
    "cs_method": [
        "Deep Learning", "Federated Learning", "Graph Neural Networks",
        "Reinforcement Learning", "Transformer Models", "Self-Supervised Learning",
        "Contrastive Learning", "Bayesian Inference", "Causal Inference",
        "Explainable AI", "Generative Adversarial Networks", "Diffusion Models",
        "Large Language Models", "Active Learning", "Meta-Learning",
        "Neural Architecture Search", "Knowledge Distillation",
        "Multimodal Learning", "Few-Shot Learning", "Zero-Shot Learning",
    ],
    "cs_task": [
        "Anomaly Detection", "Named Entity Recognition", "Link Prediction",
        "Semantic Parsing", "Image Segmentation", "Time Series Forecasting",
        "Relation Extraction", "Sentiment Classification", "Question Answering",
        "Fault Detection", "Predictive Maintenance", "Object Detection",
        "Node Classification", "Knowledge Graph Completion",
        "Sequence Labelling", "Document Classification",
    ],
    "cs_application": [
        "Intrusion Detection", "Medical Image Analysis", "Fraud Detection",
        "Crop Disease Identification", "Legal Document Summarisation",
        "Clinical Decision Support", "Autonomous Driving",
        "Sign Language Recognition", "Customer Churn Prediction",
        "Drug Repurposing", "Code Review Automation",
    ],
    "cs_data": [
        "Electronic Health Records", "Satellite Imagery", "IoT Sensor Logs",
        "Clinical Trial Data", "Remote Sensing Data",
        "Financial Transaction Logs", "Social Media Streams",
        "Scientific Literature Corpora", "Multi-Omics Data",
    ],
    "cs_sector": [
        "Rural Healthcare", "Smart Manufacturing", "Financial Services",
        "Digital Governance", "Precision Agriculture",
        "Higher Education", "Urban Traffic Management",
        "National Security", "Industrial Automation",
    ],
    "cs_system": [
        "Knowledge Extraction Pipeline", "Recommendation Engine",
        "Digital Twin Platform", "Resource Scheduling System",
        "Real-Time Monitoring System", "Intelligent Decision Support System",
    ],

    # ── Engineering ────────────────────────────
    "eng_component": [
        "Gas Turbine Blades", "Composite Rotor Blades",
        "Lithium-Ion Battery Packs", "Reinforced Concrete Beams",
        "Solar Photovoltaic Modules", "Centrifugal Pump Impellers",
        "MEMS Pressure Sensors", "Heat Pipe Arrays",
        "Bolted Flange Joints", "Thin-Walled Pressure Vessels",
    ],
    "eng_material": [
        "Carbon Fibre Reinforced Polymers", "Titanium Alloys",
        "High-Entropy Alloys", "Geopolymer Concrete",
        "Shape Memory Alloys", "Ceramic Matrix Composites",
        "Biodegradable Polymers", "Aluminium Foams",
    ],
    "eng_condition": [
        "High Cycle Fatigue", "Thermal Shock Cycling",
        "Dynamic Impact Loading", "Corrosive Marine Exposure",
        "Cryogenic Temperatures", "Elevated Humidity",
        "Oxidative Degradation",
    ],
    "eng_application": [
        "Aerospace Structures", "Offshore Wind Turbines",
        "Biomedical Implants", "Automotive Crashworthiness",
        "Railway Infrastructure", "Desalination Plants",
        "Nuclear Containment Structures",
    ],
    "eng_method": [
        "Acoustic Emission Sensing", "Finite Element Analysis",
        "Digital Image Correlation", "Ultrasonic Phased Array",
        "Machine Learning-Based Diagnostics",
        "Non-Destructive Evaluation", "Computational Fluid Dynamics",
    ],
    "eng_system": [
        "Smart Grid Energy Systems", "Structural Health Monitoring Systems",
        "Predictive Maintenance Platforms", "Autonomous Inspection Drones",
        "Hybrid Microgrids", "Water Distribution Networks",
    ],

    # ── Mathematics ────────────────────────────
    "math_method": [
        "Stochastic Partial Differential Equations",
        "Topological Data Analysis", "Spectral Methods",
        "Monte Carlo Simulation", "Bayesian Nonparametric Methods",
        "Integer Programming", "Wavelet Analysis",
        "Finite Difference Schemes", "Runge-Kutta Methods",
        "Random Matrix Theory", "Optimal Transport",
        "Algebraic Topology", "Convex Optimisation",
        "Riemannian Geometry", "Stochastic Calculus",
    ],
    "math_problem": [
        "Graph Colouring", "Inverse Problems",
        "Scheduling Under Uncertainty", "Network Flow Optimisation",
        "Eigenvalue Localisation", "Optimal Transport",
        "Combinatorial Optimisation", "High-Dimensional Inference",
    ],
    "math_phenomenon": [
        "Turbulent Flow", "Epidemic Spreading", "Market Volatility",
        "Phase Separation", "Opinion Dynamics",
        "Neural Firing Patterns", "Nonlinear Oscillations",
    ],
    "math_domain": [
        "Financial Mathematics", "Computational Biology",
        "Climate Modelling", "Operations Research",
        "Network Science", "Quantum Mechanics",
    ],
    "math_data": [
        "High-Dimensional Time Series", "Point Cloud Data",
        "Genomic Sequences", "Social Network Graphs",
        "Geophysical Signals", "Functional Neuroimaging Data",
    ],

    # ── Physics ────────────────────────────────
    "phy_material": [
        "Graphene Heterostructures", "Topological Insulators",
        "Halide Perovskites", "Weyl Semimetals",
        "Rare-Earth Manganites", "Organic Semiconductors",
        "Dilute Magnetic Semiconductors", "2D van der Waals Materials",
    ],
    "phy_phenomenon": [
        "Quantum Entanglement", "Spin Hall Effect",
        "Mott Insulator Transition", "Bose-Einstein Condensation",
        "Anderson Localisation", "Kondo Effect",
        "Superconducting Fluctuations", "Topological Phase Transitions",
    ],
    "phy_application": [
        "Quantum Computing", "Photovoltaics", "Spintronics",
        "Quantum Sensing", "Terahertz Imaging", "Nonlinear Optics",
    ],
    "phy_method": [
        "Neutron Scattering", "Scanning Tunnelling Microscopy",
        "Ultrafast Pump-Probe Spectroscopy", "Muon Spin Rotation",
        "Angle-Resolved Photoemission Spectroscopy",
        "Monte Carlo Simulation", "Ab Initio Molecular Dynamics",
    ],
    "phy_environment": [
        "Tokamak Plasma", "Interstellar Medium",
        "Ultracold Atomic Traps", "Solar Wind",
        "High-Pressure Diamond Anvil Cell",
    ],

    # ── Chemistry ──────────────────────────────
    "chem_target": [
        "Chiral Pharmaceuticals", "Biodegradable Packaging Polymers",
        "High-Energy-Density Electrolytes", "Fluorescent Organic Dyes",
        "Anti-Biofilm Coatings", "Hollow Carbon Nanospheres",
        "Enantiopure Drug Intermediates",
    ],
    "chem_application": [
        "Hydrogen Storage", "CO2 Capture", "Water Purification",
        "Drug Delivery", "Supercapacitors", "Organic Solar Cells",
        "Heterogeneous Catalysis", "Nitrogen Reduction",
    ],
    "chem_catalyst": [
        "Palladium Nanoparticles", "Single-Atom Catalysts",
        "TiO2 Photocatalysts", "Cobalt-Porphyrin Complexes",
        "Zeolite-Supported Enzymes", "Copper-Based MOFs",
    ],
    "chem_reaction": [
        "Cross-Coupling Reactions", "Hydrogenation",
        "Ring-Opening Polymerisation", "Electrochemical Reduction",
        "Oxidative Functionalisation", "Diels-Alder Cycloaddition",
        "C-H Activation", "Photocatalytic Degradation",
    ],
    "chem_material": [
        "MXene Nanosheets", "Covalent Organic Frameworks",
        "Chitosan Hydrogels", "Perovskite Quantum Dots",
        "Lignin-Derived Carbon", "Zeolitic Imidazolate Frameworks",
        "Porous Carbon Nanosheets",
    ],
    "chem_pollutant": [
        "Heavy Metal Ions", "Pharmaceuticals in Effluents",
        "Microplastics", "Persistent Organic Pollutants",
        "Nitrate Contamination", "Industrial Dye Wastewater",
    ],

    # ── Biology ────────────────────────────────
    "bio_organism": [
        "Oryza sativa", "Arabidopsis thaliana", "Zebrafish Embryos",
        "Caenorhabditis elegans", "Drosophila melanogaster",
        "Mycobacterium tuberculosis", "Streptomyces sp.",
        "Mangrove Microbiota",
    ],
    "bio_condition": [
        "Oxidative Stress", "Heat Shock", "Nutrient Starvation",
        "UV-B Radiation", "Hypoxia", "Antibiotic Exposure",
        "Salinity Stress",
    ],
    "bio_function": [
        "Cell Cycle Regulation", "Innate Immune Signalling",
        "Chromatin Remodelling", "Metabolic Reprogramming",
        "Apoptosis", "Neurogenesis", "DNA Damage Response",
    ],
    "bio_environment": [
        "Mangrove Estuaries", "Coral Reef Sediments",
        "Agricultural Rhizosphere", "Deep Subsurface Aquifers",
        "Bovine Rumen", "Urban Soil Microbiomes",
    ],
    "bio_pathogen": [
        "Plasmodium falciparum", "Dengue Virus",
        "Fusarium oxysporum", "Klebsiella pneumoniae",
        "SARS-CoV-2 Variants", "Mycobacterium tuberculosis",
    ],
    "bio_molecule": [
        "CRISPR-Associated Nucleases", "G-Quadruplex DNA",
        "Outer Membrane Vesicles", "Riboswitches",
        "Long Non-Coding RNAs", "Stress Granule Proteins",
    ],
    "bio_tissue": [
        "Neural Progenitor Cells", "Pancreatic Beta Cells",
        "Cardiac Myocytes", "Hepatocytes",
        "Retinal Ganglion Cells", "Tumour-Infiltrating Lymphocytes",
    ],
    "bio_process": [
        "Embryonic Development", "Wound Healing",
        "Tumour Microenvironment Formation", "Gut Colonisation",
        "Neurodegeneration", "Epigenetic Reprogramming",
    ],

    # ── Medical Sciences ───────────────────────
    "med_disease": [
        "Type 2 Diabetes Mellitus", "Drug-Resistant Tuberculosis",
        "Alzheimer's Disease", "Cervical Cancer",
        "Sickle Cell Disease", "Dengue Haemorrhagic Fever",
        "Non-Alcoholic Fatty Liver Disease", "Chronic Kidney Disease",
        "Postpartum Depression", "Acute Coronary Syndrome",
    ],
    "med_population": [
        "Tribal Populations", "Urban Slum Dwellers",
        "Elderly Patients", "Adolescent Girls",
        "Paediatric Cohorts", "HIV-Positive Individuals",
        "Rural Agricultural Communities", "Healthcare Workers",
    ],
    "med_setting": [
        "Primary Health Centres", "District Hospitals",
        "Community Health Programmes", "Telemedicine Platforms",
        "ICU Settings", "Emergency Departments",
    ],
    "med_drug": [
        "Metformin", "Bedaquiline", "Nivolumab",
        "mRNA Vaccine Platforms", "Monoclonal Antibodies",
        "Checkpoint Inhibitors", "CAR-T Cell Therapy",
    ],
    "med_imaging": [
        "PET Imaging", "Multiparametric MRI",
        "Cardiac Ultrasound", "Chest X-Ray",
        "Colonoscopy Imaging", "Retinal Fundus Photography",
    ],

    # ── Agriculture & Environment ───────────────
    "agr_crop": [
        "Rice", "Wheat", "Chickpea", "Pearl Millet",
        "Groundnut", "Soybean", "Sugarcane", "Tomato",
        "Finger Millet", "Banana",
    ],
    "agr_practice": [
        "Conservation Tillage", "System of Rice Intensification",
        "Integrated Nutrient Management", "Organic Farming",
        "Agroforestry", "Mulching", "Precision Irrigation",
    ],
    "agr_ecosystem": [
        "Degraded Dryland Soils", "Coastal Saline Soils",
        "Himalayan Watersheds", "Indo-Gangetic Alluvial Plains",
        "Deccan Plateau Rainfed Zones", "Western Ghats Forest Fringe",
    ],
    "agr_region": [
        "Semi-Arid Rajasthan", "Coastal Andhra Pradesh",
        "North-Eastern India", "Gangetic Plains",
        "Dryland Deccan", "Sub-Saharan Africa",
    ],
    "agr_pest": [
        "Fall Armyworm", "Brown Planthopper",
        "Helicoverpa armigera", "Stem Borer",
        "Aphids", "Whitefly",
    ],
    "agr_condition": [
        "Terminal Heat Stress", "Waterlogging",
        "Salinity Stress", "Low Phosphorus Availability",
        "Frost Exposure", "Post-Harvest Deterioration",
    ],
    "agr_nutrient": [
        "Iron", "Zinc", "Selenium", "Folate",
        "Vitamin A Precursors",
    ],

    # ── Arts & Humanities ──────────────────────
    "art_artifact": [
        "Palm-Leaf Manuscripts", "Bronze Sculptures",
        "Folk Painting Traditions", "Sanskrit Astronomical Texts",
        "Terracotta Temple Panels", "Colonial-Era Photographs",
        "Silk Weaving Traditions",
    ],
    "art_region": [
        "Coastal Tamil Nadu", "Mughal-Era Northern India",
        "Medieval Deccan Sultanates", "Himalayan Buddhist Territories",
        "Pre-Independence Bengal", "Contemporary Kerala",
        "Northeastern Tribal Belts",
    ],
    "art_community": [
        "Baul Singers", "Warli Artisans",
        "Sufi Orders of the Deccan", "Tibetan Refugee Communities",
        "Nomadic Pastoral Groups", "Devadasi Communities",
    ],
    "art_language": [
        "Gondi", "Tulu", "Maithili", "Santhali",
        "Bodo", "Kodava", "Konkani",
    ],
    "art_period": [
        "Chola-Period", "Vijayanagara-Era",
        "Early Colonial", "Post-Independence",
        "Mughal", "Contemporary",
    ],
    "art_concept": [
        "Rasa Theory", "Dharma", "Non-Self",
        "Aesthetic Emotion", "Sphota Theory of Language",
        "Karma and Agency",
    ],
    "art_tradition": [
        "Vedantic", "Buddhist", "Islamic Philosophical",
        "Jain", "Saiva Siddhanta",
    ],
    "art_text": [
        "Dalit Autobiographies", "Sangam-Era Tamil Literature",
        "Urdu Partition Narratives", "Bhakti Movement Hagiographies",
        "Colonial Gazetteers",
    ],

    # ── Social Sciences ────────────────────────
    "soc_outcome": [
        "Educational Attainment", "Food Security",
        "Social Mobility", "Political Participation",
        "Financial Inclusion", "Mental Well-Being",
        "Health-Seeking Behaviour", "Economic Empowerment",
    ],
    "soc_population": [
        "Scheduled Caste Households", "Women-Headed Households",
        "Internal Migrants", "Urban Informal Workers",
        "First-Generation Learners", "Tribal Youth",
        "Elderly Rural Communities",
    ],
    "soc_region": [
        "Rural Maharashtra", "Peri-Urban Tamil Nadu",
        "North-Eastern States", "Coal-Belt Jharkhand",
        "Coastal Odisha", "Urban Slums of Delhi",
    ],
    "soc_sector": [
        "Public Health", "Primary Education",
        "Agricultural Extension", "Microfinance",
        "Urban Housing", "Decentralised Governance",
    ],
    "soc_policy": [
        "MGNREGA", "National Food Security Act",
        "PM-KISAN", "Right to Education Act",
        "Janani Suraksha Yojana", "Pradhan Mantri Awas Yojana",
    ],
}

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                                  # 4) TEMPLATE BANK
#    Patterns extracted from real title structures.
#    Real title patterns observed:
#      "<Method> of <Topic> in <Domain>"
#      "<Topic> for <Application> in <Context>"
#      "<Method>-Based <Task> for <Application>"
#      "<Adjective> <Topic>: <Aspect> and <Aspect>"
#      "<Topic> Under <Condition>"
#      "<Method> of <Topic> Using <Tool>"
#      "Computational <Topic> of <Subject>"
#      "<Topic> and <Topic> in <Context>"
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

TEMPLATES = {

"Computer Science": [
    "{cs_method} for {cs_task} in {cs_sector}",
    "{cs_method}-Based {cs_application} Using {cs_data}",
    "Scalable {cs_method} for {cs_task} on {cs_data}",
    "{cs_method} Approaches to {cs_task} in {cs_sector}",
    "Efficient {cs_method} for Real-Time {cs_application}",
    "Privacy-Preserving {cs_method} for {cs_application} in {cs_sector}",
    "{cs_task} via {cs_method}: Scalability and Robustness",
    "Robust {cs_system} Using {cs_method} for {cs_sector}",
    "Interpretable {cs_method} for {cs_task} from {cs_data}",
    "Algorithmic Fairness in {cs_method} Applied to {cs_application}",
    "Resource-Efficient {cs_method} for {cs_task} Under Deployment Constraints",
    "{cs_method} for {cs_application}: Accuracy, Efficiency, and Explainability",
    "Secure {cs_system} Using {cs_method} for {cs_sector} Applications",
    "Adversarial Robustness of {cs_method} in {cs_application}",
    "Causal {cs_method} for {cs_task} in {cs_sector} Decision-Making",
    "{cs_method} with Limited Labelled {cs_data} for {cs_task}",
    "Multimodal {cs_method} for {cs_task} Across {cs_data}",
    "{cs_method} for Knowledge Discovery from {cs_data} in {cs_sector}",
    "Continual {cs_method} for Evolving {cs_task} in {cs_sector}",
    "Uncertainty Quantification in {cs_method} for {cs_application}",
],

"Engineering": [
    "{eng_method} for Detecting Failure in {eng_component}",
    "Topology Optimisation of {eng_component} for {eng_application}",
    "Fatigue Life Prediction of {eng_component} Under {eng_condition}",
    "Additive Manufacturing of {eng_material} for {eng_application}",
    "Thermal Management of {eng_component} in {eng_application}",
    "Structural Health Monitoring of {eng_component} via {eng_method}",
    "Non-Destructive Evaluation of {eng_material} Using {eng_method}",
    "Corrosion Behaviour of {eng_material} Under {eng_condition}",
    "Reliability Analysis of {eng_system} Under {eng_condition}",
    "Hybrid {eng_material} Composites for {eng_application}",
    "Energy Harvesting from Ambient Sources Using {eng_component}",
    "Condition Monitoring of {eng_component} Using {eng_method}",
    "Sustainable Manufacturing of {eng_component} for {eng_application}",
    "Multi-Objective Optimisation of {eng_system} for {eng_application}",
    "Vibration Damping in {eng_component} Under {eng_condition}",
    "Computational Fluid Dynamics of Nanofluids in {eng_component}",
    "Autonomous UAV Navigation in {eng_application} Environments",
    "Microfluidic {eng_component} for Biomedical {eng_application}",
    "{eng_method} for Structural Integrity Assessment of {eng_component}",
    "Low-Power VLSI Design for {eng_application}",
],

"Mathematics": [
    "{math_method} Modelling of {math_phenomenon}",
    "Graph-Theoretic Approaches to {math_problem} in {math_domain}",
    "{math_method} for {math_problem} in {math_domain}",
    "Topological Data Analysis of {math_data} in {math_domain}",
    "Numerical Stability of {math_method} for {math_problem}",
    "Probabilistic Graphical Models for {math_problem}",
    "{math_method} for High-Dimensional Inference on {math_data}",
    "Wavelet Analysis of {math_data} for Signal Denoising",
    "Mathematical Modelling of {math_phenomenon} in {math_domain}",
    "Optimal Control of {math_phenomenon} via {math_method}",
    "Algebraic Topology Methods Applied to {math_problem}",
    "Random Matrix Theory and {math_phenomenon} in {math_domain}",
    "Stochastic Calculus for {math_problem} in {math_domain}",
    "High-Dimensional {math_method} for {math_data} Analysis",
    "Discrete Mathematics and Combinatorics of {math_problem}",
    "Riemannian Geometry Methods for {math_problem} in {math_domain}",
    "Convex Optimisation Algorithms for {math_problem}",
    "Convergence Analysis of {math_method} for {math_problem}",
    "{math_method} for Uncertainty Quantification in {math_domain}",
    "Ergodic Theory and Long-Run Behaviour of {math_phenomenon}",
],

"Physics": [
    "Quantum Coherence and Decoherence in {phy_material} at Low Temperatures",
    "Topological Phases of Matter in {phy_material}",
    "Ultrafast Laser Spectroscopy of {phy_material}",
    "{phy_phenomenon} in {phy_material}: Theory and Experiment",
    "Phonon Transport and Thermal Conductivity in {phy_material}",
    "Spin-Orbit Coupling Effects in {phy_material} Heterostructures",
    "High-Pressure Behaviour of {phy_material} Using {phy_method}",
    "Optical Trapping and Manipulation of {phy_material}",
    "Exotic Superconductivity in {phy_material} Under Intense Magnetic Fields",
    "Monte Carlo Simulations of {phy_phenomenon} in {phy_material}",
    "Non-Equilibrium Dynamics of {phy_phenomenon} in {phy_material}",
    "Quantum Simulation of {phy_phenomenon} Using Ultracold Atoms",
    "{phy_phenomenon} in {phy_environment}: Modelling and Observation",
    "Condensed Matter Physics of {phy_phenomenon} in {phy_material}",
    "Photonic Band Gap Structures for {phy_application}",
    "Semiconductor Nanowires for {phy_application}",
    "Magneto-Optical Properties of {phy_material} Thin Films",
    "Plasma Instabilities and {phy_phenomenon} in {phy_environment}",
    "Nonlinear Optical Response of {phy_material} for {phy_application}",
    "Quantum Sensing Using {phy_material} for {phy_application}",
],

"Chemistry": [
    "Asymmetric Catalysis for Enantioselective Synthesis of {chem_target}",
    "{chem_material} for {chem_application}: Synthesis and Characterisation",
    "Green Synthesis of {chem_target} Using Bio-Derived Solvents",
    "Electrochemical {chem_reaction} Using {chem_catalyst} for {chem_application}",
    "Transition Metal Catalysed {chem_reaction} for {chem_target}",
    "Photocatalytic Degradation of {chem_pollutant} Using {chem_catalyst}",
    "DFT Study of {chem_reaction} Mechanisms in Organic Synthesis",
    "Enzyme-Mimicking {chem_catalyst} for {chem_reaction}",
    "Hydrothermal Synthesis of {chem_material} Nanoparticles for {chem_application}",
    "Click Chemistry Approaches for {chem_target} Functionalisation",
    "Recyclable {chem_catalyst} for Sustainable {chem_reaction}",
    "Supramolecular Self-Assembly of {chem_material} for {chem_application}",
    "Spectroscopic Kinetics of {chem_reaction} and Catalyst Deactivation",
    "Ion Transport in {chem_material} Electrolytes for {chem_application}",
    "Redox Chemistry and Electron Transfer in Metalloprotein {chem_reaction}",
    "Surface-Enhanced Raman Spectroscopy for {chem_application}",
    "Computational Chemistry of {chem_reaction} Reaction Pathways",
    "{chem_material} for Selective Detection of {chem_pollutant}",
    "Mechanochemical Synthesis of {chem_target} Without Solvents",
    "Plasma-Assisted Synthesis of {chem_material} for {chem_application}",
],

"Biology": [
    "CRISPR-Cas9 Genome Editing of {bio_organism} for Improved {bio_function}",
    "Transcriptomic Analysis of {bio_organism} Under {bio_condition}",
    "Microbiome Composition and {bio_function} in {bio_environment}",
    "Epigenetic Regulation of {bio_function} in {bio_organism}",
    "Single-Cell RNA Sequencing of {bio_tissue} During {bio_process}",
    "Metagenomic Profiling of {bio_environment} Microbial Communities",
    "Adaptive Immune Response to {bio_pathogen} in {bio_organism}",
    "Structural Biology of {bio_molecule} Involved in {bio_function}",
    "Biodiversity Assessment of {bio_organism} in {bio_environment}",
    "Autophagy Pathways and {bio_function} Under {bio_condition}",
    "Metabolic Flux Analysis in {bio_organism} Under {bio_condition}",
    "Biofilm Formation and Antibiotic Resistance in {bio_pathogen}",
    "Evolutionary Genomics of {bio_organism} Across {bio_environment}",
    "Plant-Pathogen Interactions: {bio_organism} Resistance to {bio_pathogen}",
    "Comparative Genomics of Virulence Genes in {bio_pathogen}",
    "{bio_molecule} Roles in {bio_function} of {bio_tissue}",
    "Developmental Biology of {bio_process} in {bio_organism}",
    "Molecular Biology of {bio_function} Under {bio_condition}",
    "Horizontal Gene Transfer and {bio_function} in {bio_organism}",
    "Ecological Niche Modelling of {bio_organism} in {bio_environment}",
],

"Medical Sciences": [
    "Biomarkers for Early Diagnosis of {med_disease} in {med_population}",
    "{med_drug} Efficacy in {med_disease}: A Multi-Centre Clinical Trial",
    "Pharmacogenomic Determinants of {med_drug} Response in {med_population}",
    "Gut Microbiome Alterations in {med_disease} and Therapeutic Implications",
    "Telemedicine Interventions for {med_disease} Management in {med_setting}",
    "Nanoparticle Drug Delivery for {med_disease} Treatment",
    "Epidemiology of {med_disease} in {med_population}: Trends and Risk Factors",
    "Mental Health Burden of {med_disease} in {med_population}",
    "Point-of-Care Diagnostics for {med_disease} in {med_setting}",
    "AI-Assisted {med_imaging} Interpretation for {med_disease} Diagnosis",
    "Longitudinal Cohort Study of {med_disease} Progression in {med_population}",
    "Neuroimaging Correlates of {med_disease} in {med_population}",
    "Clinical Prediction Models for {med_disease} Risk in {med_population}",
    "Genomic Profiling for Personalised {med_drug} Therapy in {med_disease}",
    "Immunotherapy and Immune Evasion in {med_disease} Tumour Microenvironment",
    "Digital Therapeutics for {med_disease} Management in {med_population}",
    "Antimicrobial Stewardship to Combat Drug Resistance in {med_setting}",
    "Surgical Innovation in Minimally Invasive Procedures: Outcomes and Safety",
    "Wearable Sensor Monitoring of {med_disease} in {med_population}",
    "Gene Therapy Approaches for Monogenic {med_disease}",
],

"Agriculture & Environment": [
    "Drought Tolerance Mechanisms in {agr_crop} Under {agr_condition}",
    "Soil Carbon Sequestration Under {agr_practice} in {agr_ecosystem}",
    "Integrated Pest Management for {agr_pest} in {agr_crop} Systems",
    "Remote Sensing-Based {agr_crop} Yield Estimation in {agr_region}",
    "Climate-Resilient {agr_crop} Varieties for {agr_region}",
    "Wastewater Reuse for {agr_crop} Irrigation: Risks and Mitigation",
    "Agroforestry Models for {agr_ecosystem} Restoration in {agr_region}",
    "Nitrogen Use Efficiency in {agr_crop} Under {agr_practice}",
    "Invasive Species Impact on {agr_ecosystem} Biodiversity in {agr_region}",
    "Livestock Methane Emissions and Mitigation Strategies in {agr_region}",
    "Mycorrhizal Networks and Nutrient Cycling in {agr_ecosystem}",
    "Seed Priming Techniques for {agr_crop} Germination Under {agr_condition}",
    "Groundwater Depletion and Sustainable {agr_crop} Farming in {agr_region}",
    "Post-Harvest Loss Reduction in {agr_crop} Agricultural Supply Chains",
    "Carbon Footprint of {agr_practice} Farming Systems in {agr_region}",
    "Biofortification of {agr_crop} with {agr_nutrient} for {agr_region}",
    "Heavy Metal Phytoremediation by {agr_crop} in {agr_ecosystem}",
    "Pollinator Habitat Conservation for {agr_crop} Productivity in {agr_region}",
    "Precision Irrigation for {agr_crop} Using IoT Soil Moisture Sensors",
    "Soil Erosion Prediction Models Using Geospatial Machine Learning in {agr_ecosystem}",
],

"Arts & Humanities": [
    "Digital Archiving of {art_artifact} from {art_region}: Methodology and Access",
    "Oral Traditions of {art_community}: Documentation and Language Revitalisation",
    "Postcolonial Readings of Literature in {art_region}",
    "Linguistic Typology of {art_language} in Endangered Language Contexts",
    "Material Culture of {art_community}: {art_artifact} and Cultural Identity",
    "Iconography of {art_period} Temple Architecture in {art_region}",
    "Memory, Trauma, and Narrative in {art_region} Partition Literature",
    "Philosophical Dimensions of {art_concept} in Classical {art_tradition}",
    "Manuscript Studies: {art_artifact} Preservation and Digital Edition",
    "Ethnomusicology of {art_community} Ritual Music in {art_region}",
    "Folklore and Social Memory in {art_community} Oral Epics",
    "Aesthetic Theory of {art_concept} in {art_tradition} Philosophy",
    "Colonial-Era Cartography and its Legacy in {art_region}",
    "Gender Representations in {art_period} Visual Culture of {art_region}",
    "Translation Studies in {art_language}: Theory and Practice",
    "Semiotics of {art_artifact} in Contemporary {art_region} Media",
    "Historiography of Collective Memory and National Identity in {art_region}",
    "Repatriation of Indigenous Archives and Digital Heritage in {art_region}",
    "Digital Humanities Approaches to {art_artifact} Corpus Analysis",
    "Philosophical Analysis of {art_concept} in {art_tradition} Thought",
],

"Social Sciences": [
    "Socioeconomic Determinants of {soc_outcome} Among {soc_population} in {soc_region}",
    "Political Economy of {soc_sector} Reform in {soc_region}",
    "Gender, Labour, and {soc_outcome} in {soc_sector} of {soc_region}",
    "Impact Evaluation of {soc_policy} on {soc_outcome} in {soc_region}",
    "Urbanisation, Migration, and {soc_outcome} in {soc_region}",
    "Digital Divide and {soc_outcome} Among {soc_population} in {soc_region}",
    "Social Movements and {soc_policy} Change in {soc_region}",
    "Mental Health, Stigma, and Help-Seeking Among {soc_population} in {soc_region}",
    "Technology Adoption and {soc_outcome} Among {soc_population}",
    "Community-Based {soc_policy} Interventions for {soc_outcome}",
    "Conflict, Displacement, and {soc_outcome} of {soc_population} in {soc_region}",
    "Media Framing of {soc_sector} Policy and Public Perception in {soc_region}",
    "Electoral Behaviour and {soc_outcome} in {soc_region} Local Governance",
    "Informal Economy and {soc_outcome} Among {soc_population} in {soc_region}",
    "Social Capital and {soc_outcome} in {soc_population} Communities",
    "Household Poverty Dynamics and {soc_policy} Effectiveness in {soc_region}",
    "Caste, Class, and Access to {soc_sector} in {soc_region}",
    "Behavioural Economics and {soc_outcome} Decision-Making Among {soc_population}",
    "Criminology of Recidivism and Criminal Justice Reform in {soc_region}",
    "Computational Social Science of Information Diffusion in {soc_region}",
],

# ── 2-domain cross templates ────────────────────────────────────────────
# Each template deliberately includes keywords from BOTH domains
# so detect_domains() will always assign the correct cross-domain pair.

"Computer Science+Engineering": [
    "Deep Learning for Predictive Maintenance of {eng_component} in {eng_application}",
    "Digital Twin-Based Optimisation of {eng_system} Using Reinforcement Learning",
    "Anomaly Detection in {eng_component} IoT Sensor Streams via Transformer Models",
    "AI-Driven Topology Optimisation of {eng_material} Structures",
    "Computer Vision-Based Structural Damage Assessment of {eng_component}",
    "Graph Neural Networks for Fault Diagnosis in {eng_system}",
    "Federated Learning for Condition Monitoring of {eng_component} Industrial Fleets",
    "Neural Architecture Search for Energy-Efficient {eng_application} Systems",
    "Reinforcement Learning Control of {eng_system} Under {eng_condition}",
    "Self-Supervised Learning for {eng_method} in {eng_application}",
],
"Computer Science+Medical Sciences": [
    "Deep Learning for {med_disease} Detection from {med_imaging}",
    "Natural Language Processing of Clinical Notes for {med_disease} in {med_population}",
    "Federated Learning for Privacy-Preserving {med_disease} Prediction in {med_setting}",
    "Explainable AI for {med_disease} Prognosis in {med_population}",
    "Graph Neural Network Drug Repurposing for {med_disease}",
    "Transformer Models for {med_disease} Risk Stratification from {cs_data}",
    "AI-Assisted {med_imaging} Interpretation for {med_disease} Diagnosis",
    "Machine Learning Predictors for {med_disease} in {med_setting}",
],
"Computer Science+Mathematics": [
    "Machine Learning-Augmented {math_method} for High-Dimensional {math_data}",
    "Graph Neural Networks for {math_problem} on Mathematical Structures",
    "Probabilistic Programming and Bayesian Inference for {math_problem}",
    "Neural Solvers for Stochastic Differential Equations in {math_domain}",
    "Deep Learning Solutions to {math_problem} in {math_domain}",
    "Scalable {cs_method} for {math_problem} in {math_domain}",
],
"Computer Science+Social Sciences": [
    "Computational Modelling of {soc_outcome} Dynamics Using {cs_data}",
    "NLP Analysis of {soc_policy} Policy Discourse in {soc_region}",
    "Agent-Based Simulation of {soc_outcome} Under {soc_policy} Interventions",
    "Machine Learning for Targeting {soc_policy} Benefits to {soc_population}",
    "Network Analysis of {soc_sector} Information Flows in {soc_region}",
    "Social Media Analytics of Political Behaviour and {soc_outcome} in {soc_region}",
],
"Computer Science+Agriculture & Environment": [
    "Machine Learning for {agr_crop} Disease Detection from Satellite Imagery in {agr_region}",
    "Deep Learning-Based Soil Health Mapping in {agr_ecosystem}",
    "IoT and AI Platform for Real-Time {agr_crop} Yield Prediction in {agr_region}",
    "Graph Neural Networks for {agr_pest} Spread Modelling in {agr_ecosystem}",
    "Federated Learning for Privacy-Preserving Agricultural Advisory in {agr_region}",
    "AI-Driven Forecasting Tools for Managing Agricultural Water Resources in {agr_region}",
],
"Engineering+Physics": [
    "Plasma-Assisted Deposition of {phy_material} Coatings for {eng_application}",
    "Quantum Sensing Using {phy_material} for {eng_application}",
    "Non-Destructive Evaluation of {eng_material} Using Physics-Based {phy_method}",
    "Phonon Engineering in {phy_material} for Thermal Management of {eng_component}",
    "Magnetohydrodynamic Modelling of Conducting Fluids in {eng_component}",
    "Laser-Based {phy_method} for Structural Analysis of {eng_material}",
],
"Engineering+Medical Sciences": [
    "3D-Printed {eng_material} Scaffolds for Tissue Engineering Applications",
    "Wearable {eng_component} Sensors for Continuous {med_disease} Monitoring in {med_population}",
    "Microfluidic Lab-on-Chip Engineering for Point-of-Care {med_disease} Diagnosis",
    "Robotic-Assisted Minimally Invasive Surgical Engineering: Outcomes and Design",
    "Biomechanical Engineering Analysis of {eng_component} Implants for {med_disease}",
    "Control Engineering for Surgical Robotics in {med_setting}",
],
"Engineering+Agriculture & Environment": [
    "Solar-Powered Drip Irrigation Engineering for {agr_crop} in {agr_region}",
    "UAV-Based Multispectral Imaging Engineering for {agr_crop} Stress Detection",
    "IoT-Enabled Soil Moisture Engineering for {agr_crop} Under {agr_condition}",
    "Biogas Plant Engineering Optimisation for {agr_crop} Waste in {agr_region}",
    "Structural Engineering of Low-Cost Cold Storage for {agr_crop} in {agr_region}",
    "Precision Engineering Systems for Sustainable Water Use in {agr_ecosystem}",
],
"Engineering+Mathematics": [
    "Mathematical Optimisation of {eng_system} Under {eng_condition}",
    "Stochastic Modelling of {math_phenomenon} in {eng_component} Failure",
    "Topology Optimisation of {eng_component} Using Numerical {math_method}",
    "Reliability Modelling of {eng_system} Using {math_method}",
    "Finite Element Analysis of {eng_component} via Mathematical {math_method}",
],
"Mathematics+Physics": [
    "Spectral Theory of {phy_phenomenon} in Disordered {phy_material}",
    "Topological Mathematical Methods in {phy_phenomenon} Classification",
    "Random Matrix Models of {phy_phenomenon} in {phy_material}",
    "Partial Differential Equations of {phy_phenomenon} in {phy_environment}",
    "Geometric Phases and {phy_phenomenon} in {phy_material} Systems",
    "Stochastic Differential Equations for {phy_phenomenon} in Physics",
],
"Mathematics+Social Sciences": [
    "Game-Theoretic Analysis of {soc_policy} Incentives for {soc_population}",
    "Network Mathematical Models of {soc_outcome} Diffusion in {soc_region}",
    "Causal Inference Statistical Methods for {soc_policy} Impact Evaluation",
    "Optimal Mechanism Design for {soc_sector} Resource Allocation",
    "Spatial Econometric Mathematical Models of {soc_outcome} in {soc_region}",
    "Stochastic Modelling of Poverty Dynamics and {soc_outcome} in {soc_region}",
],
"Physics+Chemistry": [
    "Ultrafast Spectroscopy of {chem_reaction} Dynamics in {phy_material}",
    "Quantum Chemical Modelling of {chem_catalyst} Active Sites",
    "Photophysics of {chem_material} for {phy_application}",
    "Electrochemical Impedance of {phy_material} Electrolyte Interfaces",
    "In-Situ X-Ray Characterisation of {chem_reaction} in {phy_material}",
    "DFT and Physics-Based Modelling of {chem_reaction} in {phy_material}",
],
"Chemistry+Biology": [
    "Chemical Ecology of {bio_organism}-{bio_pathogen} Interactions",
    "Metabolomics of {bio_organism} Under {bio_condition} via Chemical Profiling",
    "Small Molecule Inhibitors of {bio_molecule} in {bio_organism}",
    "Lipid Nanoparticle Delivery of {bio_molecule} to {bio_tissue}",
    "Synthetic Biology of Enzyme-Inspired Chemistry in {bio_organism}",
    "Biochemical Analysis of {bio_molecule} Involved in {bio_function}",
],
"Chemistry+Engineering": [
    "Electrochemical {chem_reaction} in {eng_component} for {chem_application}",
    "Surface Functionalisation of {eng_material} Using {chem_catalyst}",
    "Corrosion Inhibition of {eng_material} by {chem_material} Chemical Coatings",
    "Scale-Up of {chem_reaction} in Continuous Flow Engineering {eng_component}",
    "Chemical Vapour Deposition of {chem_material} on {eng_component}",
    "Green Chemical Engineering of {chem_reaction} for {chem_application}",
],
"Biology+Medical Sciences": [
    "Microbiome Dysbiosis in {med_disease}: Biological Insights from {bio_organism}",
    "CRISPR Screening for {med_disease} Drug Targets in {bio_tissue}",
    "Single-Cell Transcriptomics of {bio_tissue} in {med_disease} Progression",
    "Innate Immunity Pathways in {bio_organism} as Models for {med_disease}",
    "Metabolomics of {bio_tissue} in {med_disease} Across {med_population}",
    "Genomics and Biomarkers of Drug Resistance in {med_disease}",
],
"Biology+Agriculture & Environment": [
    "Rhizosphere Microbiome Biology for {agr_crop} Productivity in {agr_ecosystem}",
    "Plant-Growth-Promoting Bacteria for {agr_crop} Under {agr_condition}",
    "Genomic Selection for {agr_crop} Agricultural Breeding Programmes",
    "Biodiversity and Ecosystem Function in {agr_ecosystem}: {bio_organism} as Indicators",
    "Epigenetic Adaptation of {agr_crop} to {agr_condition} in {agr_ecosystem}",
    "Molecular Biology of {agr_crop} Resistance to {bio_pathogen}",
],
"Medical Sciences+Social Sciences": [
    "Social Determinants of {med_disease} Among {soc_population} in {soc_region}",
    "Community-Based Mental Health Intervention for {med_disease} in {soc_region}",
    "Health System Responsiveness to {med_disease} Needs of {soc_population}",
    "Qualitative Study of {soc_policy} Implementation for {med_disease} Control",
    "Epidemiology and Social Science of {med_disease} Burden in {soc_region}",
    "Public Health Disparities and {soc_outcome} in {soc_region}",
],
"Agriculture & Environment+Social Sciences": [
    "Farmer Decision-Making Under {agr_condition} in {soc_region}",
    "Gender and Social Access to {agr_practice} Extension Services in {soc_region}",
    "Political Economy of {agr_crop} Procurement Policy in {soc_region}",
    "Community Collective Action for {agr_ecosystem} Restoration in {soc_region}",
    "Climate Vulnerability and {soc_outcome} of {soc_population} in {agr_region}",
    "Policy Frameworks for Agricultural Climate Adaptation in {soc_region}",
],
"Arts & Humanities+Social Sciences": [
    "Cultural Memory, Identity, and {soc_outcome} of {art_community} in {soc_region}",
    "Narrative Framing of {soc_policy} in {art_region} Regional Press",
    "Oral Histories and Social Access to {soc_sector} Among {soc_population}",
    "Language, Power, and {soc_outcome} Among {art_community} in {soc_region}",
    "Ethnographic Study of {art_community} Livelihoods and {soc_sector} in {soc_region}",
    "Semiotics of Cultural Identity and {soc_outcome} in {art_region}",
],
}

def fill_slots(template: str) -> str:
    def repl(m):
        key = m.group(1)
        return random.choice(SLOTS.get(key, [key]))
    return re.sub(r"\{(\w+)\}", repl, template)

def generate_title(domain: str) -> str:
    bank = TEMPLATES.get(domain)
    if not bank:
        parts = domain.split("+")
        if len(parts) == 2:
            bank = TEMPLATES.get(parts[1] + "+" + parts[0])
    if not bank:
        bank = TEMPLATES["Computer Science"]
    return fill_slots(random.choice(bank))

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                          # 5) DATE & STATUS HELPERS
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
def random_date(start_year=2018, end_year=2024) -> str:
    y = random.randint(start_year, end_year)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return f"{d:02d}/{m:02d}/{y}"

def make_end_date(start_str: str, proj_type: str) -> str:
    try:
        d, m, y = map(int, start_str.split("/"))
    except Exception:
        return random_date(2021, 2027)
    duration = {"Internal": random.randint(1, 2),
                "External":  random.randint(2, 4),
                "Consultancy": random.randint(1, 3)}.get(proj_type, 2)
    return f"{d:02d}/{m:02d}/{y + duration}"

def make_status(start_str: str, end_str: str) -> str:
    try:
        sy = int(start_str.split("/")[2])
        ey = int(end_str.split("/")[2])
    except Exception:
        return "Datr error"
    if ey < 2024:
        return random.choice(["Completed", "Completed", "Completed", "Terminated"])
    elif sy > 2024:
        return "Yet to Start"
    return random.choice(["Ongoing", "Ongoing", "Completed"])

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                               # 6) DOMAIN POOL (weighted)
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
SINGLE_DOMAINS = list(DOMAIN_KEYWORDS.keys())
CROSS_PAIRS    = [k for k in TEMPLATES if "+" in k]
DOMAIN_POOL    = SINGLE_DOMAINS * 50 + CROSS_PAIRS * 38

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                              # 7) READ RESEARCHERS 
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
print("Reading researcher profiles ...")
df_res = pd.read_csv(RESEARCHERS_CSV)
res_domain_col = next((c for c in df_res.columns if "domain" in c.lower()),
                      df_res.columns[1])
researcher_domains = sorted(
    set(df_res[res_domain_col].dropna().astype(str).str.strip()))
print(f"  Researcher domains: {researcher_domains}")

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                                # 8) GENERATE
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
print(f"Generating {TARGET_PROJECTS} projects from templates ...")
rows        = []
seen_titles : set = set()
max_attempts = TARGET_PROJECTS * 12
attempt      = 0

while len(rows) < TARGET_PROJECTS and attempt < max_attempts:
    attempt += 1
    domain  = random.choice(DOMAIN_POOL)
    title   = generate_title(domain)

    if title in seen_titles:
        continue

    # Confirm detector agrees with at least one part of intended domain
    detected      = detect_domains(title)
    domain_parts  = set(domain.split("+"))
    detected_parts= set(detected.split("+"))
    if not domain_parts.intersection(detected_parts):
        continue   # slot fill produced a domain-ambiguous title — skip

    seen_titles.add(title)
    proj_type = random.choices(PROJECT_TYPES, weights=TYPE_WEIGHTS, k=1)[0]
    start     = random_date()
    end       = make_end_date(start, proj_type)
    pid       = f"PROJ-{str(len(rows)+1).zfill(5)}"

    rows.append({
        "Project_ID":    pid,
        "Grant_ID":      "",
        "Project_Title": title,
        "Domain":        detected,
        "Project_Type":  proj_type,
        "Start_Date":    start,
        "End_Date":      end,
        "Status":        make_status(start, end),
    })

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                                # 9) SAVE
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
df_proj = pd.DataFrame(rows)
df_proj.to_csv(OUT_PROJECTS_CSV, index=False)
print(f"Projects file without grant id linkage has been generated")

