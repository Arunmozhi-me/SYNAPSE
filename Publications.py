"""
generate_publications.py
────────────────────────
Reads: # Datasets generated from Researcher Profiles, Projects and Grants python file
  • Projects.csv   (PI, CoPI, Grant_ID, Domain, dates)
  • Grants.csv          (optional enrichment)
  • Researchers_Profile.csv   (optional author validation)

Outputs:
  • Publications.csv

Title generation strategy
─────────────────────────
The project title is treated as a structured sentence, e.g.:
  "Federated Learning for Privacy-Preserving Anomaly Detection in Rural Healthcare"

Each publication generated from it zooms into ONE meaningful aspect:
  Pub 1 → "Privacy-Preserving Anomaly Detection: A Federated Learning Approach"
  Pub 2 → "Scalable Federated Learning in Rural Healthcare: Empirical Evaluation"
  Pub 3 → "Robustness of Anomaly Detection Models Under Data Heterogeneity"

The noun phrases and method terms are parsed directly from the project title — not invented — so every pub title is semantically related to its project.

Domain matching
───────────────
For cross-domain projects (e.g. "Computer Science+Medical Sciences") each
publication independently picks the domain whose content (journal, keywords,
IF, citations) best matches the terms in the generated title.
"""

import pandas as pd
import numpy as np
import random
import re

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                                # 1) PATHS
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
PROJECTS_CSV = "Projects.csv"
GRANTS_CSV   = "Grants.csv"
PROFILE_CSV  = "researchers_profile.csv"
OUT_PUB_CSV  = "Publications.csv"

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                                # 2) SETTINGS
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MIN_PUB_PER_PROJECT = 1
MAX_PUB_PER_PROJECT = 4
PUB_PREFIX          = "PUB-"
DOI_PREFIX          = "10.12345/"
DEFAULT_YEAR_MIN    = 2016
DEFAULT_YEAR_MAX    = 2025

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                              # 3) HELPERS
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
def norm_str(x) -> str:
    return "" if pd.isna(x) else str(x).strip()

def normalize_rid(x):
    s = norm_str(x).upper()
    if s in {"", "NAN", "NONE"}:
        return None
    m = re.match(r"^R(\d+)$", s)
    if m:
        num = m.group(1)
        return "R" + (num.zfill(3) if len(num) <= 3 else num)
    return s

def split_ids(cell) -> list:
    s = norm_str(cell)
    if not s:
        return []
    out, seen = [], set()
    for p in re.split(r"[;,|\s]+", s):
        rid = normalize_rid(p)
        if rid and rid not in seen:
            out.append(rid); seen.add(rid)
    return out

def split_domains(cell: str) -> list:
    s = norm_str(cell)
    if not s:
        return []
    seen, out = set(), []
    for p in re.split(r"\+|,|;|\|", s):
        p = p.strip()
        if p and p not in seen:
            out.append(p); seen.add(p)
    return out

def parse_year(x) -> int | None:
    m = re.search(r"\b(\d{4})\b", norm_str(x))
    if m:
        y = int(m.group(1))
        if 1900 <= y <= 2100:
            return y
    return None

def make_pub_id(i: int) -> str:
    return f"{PUB_PREFIX}{str(i).zfill(6)}"

def make_doi(pub_id: str) -> str:
    return DOI_PREFIX + pub_id.replace("-", "").lower()

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                  # 4) DOMAIN CONTENT BANKS
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
DOMAIN_CONTENT = {

"Computer Science": {
    "journals": [
        "IEEE Transactions on Neural Networks and Learning Systems",
        "ACM Transactions on Intelligent Systems and Technology",
        "Journal of Machine Learning Research",
        "Pattern Recognition",
        "Neural Networks",
        "Expert Systems with Applications",
        "Information Sciences",
        "Knowledge-Based Systems",
        "Neurocomputing",
        "Computers and Security",
        "Future Generation Computer Systems",
        "IEEE Internet of Things Journal",
        "Journal of Parallel and Distributed Computing",
        "ACM Computing Surveys",
        "Artificial Intelligence",
    ],
    "keywords": [
        "deep learning; neural networks; classification; feature extraction",
        "federated learning; privacy; distributed optimization; edge computing",
        "graph neural networks; knowledge graph; link prediction; embedding",
        "natural language processing; transformer; attention mechanism; BERT",
        "reinforcement learning; policy gradient; reward shaping; agent",
        "anomaly detection; intrusion detection; cybersecurity; network traffic",
        "explainable AI; interpretability; fairness; bias mitigation",
        "computer vision; object detection; image segmentation; CNN",
        "IoT; sensor fusion; real-time processing; embedded systems",
        "cloud computing; microservices; load balancing; scalability",
    ],
    "if_range":   (2.5, 12.0),
    "cite_range": (5, 180),
    "suffixes": [
        ": A Comparative Study",
        ": An Empirical Evaluation",
        " Using Benchmark Datasets",
        ": Scalability and Performance Analysis",
        ": Design, Implementation, and Evaluation",
        " with Application to Real-World Scenarios",
    ],
},

"Engineering": {
    "journals": [
        "Engineering Structures",
        "International Journal of Fatigue",
        "Composite Structures",
        "Applied Energy",
        "Mechanical Systems and Signal Processing",
        "Journal of Manufacturing Processes",
        "NDT and E International",
        "Structural Health Monitoring",
        "Renewable Energy",
        "IEEE Transactions on Industrial Electronics",
        "Automation in Construction",
        "Journal of Sound and Vibration",
        "Materials and Design",
        "Computers in Industry",
        "Energy and Buildings",
    ],
    "keywords": [
        "structural health monitoring; fatigue; composite; finite element analysis",
        "additive manufacturing; topology optimization; lightweight design",
        "thermal management; heat transfer; nanofluids; energy efficiency",
        "smart grid; renewable energy; power electronics; microgrid",
        "vibration analysis; signal processing; condition monitoring; fault diagnosis",
        "corrosion; surface engineering; material degradation; coatings",
        "robotics; control systems; autonomous navigation; UAV",
        "seismic performance; steel structures; earthquake engineering",
        "MEMS; microfluidics; lab-on-chip; biosensing",
        "industrial automation; predictive maintenance; IIoT; digital twin",
    ],
    "if_range":   (2.0, 9.0),
    "cite_range": (4, 120),
    "suffixes": [
        ": Experimental Validation",
        ": A Case Study",
        " Under Realistic Operating Conditions",
        ": Modelling and Simulation",
        ": Design and Testing",
        " for Structural Applications",
    ],
},

"Mathematics": {
    "journals": [
        "Journal of Mathematical Analysis and Applications",
        "Applied Mathematics and Computation",
        "Linear Algebra and its Applications",
        "Journal of Computational and Applied Mathematics",
        "Stochastic Processes and their Applications",
        "Topology and its Applications",
        "Discrete Mathematics",
        "Annals of Applied Probability",
        "SIAM Journal on Numerical Analysis",
        "Journal of Differential Equations",
        "Mathematics of Operations Research",
        "Journal of Combinatorial Theory",
        "Advances in Mathematics",
        "Probability Theory and Related Fields",
        "Operations Research",
    ],
    "keywords": [
        "stochastic differential equations; Markov chains; convergence; stability",
        "topological data analysis; persistent homology; algebraic topology",
        "graph theory; combinatorial optimization; network flows",
        "numerical methods; finite element; convergence; error bounds",
        "Bayesian inference; MCMC; probabilistic models; posterior estimation",
        "game theory; mechanism design; Nash equilibrium; optimization",
        "linear algebra; sparse matrices; eigenvalue; spectral methods",
        "optimal transport; Wasserstein distance; measure theory",
        "stochastic calculus; Ito integral; martingales; derivative pricing",
        "dynamical systems; chaos; bifurcation; nonlinear stability",
    ],
    "if_range":   (1.2, 5.0),
    "cite_range": (2, 80),
    "suffixes": [
        ": Existence, Uniqueness, and Stability",
        ": A Rigorous Analysis",
        " in the Finite-Dimensional Setting",
        ": Bounds and Approximations",
        ": Theory and Algorithms",
        " Under General Conditions",
    ],
},

"Physics": {
    "journals": [
        "Physical Review Letters",
        "Physical Review B",
        "Nature Physics",
        "Journal of Physics: Condensed Matter",
        "Applied Physics Letters",
        "Superconductor Science and Technology",
        "Physical Review Applied",
        "New Journal of Physics",
        "Nuclear Physics B",
        "Astrophysical Journal",
        "Monthly Notices of the Royal Astronomical Society",
        "Physics of Plasmas",
        "Physical Review D",
        "Journal of High Energy Physics",
        "Reviews of Modern Physics",
    ],
    "keywords": [
        "topological insulators; quantum transport; condensed matter; spin-orbit coupling",
        "superconductivity; Cooper pairs; BCS theory; phase transition",
        "quantum entanglement; decoherence; quantum information; qubit",
        "ultrafast spectroscopy; pump-probe; electron dynamics; femtosecond",
        "plasma physics; magnetohydrodynamics; instabilities; tokamak",
        "neutron scattering; crystal structure; phonons; lattice dynamics",
        "astrophysics; black holes; neutron stars; accretion disk",
        "dark matter; cosmological simulations; N-body; halo",
        "photonics; optical trapping; laser cooling; Bose-Einstein condensate",
        "perovskites; thin films; optical properties; bandgap",
    ],
    "if_range":   (3.0, 20.0),
    "cite_range": (10, 300),
    "suffixes": [
        ": Ab Initio Study",
        ": Experimental Evidence",
        " at Low Temperatures",
        " Under High Pressure",
        ": Monte Carlo Investigation",
        ": Theory and Experiment",
    ],
},

"Chemistry": {
    "journals": [
        "Journal of the American Chemical Society",
        "Angewandte Chemie International Edition",
        "Chemical Science",
        "ACS Catalysis",
        "Green Chemistry",
        "Environmental Science and Technology",
        "Journal of Catalysis",
        "Chemical Engineering Journal",
        "Electrochimica Acta",
        "Dalton Transactions",
        "RSC Advances",
        "Applied Catalysis B: Environmental",
        "Polymer Chemistry",
        "Reaction Chemistry and Engineering",
        "ACS Applied Materials and Interfaces",
    ],
    "keywords": [
        "asymmetric catalysis; enantioselectivity; chiral synthesis; transition metal",
        "metal-organic frameworks; porous materials; gas adsorption; CO2 capture",
        "photocatalysis; reactive oxygen species; degradation; TiO2",
        "electrochemistry; redox; cyclic voltammetry; electron transfer",
        "green chemistry; solvent-free; biocatalysis; sustainable synthesis",
        "DFT; computational chemistry; reaction mechanism; energy barrier",
        "nanoparticles; surface chemistry; catalytic activity; size effect",
        "polymer; hydrogel; self-assembly; noncovalent interactions",
        "drug delivery; nanocarrier; release kinetics; biocompatibility",
        "heavy metals; water purification; adsorption; remediation",
    ],
    "if_range":   (3.5, 18.0),
    "cite_range": (8, 250),
    "suffixes": [
        ": Synthesis, Characterisation, and Application",
        ": Kinetics and Mechanism",
        " in Aqueous Medium",
        ": A Green Chemistry Approach",
        ": Spectroscopic Characterisation",
        ": Computational and Experimental Study",
    ],
},

"Biology": {
    "journals": [
        "PLOS Biology",
        "Molecular Cell",
        "Cell Reports",
        "PLOS ONE",
        "Genome Biology",
        "Molecular Biology and Evolution",
        "Current Biology",
        "Journal of Molecular Biology",
        "Ecology Letters",
        "Molecular Ecology",
        "BMC Genomics",
        "Journal of Experimental Biology",
        "Frontiers in Microbiology",
        "Microbiology Spectrum",
        "Plant and Cell Physiology",
    ],
    "keywords": [
        "CRISPR; gene editing; genome; off-target effects; functional genomics",
        "transcriptomics; RNA-seq; differential expression; single-cell",
        "microbiome; 16S rRNA; metagenomics; gut bacteria; diversity",
        "epigenetics; chromatin; DNA methylation; histone modification",
        "protein folding; structural biology; cryo-EM; molecular dynamics",
        "apoptosis; autophagy; cell signalling; kinase; caspase",
        "immunology; T-cell; cytokines; innate immunity; antigen",
        "ecology; biodiversity; species distribution; habitat fragmentation",
        "evolutionary genomics; phylogenetics; selection; adaptation",
        "plant pathogen; resistance; effectors; host-pathogen interaction",
    ],
    "if_range":   (3.0, 14.0),
    "cite_range": (8, 200),
    "suffixes": [
        ": Mechanistic Insights",
        " in Model Organisms",
        ": A Genome-Wide Study",
        ": Functional Consequences",
        " During Development and Disease",
        ": Transcriptomic Evidence",
    ],
},

"Medical Sciences": {
    "journals": [
        "The Lancet",
        "JAMA Internal Medicine",
        "British Medical Journal",
        "Journal of Clinical Oncology",
        "Annals of Internal Medicine",
        "PLOS Medicine",
        "Journal of the American College of Cardiology",
        "Diabetes Care",
        "Neurology",
        "Journal of Infectious Diseases",
        "American Journal of Respiratory and Critical Care Medicine",
        "BMC Medicine",
        "Clinical Infectious Diseases",
        "International Journal of Epidemiology",
        "Radiology",
    ],
    "keywords": [
        "clinical trial; randomised controlled; efficacy; adverse events; placebo",
        "biomarkers; early detection; sensitivity; specificity; AUC",
        "immunotherapy; checkpoint inhibitor; PD-L1; tumour microenvironment",
        "epidemiology; incidence; prevalence; risk factors; cohort study",
        "drug resistance; antimicrobial; MIC; pharmacokinetics",
        "imaging; MRI; PET; radiology; diagnostic accuracy",
        "machine learning; clinical decision support; prediction model; EHR",
        "diabetes; glucose; HbA1c; insulin resistance; metabolic syndrome",
        "neurodegenerative; Alzheimer; tau; amyloid; cognitive decline",
        "surgery; minimally invasive; outcomes; complications; laparoscopy",
    ],
    "if_range":   (4.0, 25.0),
    "cite_range": (10, 350),
    "suffixes": [
        ": A Retrospective Cohort Study",
        ": Results from a Randomised Trial",
        " in a Resource-Limited Setting",
        ": Systematic Review and Meta-Analysis",
        ": Findings from a Multicentre Study",
        ": A Population-Based Analysis",
    ],
},

"Agriculture & Environment": {
    "journals": [
        "Field Crops Research",
        "Agricultural Water Management",
        "Soil and Tillage Research",
        "Agriculture Ecosystems and Environment",
        "Journal of Cleaner Production",
        "Environmental Pollution",
        "Geoderma",
        "Crop Protection",
        "Plant and Soil",
        "Science of the Total Environment",
        "Bioresource Technology",
        "Journal of Environmental Management",
        "Remote Sensing of Environment",
        "Food and Energy Security",
        "Sustainable Agriculture Research",
    ],
    "keywords": [
        "drought tolerance; water deficit; crop yield; stomatal conductance",
        "soil carbon; organic matter; tillage; carbon sequestration; microbial biomass",
        "pest management; biological control; insecticide resistance; IPM",
        "remote sensing; NDVI; satellite imagery; precision agriculture",
        "nitrogen; fertilizer; use efficiency; leaching; nitrous oxide",
        "irrigation; water use efficiency; drip irrigation; deficit irrigation",
        "agroforestry; ecosystem services; biodiversity; land use",
        "heavy metals; phytoremediation; soil contamination; uptake",
        "greenhouse gas; methane; livestock; emission factor",
        "food security; crop diversity; genetic resources; climate change adaptation",
    ],
    "if_range":   (2.0, 8.0),
    "cite_range": (5, 130),
    "suffixes": [
        ": Field Trial Results",
        " Under Semi-Arid Conditions",
        ": A Multi-Season Study",
        ": Implications for Sustainable Management",
        " Across Agro-Ecological Zones",
        ": A Farmer-Participatory Evaluation",
    ],
},

"Arts & Humanities": {
    "journals": [
        "Journal of Cultural Heritage",
        "Digital Humanities Quarterly",
        "Language and Literature",
        "Journal of Linguistic Anthropology",
        "Oral Tradition",
        "Heritage Science",
        "International Journal of Heritage Studies",
        "Journal of the American Musicological Society",
        "Literature Compass",
        "Journal of Postcolonial Writing",
        "Research in African Literatures",
        "Semiotica",
        "Language Documentation and Conservation",
        "Philosophy and Literature",
        "Museum Management and Curatorship",
    ],
    "keywords": [
        "postcolonial; diaspora; identity; cultural memory; literature",
        "oral tradition; intangible heritage; documentation; revitalisation",
        "manuscript; digital archive; palaeography; preservation",
        "endangered language; typology; documentation; linguistic diversity",
        "semiotics; visual culture; rhetoric; discourse; representation",
        "ethnomusicology; ritual; performance; cultural identity",
        "philosophy; aesthetics; consciousness; ethics; metaphysics",
        "translation; equivalence; cross-cultural; literary adaptation",
        "historiography; collective memory; nation; narrative",
        "archaeology; material culture; symbolism; ancient civilisations",
    ],
    "if_range":   (0.5, 3.0),
    "cite_range": (1, 40),
    "suffixes": [
        ": A Critical Reading",
        ": Historical and Contemporary Perspectives",
        ": Case Study from South Asia",
        ": An Ethnographic Account",
        ": Textual Analysis and Interpretation",
        ": Theoretical Frameworks and Empirical Evidence",
    ],
},

"Social Sciences": {
    "journals": [
        "World Development",
        "Journal of Development Economics",
        "Social Science and Medicine",
        "Political Behavior",
        "American Journal of Sociology",
        "Journal of Public Economics",
        "Urban Studies",
        "Criminology",
        "Social Forces",
        "Journal of Health and Social Behavior",
        "Global Environmental Change",
        "Journal of Rural Studies",
        "Journal of Economic Inequality",
        "Governance",
        "International Journal of Urban and Regional Research",
    ],
    "keywords": [
        "poverty; inequality; income; social mobility; welfare",
        "policy evaluation; impact assessment; causal inference; difference-in-differences",
        "urbanisation; migration; housing; labour market; informal economy",
        "gender; women empowerment; social norms; discrimination",
        "political participation; voting; governance; democracy; civil society",
        "education; school enrolment; learning outcomes; equity",
        "mental health; stigma; help-seeking; social support; wellbeing",
        "criminology; recidivism; criminal justice; deterrence",
        "social networks; community; collective action; trust; social capital",
        "climate vulnerability; resilience; adaptation; rural; livelihoods",
    ],
    "if_range":   (1.5, 8.0),
    "cite_range": (4, 150),
    "suffixes": [
        ": Evidence from a Panel Study",
        ": A Quasi-Experimental Analysis",
        ": Lessons from Rural India",
        ": Microeconomic Evidence",
        ": A Mixed-Methods Approach",
        ": Insights from Primary Data",
    ],
},
}

FALLBACK_CONTENT = {
    "journals":   ["PLOS ONE", "Scientific Reports", "Nature Communications",
                   "Royal Society Open Science", "Frontiers in Science"],
    "keywords":   ["interdisciplinary; convergence; methodology; mixed methods",
                   "multi-domain; systems approach; integration; collaboration"],
    "if_range":   (2.0, 10.0),
    "cite_range": (5, 120),
    "suffixes":   [": An Interdisciplinary Perspective",
                   ": Methods, Results, and Implications",
                   ": A Multi-Domain Investigation"],
}

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                           # 5) DOMAIN ASSIGNMENT
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""
    Return the domain string for a publication.
    - Single domain  → that domain as-is.
    - Multi domain   → all declared domains joined with '+',
                       so every publication row reflects the full
                       cross-domain nature of the project.
    The pub_index parameter is kept for API compatibility but is
    not used (all publications from the same project share the
    same domain string).
    """
def assign_pub_domain(domains: list, pub_index: int) -> str:
    
    if not domains:
        return ""
    return "+".join(domains)

def get_content(domain: str) -> dict:
    """
    Look up domain content bank. For cross-domain strings like
    'Medical Sciences+Computer Science', fall back to the first
    declared domain so journals/keywords/IF are always domain-specific.
    """
    if domain in DOMAIN_CONTENT:
        return DOMAIN_CONTENT[domain]
    # Try the first segment of a cross-domain string
    first = domain.split("+")[0].strip()
    return DOMAIN_CONTENT.get(first, FALLBACK_CONTENT)

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                                # 6) TITLE GENERATION
# Strategy: parse the project title into a METHOD phrase and a TOPIC phrase by splitting # on common connective words (for, of, in, via, # using, with, and, under, on). Then recombine them in different ways for each publication, adding a domain suffix. Placeholders use named keys {p0}, {p1}, {p2} (NOT positional {0}, {1}, {2}) to avoid Python's str.format() IndexError when keyword arguments are passed as a dict.
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

# Connective words used to split project title into phrases
CONNECTIVES = re.compile(
    r"\b(for|of|in|via|using|with|and|under|on|from|towards|based on|applied to|"
    r"approach to|methods for|study of|analysis of|modelling of|detection of|"
    r"assessment of|evaluation of|prediction of|estimation of)\b",
    re.IGNORECASE
)

def parse_title_phrases(project_title: str) -> tuple:
    """
    Split project title into up to 3 meaningful phrases.
    Returns (phrase_list) where each element is a non-empty string.
    """
    parts = CONNECTIVES.split(project_title)
    # Remove connective tokens (odd indices after split) and clean up
    phrases = []
    for i, p in enumerate(parts):
        p = p.strip(" :-–—")
        if len(p) > 3 and not CONNECTIVES.fullmatch(p.strip()):
            phrases.append(p)
    return phrases if phrases else [project_title]

# Per-domain recombination patterns.
# {p0}, {p1}, {p2} refer to the parsed phrases (method, topic, context).
# {suffix} is a domain-specific suffix string.
# Using named placeholders avoids IndexError from str.format() positional arg lookup.
RECOMBINATION_PATTERNS = {
    "Computer Science": [
        "{p1}: A {p0} Approach",
        "{p0} in {p2}: {suffix}",
        "Towards {p1} via {p0}",
        "Efficient {p0} for {p1}",
        "Scalable {p0} Applied to {p1}: {suffix}",
        "{p1} Using {p0}: {suffix}",
        "Robust {p1} Under {p0} Constraints",
        "{p0} for {p1} in {p2}",
    ],
    "Engineering": [
        "Experimental Analysis of {p1} Using {p0}",
        "{p0} for {p1}: {suffix}",
        "Performance of {p1} Under {p0}",
        "Design and Validation of {p0} for {p1}",
        "{p1} in {p2}: {suffix}",
        "Numerical Study of {p1} via {p0}",
        "Optimisation of {p1} Using {p0}: {suffix}",
        "{p0}-Based Assessment of {p1}",
    ],
    "Mathematics": [
        "Existence and Properties of {p1} Under {p0}",
        "{p0} Applied to {p1}: {suffix}",
        "Convergence of {p0} for {p1}",
        "A {p0} Framework for {p1}",
        "{p1} via {p0}: {suffix}",
        "Theoretical Analysis of {p1} Using {p0}",
        "On the Structure of {p1} in {p0}",
        "{p0} Methods for {p1}: {suffix}",
    ],
    "Physics": [
        "Observation of {p1} in {p2}",
        "{p0} Study of {p1}: {suffix}",
        "Electronic Properties of {p1} Under {p0}",
        "{p1} in {p2}: {suffix}",
        "Transport Phenomena in {p1} via {p0}",
        "Phase Behaviour of {p1}: {suffix}",
        "Spectroscopic Investigation of {p1} Using {p0}",
        "{p0} Signatures of {p1} in {p2}",
    ],
    "Chemistry": [
        "Synthesis and Characterisation of {p1} via {p0}",
        "{p0} of {p1}: {suffix}",
        "Mechanistic Study of {p1} Using {p0}",
        "{p1} via {p0}: {suffix}",
        "Catalytic {p0} for {p1}: {suffix}",
        "Green {p0} of {p1} in {p2}",
        "Spectroscopic Analysis of {p1} Under {p0}",
        "{p0}-Driven Synthesis of {p1}: {suffix}",
    ],
    "Biology": [
        "Molecular Mechanisms of {p1} in {p2}",
        "{p0} of {p1}: {suffix}",
        "Functional Role of {p1} in {p0}",
        "{p1} Under {p0}: {suffix}",
        "Genome-Wide Analysis of {p1} via {p0}",
        "{p0} Reveals {p1} in {p2}",
        "Transcriptomic Basis of {p1} During {p0}",
        "{p0}-Mediated Regulation of {p1}",
    ],
    "Medical Sciences": [
        "Clinical Outcomes of {p1} in {p2}",
        "{p0} for {p1}: {suffix}",
        "Efficacy of {p0} in {p1}: {suffix}",
        "Epidemiology of {p1} in {p2}",
        "{p1} Among {p2}: {suffix}",
        "Diagnostic Accuracy of {p0} for {p1}",
        "Risk Factors for {p1} in {p2}: {suffix}",
        "{p0}-Based {p1}: {suffix}",
    ],
    "Agriculture & Environment": [
        "Effect of {p0} on {p1} in {p2}",
        "{p0} for {p1}: {suffix}",
        "Response of {p1} to {p0} Under {p2}",
        "{p1} Under {p0} Conditions: {suffix}",
        "Impact of {p0} on {p1} in {p2}",
        "Management of {p1} Through {p0}: {suffix}",
        "{p0}-Based Improvement of {p1} in {p2}",
        "Spatial Variability of {p1} Under {p0}",
    ],
    "Arts & Humanities": [
        "Postcolonial Perspectives on {p1} in {p2}",
        "{p0} and {p1}: {suffix}",
        "Memory, Identity, and {p1} in {p2}",
        "{p1} in {p2}: {suffix}",
        "A Critical Reading of {p1} Through {p0}",
        "Language, Power, and {p1} in {p2}",
        "{p0} as a Lens for {p1}: {suffix}",
        "Representation of {p1} in {p0}",
    ],
    "Social Sciences": [
        "Determinants of {p1} Among {p2}",
        "Impact of {p0} on {p1}: {suffix}",
        "{p1} in {p2}: {suffix}",
        "Gender and {p1} in {p2}: {suffix}",
        "Does {p0} Improve {p1}? {suffix}",
        "Barriers to {p1} Among {p2}",
        "Policy Implications of {p0} for {p1}",
        "{p0} and {p1}: Evidence from {p2}",
    ],
}

FALLBACK_PATTERNS = [
    "{p0} and {p1}: An Interdisciplinary Study",
    "Integrated {p0} for {p1} in {p2}",
    "{p1} via {p0}: Methods and Implications",
    "Cross-Domain Analysis of {p1} Using {p0}",
]

def make_pub_title(project_title: str, domains: list, pub_index: int) -> str:
    """
    Generate a realistic paper title derived from the project title.
    Each call uses a different recombination pattern so publications
    from the same project look distinct but related.
    """
    primary = domains[0] if domains else ""
    content  = get_content(primary)
    suffix   = random.choice(content.get("suffixes", FALLBACK_CONTENT["suffixes"]))

    phrases = parse_title_phrases(project_title)

    # Pad to at least 3 slots
    while len(phrases) < 3:
        phrases.append(phrases[-1])

    p0 = phrases[0].strip()
    p1 = phrases[1].strip() if len(phrases) > 1 else p0
    p2 = phrases[2].strip() if len(phrases) > 2 else p1

    patterns = RECOMBINATION_PATTERNS.get(primary, FALLBACK_PATTERNS)
    # Rotate through patterns based on pub_index so each publication differs
    pattern  = patterns[pub_index % len(patterns)]

    # FIX: use named placeholders {p0}, {p1}, {p2}, {suffix} instead of
    # positional {0}, {1}, {2} to avoid IndexError from str.format()
    # when keyword-only arguments are passed.
    title = pattern.format(p0=p0, p1=p1, p2=p2, suffix=suffix)

    # Clean up: collapse multiple spaces, fix capitalisation artefacts
    title = re.sub(r"\s{2,}", " ", title).strip()
    # Ensure first character is uppercase
    if title:
        title = title[0].upper() + title[1:]

    return title

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                              # 7) NUMERIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
def make_volume(year: int) -> int:
    base = random.randint(25, 55)
    return max(1, year - (2025 - base))

def make_issue(domain: str) -> int:
    first = domain.split("+")[0].strip()
    if first in {"Arts & Humanities", "Mathematics"}:
        return random.randint(1, 4)
    return random.randint(1, 12)

def make_citation_count(year: int, content: dict) -> int:
    lo, hi = content["cite_range"]
    recency = max(0.2, 1.0 - (year - 2016) * 0.10)
    return random.randint(lo, max(lo + 1, int(hi * recency)))

def make_impact_factor(content: dict) -> float:
    lo, hi = content["if_range"]
    return round(random.uniform(lo, hi), 3)

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                                  # 8) LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
print("Loading projects ...")
df = pd.read_csv(PROJECTS_CSV)

COL_PROJECT_ID = "Project_ID"
COL_GRANT_ID   = "Grant_ID"
COL_TITLE      = "Project_Title"
COL_DOMAIN     = "Domain"
COL_SDATE      = "Start_Date"
COL_EDATE      = "End_Date"
COL_PI         = "Principal_Investigator"
COL_COPI       = "Co_Investigators"

for c in [COL_PROJECT_ID, COL_GRANT_ID, COL_TITLE, COL_DOMAIN,
          COL_SDATE, COL_EDATE, COL_PI, COL_COPI]:
    if c not in df.columns:
        df[c] = ""

df[COL_PROJECT_ID] = df[COL_PROJECT_ID].astype(str).str.strip()
df[COL_GRANT_ID]   = df[COL_GRANT_ID].astype(str).str.strip()
df[COL_TITLE]      = df[COL_TITLE].astype(str).str.strip()
df[COL_DOMAIN]     = df[COL_DOMAIN].astype(str).str.strip()
df[COL_PI]         = df[COL_PI].apply(normalize_rid)
df[COL_COPI]       = df[COL_COPI].apply(
    lambda x: split_ids(x) if not isinstance(x, list) else x
)
print(f"  {len(df)} projects loaded")

# Optional: grants file (for grant_id cross-check)
try:
    df_grants = pd.read_csv(GRANTS_CSV)
    funded    = set(df_grants["project_id"].astype(str).str.strip())
    print(f"  {len(funded)} funded projects from {GRANTS_CSV}")
except Exception:
    print("  Grants file not found — using Grant_ID from project file")

# Optional: profile validation
profile_ids = None
try:
    prof    = pd.read_csv(PROFILE_CSV)
    rid_col = next((c for c in prof.columns if "r_id" in c.lower()), prof.columns[0])
    prof[rid_col] = prof[rid_col].apply(normalize_rid)
    profile_ids   = set(prof[rid_col].dropna())
    print(f"  {len(profile_ids)} researchers in profile (author validation ON)")
except Exception:
    print("  Profile not loaded — author validation OFF")

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                                      # 9) GENERATE
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
print("\nGenerating publications ...")
pub_rows    = []
pub_counter = 1

for _, row in df.iterrows():
    project_id  = norm_str(row[COL_PROJECT_ID])
    grant_id    = norm_str(row[COL_GRANT_ID])
    proj_title  = norm_str(row[COL_TITLE])
    domains     = split_domains(row[COL_DOMAIN])

    pi    = row[COL_PI]
    copis = row[COL_COPI] if isinstance(row[COL_COPI], list) else []
    team  = []
    if pi:
        team.append(pi)
    team += [c for c in copis if c and c != pi]
    team  = list(dict.fromkeys(team))

    if profile_ids is not None:
        team = [t for t in team if t in profile_ids]

    if not team:
        continue

    # Year window
    y1 = parse_year(row.get(COL_SDATE, ""))
    y2 = parse_year(row.get(COL_EDATE, ""))
    if y1 is None and y2 is None:
        y_low, y_high = DEFAULT_YEAR_MIN, DEFAULT_YEAR_MAX
    else:
        y_low  = y1 if y1 is not None else max(DEFAULT_YEAR_MIN, (y2 or 2020) - 3)
        y_high = y2 if y2 is not None else min(DEFAULT_YEAR_MAX, (y1 or 2018) + 3)
        y_low  = max(DEFAULT_YEAR_MIN, y_low)
        y_high = min(DEFAULT_YEAR_MAX, y_high)
        if y_low > y_high:
            y_low, y_high = DEFAULT_YEAR_MIN, DEFAULT_YEAR_MAX

    kpub = random.randint(MIN_PUB_PER_PROJECT, MAX_PUB_PER_PROJECT)

    for j in range(kpub):
        pub_id = make_pub_id(pub_counter)
        pub_counter += 1

        # ── Authors ──────────────────────────────
        if len(team) == 1:
            authors = team[:]
        else:
            k_auth = random.randint(2, min(len(team), 4))
            if pi and pi in team and random.random() < 0.85:
                rest    = [a for a in team if a != pi]
                authors = [pi] + random.sample(rest, min(k_auth - 1, len(rest)))
            else:
                authors = random.sample(team, k_auth)

        # ── Year ─────────────────────────────────
        year = random.randint(y_low, y_high)

        # ── Title (derived from project title) ───
        title = make_pub_title(proj_title, domains, j)

        # ── Domain for this pub (round-robin across declared domains) ─
        pub_domain = assign_pub_domain(domains, j)
        content    = get_content(pub_domain)

        # ── Journal (matches pub_domain) ─────────
        journal = random.choice(content["journals"])

        # ── Keywords (matches pub_domain) ────────
        keywords = random.choice(content["keywords"])

        # ── Volume / Issue ───────────────────────
        volume = make_volume(year)
        issue  = make_issue(pub_domain)

        # ── IF and Citations ─────────────────────
        impact_factor  = make_impact_factor(content)
        citation_count = make_citation_count(year, content)

        pub_rows.append({
            "publication_id":  pub_id,
            "title":           title,
            "journal":         journal,
            "volume":          volume,
            "issue":           issue,
            "year":            year,
            "doi":             make_doi(pub_id),
            "keywords":        keywords,
            "citation_count":  citation_count,
            "impact_factor":   impact_factor,
            "domain":          pub_domain,
            "project_id":      project_id,
            "grant_id":        grant_id,
            "author_ids":      ";".join(authors),
        })

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                        # 10) SAVE & REPORT
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
pub_df = pd.DataFrame(pub_rows, columns=[
    "publication_id", "title", "journal", "volume", "issue", "year",
    "doi", "keywords", "citation_count", "impact_factor", "domain",
    "project_id", "grant_id", "author_ids",
])
pub_df.to_csv(OUT_PUB_CSV, index=False)


