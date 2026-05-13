# Researcher Profile Dataset Generator
# Generates 3000 synthetic researcher profiles with all required fields.

import pandas as pd
import numpy as np
import random
import math
import csv
from collections import Counter
from faker import Faker

fake = Faker()  
random.seed(42)
np.random.seed(42)

# =================================================================================================================================================
                                                      # 1. DOMAIN & DEPARTMENT DEFINITIONS
# =================================================================================================================================================

DOMAIN_KEYWORDS = {
    "Computer Science": [
        "deep learning", "GNN", "reinforcement learning", "transfer learning", "blockchain",
        "cybersecurity", "computer vision", "NLP", "federated learning",
        "cloud computing", "edge computing", "data mining",
        "big data", "XAI", "LLMs", "embeddings", "GANs", "HCI",
        "cryptography", "graph analytics", "clustering",
        "distributed systems", "IoT", "anomaly detection",
        "recommender systems", "information retrieval", "bioinformatics",
        "optimization", "virtualization", "semantic web"
    ],
    "Engineering": [
        "CFD", "FEA", "MEMS", "tribology", "nanomaterials", "smart grid",
        "robotics", "embedded systems", "power electronics", "mechatronics",
        "signal processing", "control systems", "nonlinear dynamics",
        "thermal analysis", "structural design", "vibration control",
        "additive manufacturing", "photonics", "acoustics",
        "wireless power", "sensor networks", "EV systems", "automation",
        "hydraulics", "aerodynamics", "materials testing",
        "microfabrication", "renewable energy", "instrumentation",
        "telecommunication systems"
    ],
    "Mathematics": [
        "PDEs", "ODEs", "stochastic processes", "combinatorics", "optimization",
        "number theory", "topology", "graph theory", "probability", "statistics",
        "algebraic structures", "functional analysis", "real analysis",
        "complex analysis", "spectral theory", "Markov chains",
        "linear algebra", "game theory", "numerical methods",
        "cryptography", "fractals", "measure theory",
        "dynamical systems", "tensor calculus", "operator theory",
        "random matrices", "multivariate analysis",
        "discrete mathematics", "calculus of variations", "geometry"
    ],
    "Physics": [
        "quantum mechanics", "relativity", "optics", "electromagnetism",
        "condensed matter", "astrophysics", "cosmology", "nanophotonics",
        "plasma physics", "thermodynamics", "spectroscopy",
        "quantum optics", "superconductivity", "particle physics",
        "nuclear physics", "fluid dynamics", "quantum computing",
        "magnetism", "photonics", "high-energy physics",
        "atomic physics", "wave propagation", "nonlinear optics",
        "semiconductor physics", "radio astronomy", "acoustics",
        "computational physics", "statistical physics", "dark matter",
        "gravitational waves"
    ],
    "Chemistry": [
        "catalysis", "spectroscopy", "chromatography", "electrochemistry",
        "nanochemistry", "polymer science", "thermodynamics",
        "reaction kinetics", "medicinal chemistry", "organic synthesis",
        "photochemistry", "crystallography", "mass spectrometry",
        "quantum chemistry", "chemical bonding", "heterogeneous catalysis",
        "enzymology", "organometallics", "supramolecular chemistry",
        "peptide synthesis", "metabolomics", "redox chemistry",
        "chemical kinetics", "materials chemistry", "green chemistry",
        "coordination chemistry", "analytical chemistry", "adsorption",
        "synthetic chemistry", "surface chemistry"
    ],
    "Biology": [
        "genomics", "proteomics", "transcriptomics", "metabolomics",
        "cell signaling", "microbiome", "bioinformatics", "molecular biology",
        "immunology", "neurobiology", "evolutionary biology",
        "structural biology", "enzyme kinetics", "virology", "epigenetics",
        "DNA repair", "gene expression", "protein folding", "CRISPR",
        "developmental biology", "systems biology", "ecology",
        "microbiology", "cytology", "phylogenetics", "toxicology",
        "cell cycle", "biomaterials", "metabolic pathways", "biodiversity"
    ],
    "Medical Sciences": [
        "oncology", "radiology", "pathology", "neurology", "cardiology",
        "immunotherapy", "pharmacology", "genomics", "biomarkers",
        "epidemiology", "diagnostics", "MRI", "CT imaging", "ultrasound",
        "clinical trials", "pediatrics", "geriatrics", "endocrinology",
        "hematology", "psychiatry", "infectious diseases",
        "metabolic disorders", "public health", "surgery",
        "therapeutic efficacy", "neuroimaging", "rehabilitation",
        "drug delivery", "gene therapy", "respiratory function"
    ],
    "Agriculture & Environment": [
        "agronomy", "soil fertility", "crop modeling", "precision farming",
        "hydrology", "biodiversity", "irrigation", "entomology",
        "agroforestry", "carbon sequestration", "climate resilience",
        "land use", "pest management", "pollution", "sustainability",
        "GIS", "remote sensing", "water quality", "nutrient cycling",
        "marine ecology", "forest management", "watershed hydrology",
        "organic farming", "invasive species", "GHG emissions",
        "ecosystem dynamics", "fisheries", "plant genetics",
        "soil microbiology", "conservation"
    ],
    "Arts & Humanities": [
        "semiotics", "hermeneutics", "classical literature", "syntax",
        "narrative theory", "aesthetics", "discourse analysis", "pragmatics",
        "rhetoric", "historiography", "translation studies", "morphology",
        "phonology", "semantics", "textual criticism", "art history",
        "visual culture", "anthropology", "film theory", "media aesthetics",
        "iconography", "ethnography", "literature", "symbolism",
        "poetics", "dramaturgy", "lexicography", "epigraphy",
        "folklore", "historical linguistics"
    ],
    "Social Sciences": [
        "governance", "public policy", "social networks", "demography",
        "migration", "gender studies", "political economy", "urban studies",
        "development", "globalization", "social capital", "inequality",
        "community studies", "behavioral economics", "education policy",
        "digital society", "social mobility", "human geography",
        "criminology", "cultural studies", "social psychology",
        "public administration", "poverty", "welfare systems",
        "civic engagement", "media studies", "entrepreneurship",
        "labour markets", "institutional economics", "social research"
    ]
}

DEPARTMENTS = {
    "Computer Science": "Computer Science",
    "Information Technology": "Computer Science",
    "Data Science": "Computer Science",
    "AI & Machine Learning": "Computer Science",
    "Mechanical Engineering": "Engineering",
    "Electrical Engineering": "Engineering",
    "Civil Engineering": "Engineering",
    "Electronics & Communication Engineering": "Engineering",
    "Automobile Engineering": "Engineering",
    "Mathematics": "Mathematics",
    "Statistics": "Mathematics",
    "Physics": "Physics",
    "Applied Physics": "Physics",
    "Astrophysics": "Physics",
    "Chemistry": "Chemistry",
    "Organic Chemistry": "Chemistry",
    "Materials Chemistry": "Chemistry",
    "Biotechnology": "Biology",
    "Microbiology": "Biology",
    "Life Sciences": "Biology",
    "Genetics": "Biology",
    "Medicine": "Medical Sciences",
    "Nursing": "Medical Sciences",
    "Physiotherapy": "Medical Sciences",
    "Pharmacy": "Medical Sciences",
    "Agriculture": "Agriculture & Environment",
    "Environmental Science": "Agriculture & Environment",
    "Horticulture": "Agriculture & Environment",
    "English": "Arts & Humanities",
    "History": "Arts & Humanities",
    "Philosophy": "Arts & Humanities",
    "Sociology": "Social Sciences",
    "Political Science": "Social Sciences",
    "Public Administration": "Social Sciences"
}

DEPARTMENT_KEYWORDS = {
    "Computer Science": ["algorithms", "data structures", "software engineering", "operating systems", "databases"],
    "Information Technology": ["information systems", "IT infrastructure", "DevOps", "system administration", "network security"],
    "Data Science": ["data analytics", "feature engineering", "time series", "A/B testing", "causal inference"],
    "AI & Machine Learning": ["self-supervised learning", "foundation models", "prompt engineering", "RAG", "model robustness"],
    "Mechanical Engineering": ["machine design", "manufacturing", "CAD/CAM", "vibration analysis", "heat transfer"],
    "Electrical Engineering": ["power systems", "grid integration", "protection systems", "microgrids", "power quality"],
    "Civil Engineering": ["structural engineering", "construction management", "transportation", "geotechnical", "BIM"],
    "Electronics & Communication Engineering": ["VLSI", "embedded systems", "wireless communication", "antenna design", "DSP"],
    "Automobile Engineering": ["vehicle dynamics", "powertrain", "engine modeling", "EV design", "ADAS"],
    "Mathematics": ["linear algebra", "real analysis", "numerical methods", "optimization theory", "graph theory"],
    "Statistics": ["regression", "experimental design", "Bayesian statistics", "survival analysis", "multivariate analysis"],
    "Physics": ["condensed matter", "spectroscopy", "statistical physics", "optics", "plasma physics"],
    "Applied Physics": ["semiconductors", "thin films", "device physics", "photonics", "instrumentation"],
    "Astrophysics": ["cosmology", "dark matter", "gravitational waves", "stellar evolution", "radio astronomy"],
    "Chemistry": ["analytical chemistry", "chemical kinetics", "spectroscopy", "electrochemistry", "catalysis"],
    "Organic Chemistry": ["organic synthesis", "reaction mechanisms", "stereochemistry", "medicinal chemistry", "heterocycles"],
    "Materials Chemistry": ["nanomaterials", "polymers", "surface chemistry", "energy materials", "material characterization"],
    "Biotechnology": ["bioprocessing", "fermentation", "recombinant DNA", "synthetic biology", "bioreactors"],
    "Microbiology": ["antimicrobial resistance", "pathogenesis", "biofilms", "clinical microbiology", "microbiome"],
    "Life Sciences": ["cell biology", "physiology", "ecology", "developmental biology", "biochemistry"],
    "Genetics": ["population genetics", "genome editing", "GWAS", "epigenetics", "functional genomics"],
    "Medicine": ["diagnostics", "clinical medicine", "epidemiology", "therapeutics", "surgery"],
    "Nursing": ["patient care", "community health", "infection control", "critical care", "nursing ethics"],
    "Physiotherapy": ["rehabilitation", "biomechanics", "exercise therapy", "manual therapy", "gait analysis"],
    "Pharmacy": ["pharmacokinetics", "drug delivery", "formulation", "pharmacovigilance", "clinical pharmacy"],
    "Agriculture": ["crop science", "soil science", "irrigation", "pest management", "precision farming"],
    "Environmental Science": ["environmental monitoring", "air quality", "water quality", "waste management", "climate change"],
    "Horticulture": ["floriculture", "plant breeding", "nursery management", "post-harvest", "protected cultivation"],
    "English": ["literary theory", "linguistics", "discourse analysis", "translation studies", "digital humanities"],
    "History": ["historiography", "archival studies", "cultural history", "political history", "economic history"],
    "Philosophy": ["ethics", "logic", "epistemology", "political philosophy", "philosophy of science"],
    "Sociology": ["social theory", "inequality", "community studies", "digital society", "social research methods"],
    "Political Science": ["governance", "international relations", "public policy", "political economy", "elections"],
    "Public Administration": ["public management", "policy implementation", "bureaucracy", "e-governance", "accountability"],
}

# =================================================================================================================================================
                                                            # 2. CONSTANTS
# =================================================================================================================================================

UNIVERSITIES = [
    "IIT Madras", "IIT Delhi", "IIT Bombay", "IIT Kanpur", "IIT Kharagpur",
    "NIT Trichy", "Anna University", "IISc Bangalore", "University of Hyderabad",
    "JNU Delhi", "Pondicherry University"
]

ROLES = ["Professor", "Associate Professor", "Assistant Professor", "Research Scholar"]
# ===========================================================================
# 2.1 CONSTRAINT: Role-based h_index ranges
# ===========================================================================

# Professor: 15–40 | Associate Professor: 8–25 | Assistant Professor: 2–15 | Research Scholar: 0–8

ROLE_H_INDEX_RANGE = {
    "Professor":            (15, 40),
    "Associate Professor":  (8,  25),
    "Assistant Professor":  (2,  15),
    "Research Scholar":     (0,   8),
}
# ===========================================================================
# 2.2 CONSTRAINT: Role-based career start year ranges
# ===========================================================================

# Professor: 1985–2005 | Associate Professor: 1995–2012 | Assistant Professor: 2005–2018 | Research Scholar: 2015–2023

ROLE_CAREER_START_RANGE = {
    "Professor":            (1985, 2005),
    "Associate Professor":  (1995, 2012),
    "Assistant Professor":  (2005, 2018),
    "Research Scholar":     (2015, 2023),
}

CURRENT_YEAR = 2025
# ===========================================================================
# 2.3 CONSTRAINT: d_id is a fixed mapping — each of the 34 departments gets exactly one ID from D001 to D034.
# ===========================================================================

DEPT_TO_ID = {dept: f"D{i:03d}" for i, dept in enumerate(DEPARTMENTS.keys(), start=1)}
ALL_DEPTS   = list(DEPARTMENTS.keys())
ALL_DOMAINS = list(DOMAIN_KEYWORDS.keys())
K           = len(ALL_DOMAINS)
H_MAX       = math.log(K) if K > 1 else 1.0

# ===========================================================================
# 2.4 KEYWORD → DOMAIN MAP  (for inter_score)
# ===========================================================================

KEYWORD_TO_DOMAIN: dict[str, set] = {}

for _dom, _kws in DOMAIN_KEYWORDS.items():
    for _kw in _kws:
        _kw = str(_kw).strip().lower()
        if _kw:
            KEYWORD_TO_DOMAIN.setdefault(_kw, set()).add(_dom)

for _dept, _kws in DEPARTMENT_KEYWORDS.items():
    _dom = DEPARTMENTS[_dept]
    for _kw in _kws:
        _kw = str(_kw).strip().lower()
        if _kw:
            KEYWORD_TO_DOMAIN.setdefault(_kw, set()).add(_dom)

# ===========================================================================
# 2.5 HELPER FUNCTIONS
# ===========================================================================

def generate_orcid() -> str:
    """Generate a random ORCID-format string."""
    return "-".join(str(random.randint(1000, 9999)) for _ in range(4))


def generate_name() -> str:
    """
    **CONSTRAINT: Names are realistic Indian full names (first + last).**
    
    """
    return fake.name()

def dept_keyword_pool(d_name: str) -> list:
    """Return combined domain + department keywords for a given department."""
    dom   = DEPARTMENTS.get(d_name, "")
    d_kws = DEPARTMENT_KEYWORDS.get(d_name, [])
    dom_kws = DOMAIN_KEYWORDS.get(dom, [])
    seen, pool = set(), []
    for kw in dom_kws + d_kws:
        kw_norm = str(kw).strip()
        if kw_norm and kw_norm not in seen:
            seen.add(kw_norm)
            pool.append(kw_norm)
    return pool

#Generates the list of keywords for authors
 """
    **CONSTRAINT: 3–6 keywords per researcher.**
    **CONSTRAINT: 35% chance of cross-domain keywords (1–3 from another domain).**
    """

def sample_author_keywords(d_name: str, primary_domain: str,
                            p_cross: float = 0.35) -> list[str]:
    
    total_k = random.randint(3, 6)
    base    = dept_keyword_pool(d_name)

    if random.random() < p_cross and K > 1 and total_k >= 2:
        cross_k  = random.randint(1, min(3, total_k - 1))
        base_k   = total_k - cross_k

        other_dom  = random.choice([d for d in ALL_DOMAINS if d != primary_domain])
        cross_pool = DOMAIN_KEYWORDS[other_dom]

        base_sample  = random.sample(base,       min(base_k,  len(base)))
        cross_sample = random.sample(cross_pool, min(cross_k, len(cross_pool)))
        kws = base_sample + cross_sample
        random.shuffle(kws)
        return kws

    return random.sample(base, min(total_k, len(base)))
#======================================================================
# 2.6 interdisciplinary score calculation
#======================================================================
def inter_score_shannon(kw_list: list[str], primary_domain: str) -> float:
    """
    **CONSTRAINT: inter_score is Shannon entropy (0–1) over domain distribution
    of the researcher's keywords, normalised by log(10).**
    0 = purely single-domain; 1 = perfectly spread across all 10 domains.
    """
    domains = []
    for kw in kw_list:
        kw_l = kw.strip().lower()
        dset = KEYWORD_TO_DOMAIN.get(kw_l)
        if not dset:
            continue
        domains.append(primary_domain if primary_domain in dset else next(iter(dset)))

    if not domains:
        return 0.0

    cnt   = Counter(domains)
    total = sum(cnt.values())
    H     = -sum((c / total) * math.log(c / total) for c in cnt.values())
    return round(abs(H / H_MAX), 3)
#======================================================================
# 2.7 Topic Vector Computations
#======================================================================
"""
    **CONSTRAINT: topic_vector is a 10-dim normalised float vector, one entry
    per domain, derived from keyword-to-domain mapping.**
    """
def compute_topic_vector(kw_list: list[str], primary_domain: str) -> list[float]:
    
    domain_index = {d: i for i, d in enumerate(ALL_DOMAINS)}
    vec = [0.0] * K

    for kw in kw_list:
        kw_l = kw.strip().lower()
        dset = KEYWORD_TO_DOMAIN.get(kw_l)
        if not dset:
            continue
        d = primary_domain if primary_domain in dset else next(iter(dset))
        vec[domain_index[d]] += 1.0

    total = sum(vec)
    if total > 0:
        vec = [round(v / total, 2) for v in vec]
    return vec


# =================================================================================================================================================
                                                      # 3. MAIN GENERATION LOOP
# =================================================================================================================================================
# ─────────────────────────────
# **CONSTRAINT: Total researchers = 3000, r_id from R001 to R3000.**
──────────────────────────────
NUM_RESEARCHERS = 3000

rows = []

for i in range(1, NUM_RESEARCHERS + 1):
    r_id   = f"R{i:04d}"
    d_name = random.choice(ALL_DEPTS)
    d_id   = DEPT_TO_ID[d_name]
    primary_domain = DEPARTMENTS[d_name]
    role = random.choice(ROLES)
    name = generate_name() #name

    # Keywords, inter_score, topic_vector
    kw_list       = sample_author_keywords(d_name, primary_domain)
    kw_str        = "; ".join(kw_list)   # semicolon separator avoids CSV column-shift in Excel
    inter_score   = inter_score_shannon(kw_list, primary_domain)
    topic_vec     = compute_topic_vector(kw_list, primary_domain)

    # **CONSTRAINT: h_index range depends on role (see ROLE_H_INDEX_RANGE).**

    h_lo, h_hi = ROLE_H_INDEX_RANGE[role]
    h_index    = random.randint(h_lo, h_hi)

    # **CONSTRAINT: career_start_year range depends on role
    #   (see ROLE_CAREER_START_RANGE).**

    yr_lo, yr_hi  = ROLE_CAREER_START_RANGE[role]
    career_start  = random.randint(yr_lo, yr_hi)

    # **CONSTRAINT: is_active is binary (0 or 1).**
    is_active = random.choice([0, 1])

    # **CONSTRAINT: career_end_date is NULL for active researchers;
    #   for inactive researchers it is career_start_year+1 … 2023.**
    if is_active:
        career_end_date = pd.NA
        # **CONSTRAINT: years_exp = CURRENT_YEAR − career_start_year
        #   for active researchers.**
        years_exp = CURRENT_YEAR - career_start
    else:
        end_hi = max(career_start + 1, 2023)
        career_end_date = random.randint(career_start + 1, end_hi)
        # **CONSTRAINT: years_exp = career_end_date − career_start_year
        #   for inactive researchers.**
        years_exp = int(career_end_date) - career_start

    # **CONSTRAINT: ORCID format is XXXX-XXXX-XXXX-XXXX (4-digit groups).**
    orcid = generate_orcid()

    # **CONSTRAINT: Email is a fake but plausible address.**
    email = fake.email()

    # **CONSTRAINT: Affiliation is one of 11 recognised Indian universities.**
    affiliation = random.choice(UNIVERSITIES)

    rows.append({
        "r_id":             r_id,
        "name":             name,
        "d_id":             d_id,
        "d_name":           d_name,
        "role":             role,
        "primary_domain":   primary_domain,
        "author_keywords":  kw_str,
        "h_index":          h_index,
        "inter_score":      inter_score,
        "career_start_year": career_start,
        "is_active":        is_active,
        "career_end_date":  career_end_date,
        "years_exp":        years_exp,
        "ORCID":            orcid,
        "email":            email,
        "affiliation":      affiliation,
        "topic_vector":     topic_vec,
    })

# =================================================================================================================================================
                                                    # 4. BUILD DATAFRAME & FIX TYPES
# =================================================================================================================================================

df = pd.DataFrame(rows)

# **CONSTRAINT: Year columns are stored as nullable integers (no decimals).**
for col in ["career_start_year", "career_end_date", "years_exp"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

# =================================================================================================================================================
                                                            # 5. SAVE
# =================================================================================================================================================

OUT_CSV = "Researchers_Profile.csv"
df.to_csv(OUT_CSV, index=False, encoding="utf-8", na_rep="", quoting=csv.QUOTE_ALL)

# =================================================================================================================================================
                                                                # 6. SUMMARY
# =================================================================================================================================================

print(f"  Saved → {OUT_CSV}")
print(f"   Shape : {df.shape}")
#print(f"   Duplicate r_id : {df['r_id'].duplicated().sum()}")
#print(f"\nRole distribution:\n{df['role'].value_counts().to_string()}")
#print(f"\nis_active distribution:\n{df['is_active'].value_counts().to_string()}")
#print(f"\ninter_score summary:\n{df['inter_score'].describe().to_string()}")
#print(f"\nh_index by role:\n{df.groupby('role')['h_index'].agg(['min','max','mean']).to_string()}")
print(f"\nSample rows:\n{df.head(3).to_string()}")
