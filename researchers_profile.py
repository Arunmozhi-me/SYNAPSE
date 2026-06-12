# ====================================================================================================================================================================================
                                                        # Researcher Profile Generator with Temporal Domain Drift
# ====================================================================================================================================================================================

import pandas as pd
import numpy as np
import random
import math
import csv
from collections import Counter
from faker import Faker

fake = Faker("en_IN")
random.seed(42)
np.random.seed(42)

# ====================================================================================================================================================================================
                                                                        # 1. DOMAIN KEYWORDS
# ====================================================================================================================================================================================

DOMAIN_KEYWORDS = {
    "Computer Science": [
        "deep learning", "GNN", "reinforcement learning", "transfer learning", "blockchain", "cybersecurity", "computer vision", "NLP",
        "federated learning", "cloud computing", "edge computing", "data mining",
        "big data", "XAI", "LLMs", "embeddings", "GANs", "HCI",
        "cryptography", "graph analytics", "clustering", "distributed systems",
        "IoT", "anomaly detection", "recommender systems",
        "information retrieval", "bioinformatics", "optimization",
        "virtualization", "semantic web",
    ],
    "Engineering": [
        "CFD", "FEA", "MEMS", "tribology", "nanomaterials", "smart grid",
        "robotics", "embedded systems", "power electronics", "mechatronics",
        "signal processing", "control systems", "nonlinear dynamics",
        "thermal analysis", "structural design", "vibration control",
        "additive manufacturing", "photonics", "acoustics", "wireless power",
        "sensor networks", "EV systems", "automation", "hydraulics",
        "aerodynamics", "materials testing", "microfabrication",
        "renewable energy", "instrumentation", "telecommunication systems",
    ],
    "Mathematics": [
        "PDEs", "ODEs", "stochastic processes", "combinatorics", "optimization",
        "number theory", "topology", "graph theory", "probability", "statistics",
        "algebraic structures", "functional analysis", "real analysis",
        "complex analysis", "spectral theory", "Markov chains", "linear algebra",
        "game theory", "numerical methods", "cryptography", "fractals",
        "measure theory", "dynamical systems", "tensor calculus",
        "operator theory", "random matrices", "multivariate analysis",
        "discrete mathematics", "calculus of variations", "geometry",
    ],
    "Physics": [
        "quantum mechanics", "relativity", "optics", "electromagnetism",
        "condensed matter", "astrophysics", "cosmology", "nanophotonics",
        "plasma physics", "thermodynamics", "spectroscopy", "quantum optics",
        "superconductivity", "particle physics", "nuclear physics",
        "fluid dynamics", "quantum computing", "magnetism", "photonics",
        "high-energy physics", "atomic physics", "wave propagation",
        "nonlinear optics", "semiconductor physics", "radio astronomy",
        "acoustics", "computational physics", "statistical physics",
        "dark matter", "gravitational waves",
    ],
    "Chemistry": [
        "catalysis", "spectroscopy", "chromatography", "electrochemistry",
        "nanochemistry", "polymer science", "thermodynamics", "reaction kinetics",
        "medicinal chemistry", "organic synthesis", "photochemistry",
        "crystallography", "mass spectrometry", "quantum chemistry",
        "chemical bonding", "heterogeneous catalysis", "enzymology",
        "organometallics", "supramolecular chemistry", "peptide synthesis",
        "metabolomics", "redox chemistry", "chemical kinetics",
        "materials chemistry", "green chemistry", "coordination chemistry",
        "analytical chemistry", "adsorption", "synthetic chemistry",
        "surface chemistry",
    ],
    "Biology": [
        "genomics", "proteomics", "transcriptomics", "metabolomics",
        "cell signaling", "microbiome", "bioinformatics", "molecular biology",
        "immunology", "neurobiology", "evolutionary biology",
        "structural biology", "enzyme kinetics", "virology", "epigenetics",
        "DNA repair", "gene expression", "protein folding", "CRISPR",
        "developmental biology", "systems biology", "ecology", "microbiology",
        "cytology", "phylogenetics", "toxicology", "cell cycle",
        "biomaterials", "metabolic pathways", "biodiversity",
    ],
    "Medical Sciences": [
        "oncology", "radiology", "pathology", "neurology", "cardiology",
        "immunotherapy", "pharmacology", "genomics", "biomarkers",
        "epidemiology", "diagnostics", "MRI", "CT imaging", "ultrasound",
        "clinical trials", "pediatrics", "geriatrics", "endocrinology",
        "hematology", "psychiatry", "infectious diseases",
        "metabolic disorders", "public health", "surgery",
        "therapeutic efficacy", "neuroimaging", "rehabilitation",
        "drug delivery", "gene therapy", "respiratory function",
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
        "soil microbiology", "conservation",
    ],
    "Arts & Humanities": [
        "semiotics", "hermeneutics", "classical literature", "syntax",
        "narrative theory", "aesthetics", "discourse analysis", "pragmatics",
        "rhetoric", "historiography", "translation studies", "morphology",
        "phonology", "semantics", "textual criticism", "art history",
        "visual culture", "anthropology", "film theory", "media aesthetics",
        "iconography", "ethnography", "literature", "symbolism",
        "poetics", "dramaturgy", "lexicography", "epigraphy",
        "folklore", "historical linguistics",
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
        "labour markets", "institutional economics", "social research",
    ],
}

# ====================================================================================================================================================================================
                                                                # 2. DEPARTMENT KEYWORDS
# ====================================================================================================================================================================================

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

# ====================================================================================================================================================================================
                                                              # 3. THREE-LEVEL DOMAIN HIERARCHY
# ====================================================================================================================================================================================

DEPARTMENTS = {
    "Computer Science": ("Core Computing", "Computer Science"),
    "Information Technology": ("Data & Applied Computing", "Computer Science"),
    "Data Science": ("Data & Applied Computing", "Computer Science"),
    "AI & Machine Learning": ("Data & Applied Computing", "Computer Science"),

    "Mechanical Engineering": ("Mechanical & Manufacturing", "Engineering"),
    "Electrical Engineering": ("Electrical & Electronics", "Engineering"),
    "Civil Engineering": ("Civil & Infrastructure", "Engineering"),
    "Electronics & Communication Engineering": ("Electrical & Electronics", "Engineering"),
    "Automobile Engineering": ("Mechanical & Manufacturing", "Engineering"),

    "Mathematics": ("Pure Mathematics", "Mathematics"),
    "Statistics": ("Statistics & Applied Maths", "Mathematics"),

    "Physics": ("Fundamental & Applied Physics", "Physics"),
    "Applied Physics": ("Fundamental & Applied Physics", "Physics"),
    "Astrophysics": ("Astrophysics", "Physics"),

    "Chemistry": ("Physical & Organic Chemistry", "Chemistry"),
    "Organic Chemistry": ("Physical & Organic Chemistry", "Chemistry"),
    "Materials Chemistry": ("Materials Science", "Chemistry"),

    "Biotechnology": ("Biotechnology & Genetics", "Biology"),
    "Microbiology": ("Microbiology", "Biology"),
    "Life Sciences": ("Life Sciences", "Biology"),
    "Genetics": ("Biotechnology & Genetics", "Biology"),

    "Medicine": ("Clinical Medicine", "Medical Sciences"),
    "Nursing": ("Healthcare & Allied", "Medical Sciences"),
    "Physiotherapy": ("Healthcare & Allied", "Medical Sciences"),
    "Pharmacy": ("Pharmaceutical Sciences", "Medical Sciences"),

    "Agriculture": ("Agricultural Sciences", "Agriculture & Environment"),
    "Environmental Science": ("Environmental Sciences", "Agriculture & Environment"),
    "Horticulture": ("Agricultural Sciences", "Agriculture & Environment"),

    "English": ("Arts & Humanities", "Arts & Humanities"),
    "History": ("Arts & Humanities", "Arts & Humanities"),
    "Philosophy": ("Arts & Humanities", "Arts & Humanities"),

    "Sociology": ("Social Sciences", "Social Sciences"),
    "Political Science": ("Social Sciences", "Social Sciences"),
    "Public Administration": ("Social Sciences", "Social Sciences"),
}

DEPT_TO_CLUSTER = {k: v[0] for k, v in DEPARTMENTS.items()}
DEPT_TO_BROAD = {k: v[1] for k, v in DEPARTMENTS.items()}

# ====================================================================================================================================================================================
                                                                    # 4. CONSTANTS
# ====================================================================================================================================================================================

UNIVERSITIES = [
    "IIT Madras", "IIT Delhi", "IIT Bombay", "IIT Kanpur", "IIT Kharagpur",
    "NIT Trichy", "Anna University", "IISc Bangalore",
    "University of Hyderabad", "JNU Delhi", "Pondicherry University",
]

ROLES = ["Professor", "Associate Professor", "Assistant Professor", "Research Scholar"]

ROLE_WEIGHTS = [0.20, 0.25, 0.30, 0.25]

ROLE_H_INDEX_RANGE = {
    "Professor": (15, 40),
    "Associate Professor": (8, 17),
    "Assistant Professor": (2, 12),
    "Research Scholar": (0, 8),
}

ROLE_CAREER_START_RANGE = {
    "Professor": (1985, 2005),
    "Associate Professor": (1995, 2012),
    "Assistant Professor": (2005, 2018),
    "Research Scholar": (2015, 2023),
}

CURRENT_YEAR = 2025
NUM_RESEARCHERS = 3000

DEPT_TO_ID = {dept: f"D{i:03d}" for i, dept in enumerate(DEPARTMENTS.keys(), start=1)}
ALL_DEPTS = list(DEPARTMENTS.keys())

ALL_DOMAINS = [
    "Computer Science",
    "Engineering",
    "Mathematics",
    "Physics",
    "Chemistry",
    "Biology",
    "Medical Sciences",
    "Agriculture & Environment",
    "Arts & Humanities",
    "Social Sciences",
]

K = len(ALL_DOMAINS)
H_MAX = math.log(K)

# ====================================================================================================================================================================================
                                                                        # 5. KEYWORD TO DOMAIN MAP
# ====================================================================================================================================================================================

KEYWORD_TO_DOMAIN = {}

for dom, kws in DOMAIN_KEYWORDS.items():
    for kw in kws:
        KEYWORD_TO_DOMAIN.setdefault(kw.strip().lower(), set()).add(dom)

for dept, kws in DEPARTMENT_KEYWORDS.items():
    broad_domain = DEPT_TO_BROAD[dept]
    for kw in kws:
        KEYWORD_TO_DOMAIN.setdefault(kw.strip().lower(), set()).add(broad_domain)

# ====================================================================================================================================================================================
                                                          # 6. REALISTIC DOMAIN DRIFT RULES
# ====================================================================================================================================================================================

REALISTIC_BROAD_DOMAIN_NEIGHBORS = {
    "Computer Science": [
        "Engineering",
        "Mathematics",
        "Medical Sciences",
        "Biology",
        "Agriculture & Environment",
        "Social Sciences",
    ],
    "Engineering": [
        "Computer Science",
        "Physics",
        "Chemistry",
        "Agriculture & Environment",
        "Medical Sciences",
    ],
    "Mathematics": [
        "Computer Science",
        "Physics",
        "Engineering",
        "Social Sciences",
        "Biology",
    ],
    "Physics": [
        "Mathematics",
        "Engineering",
        "Chemistry",
        "Computer Science",
    ],
    "Chemistry": [
        "Biology",
        "Medical Sciences",
        "Physics",
        "Engineering",
        "Agriculture & Environment",
    ],
    "Biology": [
        "Medical Sciences",
        "Chemistry",
        "Agriculture & Environment",
        "Computer Science",
    ],
    "Medical Sciences": [
        "Biology",
        "Chemistry",
        "Computer Science",
        "Social Sciences",
    ],
    "Agriculture & Environment": [
        "Biology",
        "Chemistry",
        "Engineering",
        "Computer Science",
        "Social Sciences",
    ],
    "Arts & Humanities": [
        "Social Sciences",
        "Computer Science",
    ],
    "Social Sciences": [
        "Arts & Humanities",
        "Computer Science",
        "Medical Sciences",
        "Agriculture & Environment",
        "Mathematics",
    ],
}

BLOCKED_BROAD_DOMAIN_PAIRS = {
    tuple(sorted(("Arts & Humanities", "Physics"))),
    tuple(sorted(("Arts & Humanities", "Chemistry"))),
    tuple(sorted(("Arts & Humanities", "Engineering"))),
    tuple(sorted(("Arts & Humanities", "Medical Sciences"))),
    tuple(sorted(("Physics", "Social Sciences"))),
    tuple(sorted(("Chemistry", "Social Sciences"))),
}

ROLE_DRIFT_PROB = {
    "Professor": 0.45,
    "Associate Professor": 0.35,
    "Assistant Professor": 0.22,
    "Research Scholar": 0.08,
}

ROLE_DRIFT_STRENGTH_RANGE = {
    "Professor": (0.45, 0.75),
    "Associate Professor": (0.35, 0.65),
    "Assistant Professor": (0.20, 0.50),
    "Research Scholar": (0.05, 0.25),
}

# ====================================================================================================================================================================================
                                                                    # 7. HELPER FUNCTIONS
# ====================================================================================================================================================================================

def generate_name():
    return fake.name()


def generate_orcid():
    return "0000-0002-" + f"{random.randint(1000, 9999)}-{random.randint(1000, 9999)}"


def dept_keyword_pool(d_name):
    broad_domain = DEPT_TO_BROAD[d_name]
    dept_kws = DEPARTMENT_KEYWORDS.get(d_name, [])
    domain_kws = DOMAIN_KEYWORDS.get(broad_domain, [])
    return list(dict.fromkeys(dept_kws + domain_kws))


def keyword_domains(keyword):
    return KEYWORD_TO_DOMAIN.get(str(keyword).strip().lower(), set())


def compute_topic_vector(keywords, primary_domain):
    vec = np.zeros(len(ALL_DOMAINS), dtype=float)

    for kw in keywords:
        domains = keyword_domains(kw)

        if not domains:
            if primary_domain in ALL_DOMAINS:
                vec[ALL_DOMAINS.index(primary_domain)] += 1
        else:
            for d in domains:
                if d in ALL_DOMAINS:
                    vec[ALL_DOMAINS.index(d)] += 1 / len(domains)

    if vec.sum() > 0:
        vec = vec / vec.sum()

    return [round(float(x), 3) for x in vec]


def choose_realistic_target_domain(original_domain):
    candidates = REALISTIC_BROAD_DOMAIN_NEIGHBORS.get(original_domain, [])

    valid = []

    for d in candidates:
        pair = tuple(sorted((original_domain, d)))

        if pair not in BLOCKED_BROAD_DOMAIN_PAIRS and d in ALL_DOMAINS:
            valid.append(d)

    if not valid:
        return None

    return random.choice(valid)


def make_domain_vector_from_weights(weight_dict):
    vec = []

    for d in ALL_DOMAINS:
        vec.append(float(weight_dict.get(d, 0.0)))

    total = sum(vec)

    if total > 0:
        vec = [round(v / total, 3) for v in vec]
    else:
        vec = [0.0 for _ in vec]

    return vec


def drift_stage_from_jsd(jsd):
    """Derive drift stage from normalised JSD score [0, 1]."""
    if jsd <= 0.05:
        return "stable"
    elif jsd <= 0.20:
        return "exploratory"
    elif jsd <= 0.40:
        return "transitioning"
    elif jsd <= 0.60:
        return "shifted"
    else:
        return "strong_drift"


def generate_domain_evolution(
    r_id,
    role,
    original_domain,
    career_start_year,
    career_end_year
):
    if pd.isna(career_end_year):
        career_end_year = CURRENT_YEAR

    career_start_year = int(career_start_year)
    career_end_year = int(career_end_year)

    if career_end_year < career_start_year:
        career_end_year = career_start_year

    years = list(range(career_start_year, career_end_year + 1))

    will_drift = random.random() < ROLE_DRIFT_PROB.get(role, 0.20)
    target_domain = choose_realistic_target_domain(original_domain) if will_drift else None

    if target_domain is None or len(years) <= 2:
        rows = []

        for y in years:
            weight_dict = {original_domain: 1.0}

            rows.append({
                "year": y,
                "r_id": r_id,
                "original_primary_domain": original_domain,
                "current_dominant_domain": original_domain,
                "target_drift_domain": "",
                "original_domain_weight": 1.0,
                "target_domain_weight": 0.0,
                "domain_vector": make_domain_vector_from_weights(weight_dict),
            })

        return rows, None, 0.0

    lo, hi = ROLE_DRIFT_STRENGTH_RANGE.get(role, (0.20, 0.50))
    final_drift_score = round(random.uniform(lo, hi), 3)

    n = len(years)

    drift_start_index = max(1, int(n * random.uniform(0.30, 0.50)))

    rows = []

    for idx, y in enumerate(years):
        if idx < drift_start_index:
            target_weight = 0.0
        else:
            progress = (idx - drift_start_index) / max(1, (n - 1 - drift_start_index))
            target_weight = final_drift_score * (progress ** 1.2)

        target_weight = round(float(target_weight), 3)
        original_weight = round(1.0 - target_weight, 3)

        weight_dict = {
            original_domain: original_weight,
            target_domain: target_weight,
        }

        current_domain = target_domain if target_weight > original_weight else original_domain

        rows.append({
            "year": y,
            "r_id": r_id,
            "original_primary_domain": original_domain,
            "current_dominant_domain": current_domain,
            "target_drift_domain": target_domain,
            "original_domain_weight": original_weight,
            "target_domain_weight": target_weight,
            "domain_vector": make_domain_vector_from_weights(weight_dict),
        })

    return rows, target_domain, final_drift_score


def sample_author_keywords(
    d_name,
    primary_domain,
    drift_domain=None,
    final_drift_score=0.0
):
    total_k = random.randint(3, 6)

    base_pool = dept_keyword_pool(d_name)

    if not base_pool:
        base_pool = DOMAIN_KEYWORDS.get(primary_domain, [])

    if drift_domain is None or final_drift_score < 0.10:
        return random.sample(base_pool, min(total_k, len(base_pool)))

    target_pool = DOMAIN_KEYWORDS.get(drift_domain, [])

    if not target_pool:
        return random.sample(base_pool, min(total_k, len(base_pool)))

    if final_drift_score >= 0.60:
        drift_k = random.randint(2, min(3, total_k - 1))
    elif final_drift_score >= 0.35:
        drift_k = random.randint(1, min(2, total_k - 1))
    else:
        drift_k = 1

    base_k = total_k - drift_k

    base_sample = random.sample(base_pool, min(base_k, len(base_pool)))
    drift_sample = random.sample(target_pool, min(drift_k, len(target_pool)))

    keywords = base_sample + drift_sample
    random.shuffle(keywords)

    return keywords

# ====================================================================================================================================================================================
                                                                # 8. GENERATE RESEARCHER PROFILE
# ====================================================================================================================================================================================

rows = []
domain_evolution_rows = []

for i in range(1, NUM_RESEARCHERS + 1):

    r_id = f"R{i:04d}"

    d_name = random.choice(ALL_DEPTS)
    d_id = DEPT_TO_ID[d_name]

    discipline_cluster = DEPT_TO_CLUSTER[d_name]
    primary_domain = DEPT_TO_BROAD[d_name]

    role = random.choices(ROLES, weights=ROLE_WEIGHTS, k=1)[0]

    name = generate_name()

    h_low, h_high = ROLE_H_INDEX_RANGE[role]
    h_index = random.randint(h_low, h_high)

    y_low, y_high = ROLE_CAREER_START_RANGE[role]
    career_start_year = random.randint(y_low, y_high)

    is_active = random.choices([1, 0], weights=[0.82, 0.18], k=1)[0]

    if is_active == 1:
        career_end_date = pd.NA
        career_end_year_for_evolution = CURRENT_YEAR
        years_exp = CURRENT_YEAR - career_start_year
    else:
        min_end = min(career_start_year + 1, CURRENT_YEAR)
        max_end = CURRENT_YEAR

        if min_end > max_end:
            career_end_year = CURRENT_YEAR
        else:
            career_end_year = random.randint(min_end, max_end)

        career_end_date = career_end_year
        career_end_year_for_evolution = career_end_year
        years_exp = career_end_year - career_start_year

    evolution_rows, drift_domain, final_drift_score = generate_domain_evolution(
        r_id=r_id,
        role=role,
        original_domain=primary_domain,
        career_start_year=career_start_year,
        career_end_year=career_end_year_for_evolution,
    )

    domain_evolution_rows.extend(evolution_rows)

    keywords = sample_author_keywords(
        d_name=d_name,
        primary_domain=primary_domain,
        drift_domain=drift_domain,
        final_drift_score=final_drift_score,
    )

    author_keywords = "; ".join(keywords)

    topic_vector = compute_topic_vector(keywords, primary_domain)

    orcid = generate_orcid()
    email = fake.email()
    affiliation = random.choice(UNIVERSITIES)

    rows.append({
        "r_id": r_id,
        "name": name,
        "d_id": d_id,
        "d_name": d_name,
        "role": role,
        "primary_domain": primary_domain,
        "author_keywords": author_keywords,
        "h_index": h_index,
        "career_start_year": career_start_year,
        "is_active": is_active,
        "career_end_date": career_end_date,
        "years_exp": years_exp,
        "orcid": orcid,
        "email": email,
        "affiliation": affiliation,
        "topic_vector": topic_vector,
    })

# ====================================================================================================================================================================================
                                                                          # 9. JSD DRIFT SCORE
# ====================================================================================================================================================================================
# For each researcher, JSD is measured between the year-1 baseline
# domain_vector and every subsequent year's vector.
# Normalised to [0, 1] by dividing by ln(2).
# drift_stage is derived from jsd_drift_score.

_LOG2 = math.log(2)


def _jsd_normalised(p, q):
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    p = p / p.sum() if p.sum() > 0 else np.ones(len(p)) / len(p)
    q = q / q.sum() if q.sum() > 0 else np.ones(len(q)) / len(q)
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
    raw = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return round(raw / _LOG2, 6)


def _compute_jsd_for_evolution(evo_rows):
    """
    Adds jsd_drift_score and drift_stage to each row in-place.
    Baseline is the first year row per researcher.
    """
    from collections import defaultdict
    grouped = defaultdict(list)
    for row in evo_rows:
        grouped[row["r_id"]].append(row)

    for r_id, grp in grouped.items():
        grp.sort(key=lambda r: r["year"])
        baseline_vec = None
        for row in grp:
            vec = row["domain_vector"]
            if baseline_vec is None:
                baseline_vec = vec
                row["jsd_drift_score"] = 0.0
            else:
                row["jsd_drift_score"] = _jsd_normalised(baseline_vec, vec)
            row["drift_stage"] = drift_stage_from_jsd(row["jsd_drift_score"])


_compute_jsd_for_evolution(domain_evolution_rows)

# ====================================================================================================================================================================================
                                                                          # 10. SAVE OUTPUTS
# ====================================================================================================================================================================================

df_profiles = pd.DataFrame(rows)
df_domain_evolution = pd.DataFrame(domain_evolution_rows)

df_profiles["topic_vector"] = df_profiles["topic_vector"].apply(
    lambda x: ";".join(map(str, x))
)

df_domain_evolution["domain_vector"] = df_domain_evolution["domain_vector"].apply(
    lambda x: ";".join(map(str, x))
)

df_profiles.to_csv(
    "researchers_profile.csv",
    index=False,
    quoting=csv.QUOTE_MINIMAL
)

df_domain_evolution.to_csv(
    "researcher_domain_evolution.csv",
    index=False,
    quoting=csv.QUOTE_MINIMAL
)

print("Done")
