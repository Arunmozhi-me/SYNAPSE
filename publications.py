import bisect
import csv
import random
import re

import numpy as np
import pandas as pd

# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                      # 1) PATHS
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

PROJECTS_CSV  = "Project_with_grant.csv"
GRANTS_CSV    = "Grants.csv"
PROFILE_CSV   = "researchers_profile.csv"
EVOLUTION_CSV = "researcher_domain_evolution.csv"
OUT_PUB_CSV   = "Publications.csv"

# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                      # 2) SETTINGS
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MIN_PUB_PER_PROJECT = 1
MAX_PUB_PER_PROJECT = 4

PUB_PREFIX = "PUB-"
DOI_PREFIX = "10.12345/"

EMERGENCY_YEAR_MIN = 2000
EMERGENCY_YEAR_MAX = 2025

# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                      # 3) HELPERS
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

def norm_str(x) -> str:
    return "" if pd.isna(x) else str(x).strip()

def normalize_rid(x) -> str:
    s = norm_str(x).upper()
    if s in {"", "NAN", "NONE"}:
        return ""
    m = re.match(r"^R(\d+)$", s)
    return "R" + m.group(1).zfill(4) if m else s


def split_ids(cell) -> list[str]:
    s = norm_str(cell)
    if not s:
        return []
    out, seen = [], set()
    for p in re.split(r"[;,|\s]+", s):
        rid = normalize_rid(p)
        if rid and rid not in seen:
            out.append(rid)
            seen.add(rid)
    return out


def split_domains(cell: str) -> list[str]:
    s = norm_str(cell)
    if not s:
        return []
    seen, out = set(), []
    for p in re.split(r"\+|,|;|\|", s):
        p = p.strip()
        if p and p not in seen:
            out.append(p)
            seen.add(p)
    return out[:2]


def parse_year(x) -> int | None:
    m = re.search(r"\b(\d{4})\b", norm_str(x))
    if m:
        y = int(m.group(1))
        if 1900 <= y <= 2100:
            return y
    return None


def make_doi(pub_id: str) -> str:
    return DOI_PREFIX + pub_id.replace("-", "").lower()


def find_col(df, candidates) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def collect_years_from_df(df, start_candidates, end_candidates) -> list[int]:
    years = []
    for candidates in (start_candidates, end_candidates):
        col = find_col(df, candidates)
        if col is not None:
            extracted = (
                df[col].astype(str)
                .str.extract(r"\b(\d{4})\b")[0]
                .dropna()
                .astype(int)
            )
            years.extend(extracted[(extracted >= 1900) & (extracted <= 2100)].tolist())
    return years

# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                            # 4) DOMAIN CONTENT BANKS  
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

DOMAIN_CONTENT = {
    "Computer Science": {
        "journals": [
            "Journal of Machine Learning Research", "Pattern Recognition",
            "Information Sciences", "IEEE Internet of Things Journal",
            "Expert Systems with Applications",
        ],
        "keywords": [
            "deep learning; neural networks; classification; feature extraction",
            "federated learning; privacy; distributed optimization; edge computing",
            "graph neural networks; knowledge graph; link prediction; embedding",
            "natural language processing; transformer; attention mechanism; BERT",
            "anomaly detection; cybersecurity; network traffic; intrusion detection",
        ],
        "if_range": (2.5, 12.0), "cite_range": (5, 180),
        "suffixes": [": An Empirical Evaluation", ": Scalability and Performance Analysis",
                     ": Design, Implementation, and Evaluation"],
    },
    "Engineering": {
        "journals": [
            "Engineering Structures", "Applied Energy",
            "Mechanical Systems and Signal Processing", "Renewable Energy",
            "IEEE Transactions on Industrial Electronics",
        ],
        "keywords": [
            "structural health monitoring; fatigue; composite; finite element analysis",
            "smart grid; renewable energy; power electronics; microgrid",
            "vibration analysis; signal processing; condition monitoring; fault diagnosis",
            "robotics; control systems; autonomous navigation; UAV",
        ],
        "if_range": (2.0, 9.0), "cite_range": (4, 120),
        "suffixes": [": Experimental Validation", ": Modelling and Simulation", ": Design and Testing"],
    },
    "Mathematics": {
        "journals": [
            "Applied Mathematics and Computation", "Linear Algebra and its Applications",
            "SIAM Journal on Numerical Analysis", "Operations Research",
        ],
        "keywords": [
            "stochastic differential equations; Markov chains; convergence; stability",
            "graph theory; combinatorial optimization; network flows",
            "Bayesian inference; MCMC; probabilistic models; posterior estimation",
            "game theory; mechanism design; Nash equilibrium; optimization",
        ],
        "if_range": (1.2, 5.0), "cite_range": (2, 80),
        "suffixes": [": A Rigorous Analysis", ": Theory and Algorithms", ": Bounds and Approximations"],
    },
    "Physics": {
        "journals": [
            "Physical Review Letters", "Physical Review B",
            "Nature Physics", "Applied Physics Letters",
        ],
        "keywords": [
            "quantum transport; condensed matter; spin-orbit coupling",
            "superconductivity; phase transition; lattice dynamics",
            "plasma physics; magnetohydrodynamics; instabilities",
            "photonics; optical trapping; laser cooling",
        ],
        "if_range": (3.0, 20.0), "cite_range": (10, 300),
        "suffixes": [": Experimental Evidence", ": Theory and Experiment", ": Monte Carlo Investigation"],
    },
    "Chemistry": {
        "journals": [
            "Journal of the American Chemical Society", "Chemical Science",
            "ACS Catalysis", "Green Chemistry", "Chemical Engineering Journal",
        ],
        "keywords": [
            "catalysis; reaction mechanism; transition metal; synthesis",
            "green chemistry; solvent-free; sustainable synthesis",
            "nanoparticles; surface chemistry; catalytic activity",
            "drug delivery; nanocarrier; release kinetics; biocompatibility",
        ],
        "if_range": (3.5, 18.0), "cite_range": (8, 250),
        "suffixes": [": Synthesis, Characterisation, and Application",
                     ": Kinetics and Mechanism", ": Computational and Experimental Study"],
    },
    "Biology": {
        "journals": [
            "PLOS Biology", "Genome Biology", "Molecular Biology and Evolution",
            "Frontiers in Microbiology", "BMC Genomics",
        ],
        "keywords": [
            "CRISPR; gene editing; genome; functional genomics",
            "transcriptomics; RNA-seq; differential expression",
            "microbiome; metagenomics; gut bacteria; diversity",
            "ecology; biodiversity; species distribution; adaptation",
        ],
        "if_range": (3.0, 14.0), "cite_range": (8, 200),
        "suffixes": [": Mechanistic Insights", ": A Genome-Wide Study", ": Transcriptomic Evidence"],
    },
    "Medical Sciences": {
        "journals": [
            "The Lancet", "British Medical Journal", "Journal of Clinical Oncology",
            "PLOS Medicine", "Radiology",
        ],
        "keywords": [
            "clinical trial; efficacy; adverse events; placebo",
            "biomarkers; early detection; sensitivity; specificity",
            "medical imaging; MRI; diagnostic accuracy; radiology",
            "epidemiology; incidence; prevalence; risk factors",
        ],
        "if_range": (4.0, 25.0), "cite_range": (10, 350),
        "suffixes": [": A Retrospective Cohort Study", ": Findings from a Multicentre Study",
                     ": A Population-Based Analysis"],
    },
    "Agriculture & Environment": {
        "journals": [
            "Field Crops Research", "Agricultural Water Management",
            "Science of the Total Environment", "Environmental Pollution",
            "Remote Sensing of Environment",
        ],
        "keywords": [
            "drought tolerance; crop yield; water deficit; stomatal conductance",
            "soil carbon; organic matter; carbon sequestration; microbial biomass",
            "remote sensing; NDVI; satellite imagery; precision agriculture",
            "food security; crop diversity; climate change adaptation",
        ],
        "if_range": (2.0, 8.0), "cite_range": (5, 130),
        "suffixes": [": Field Trial Results", ": A Multi-Season Study",
                     ": Implications for Sustainable Management"],
    },
    "Arts & Humanities": {
        "journals": [
            "Journal of Cultural Heritage", "Digital Humanities Quarterly",
            "Language and Literature", "Heritage Science",
        ],
        "keywords": [
            "postcolonial; diaspora; identity; cultural memory",
            "manuscript; digital archive; preservation; heritage",
            "translation; equivalence; cross-cultural; literary adaptation",
            "historiography; collective memory; nation; narrative",
        ],
        "if_range": (0.5, 3.0), "cite_range": (1, 40),
        "suffixes": [": A Critical Reading", ": Historical and Contemporary Perspectives",
                     ": An Ethnographic Account"],
    },
    "Social Sciences": {
        "journals": [
            "World Development", "Social Science and Medicine", "Political Behavior",
            "Urban Studies", "Governance",
        ],
        "keywords": [
            "poverty; inequality; income; social mobility; welfare",
            "policy evaluation; impact assessment; causal inference",
            "urbanisation; migration; housing; labour market",
            "social networks; community; collective action; social capital",
        ],
        "if_range": (1.5, 8.0), "cite_range": (4, 150),
        "suffixes": [": Evidence from a Panel Study", ": A Mixed-Methods Approach",
                     ": Insights from Primary Data"],
    },
}

FALLBACK_CONTENT = {
    "journals": ["PLOS ONE", "Scientific Reports", "Nature Communications",
                 "Royal Society Open Science"],
    "keywords": [
        "interdisciplinary; convergence; methodology; mixed methods",
        "multi-domain; systems approach; integration; collaboration",
    ],
    "if_range": (2.0, 10.0), "cite_range": (5, 120),
    "suffixes": [": An Interdisciplinary Perspective", ": Methods, Results, and Implications"],
}

# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                  # 5) DOMAIN ASSIGNMENT  
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

def assign_pub_domain(domains: list, pub_index: int) -> str:
    return "+".join(domains[:2]) if domains else ""


def get_content(domain: str) -> dict:
    if domain in DOMAIN_CONTENT:
        return DOMAIN_CONTENT[domain]
    first = domain.split("+")[0].strip()
    return DOMAIN_CONTENT.get(first, FALLBACK_CONTENT)

# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                    # 6) TITLE GENERATION 
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

CONNECTIVES = re.compile(
    r"\b(for|of|in|via|using|with|and|under|on|from|towards|based on|"
    r"applied to|approach to|methods for|study of|analysis of|modelling of|"
    r"detection of|assessment of|evaluation of|prediction of|estimation of)\b",
    re.IGNORECASE,
)

RECOMBINATION_PATTERNS = {
    "Computer Science": [
        "{p1}: A {p0} Approach", "Towards {p1} via {p0}",
        "Efficient {p0} for {p1}", "{p1} Using {p0}: {suffix}",
    ],
    "Engineering": [
        "Experimental Analysis of {p1} Using {p0}", "{p0} for {p1}: {suffix}",
        "Design and Validation of {p0} for {p1}",
    ],
    "Mathematics": [
        "{p0} Applied to {p1}: {suffix}", "Convergence of {p0} for {p1}",
        "A {p0} Framework for {p1}",
    ],
    "Physics": [
        "{p0} Study of {p1}: {suffix}", "Spectroscopic Investigation of {p1} Using {p0}",
        "Phase Behaviour of {p1}: {suffix}",
    ],
    "Chemistry": [
        "Synthesis and Characterisation of {p1} via {p0}",
        "Mechanistic Study of {p1} Using {p0}",
        "{p0}-Driven Synthesis of {p1}: {suffix}",
    ],
    "Biology": [
        "Molecular Mechanisms of {p1} in {p2}", "Genome-Wide Analysis of {p1} via {p0}",
        "{p0}-Mediated Regulation of {p1}",
    ],
    "Medical Sciences": [
        "Clinical Outcomes of {p1} in {p2}", "Diagnostic Accuracy of {p0} for {p1}",
        "{p0}-Based {p1}: {suffix}",
    ],
    "Agriculture & Environment": [
        "Effect of {p0} on {p1} in {p2}", "{p0} for {p1}: {suffix}",
        "Management of {p1} Through {p0}: {suffix}",
    ],
    "Arts & Humanities": [
        "{p0} and {p1}: {suffix}", "A Critical Reading of {p1} Through {p0}",
        "Representation of {p1} in {p0}",
    ],
    "Social Sciences": [
        "Impact of {p0} on {p1}: {suffix}", "{p1} in {p2}: {suffix}",
        "Policy Implications of {p0} for {p1}",
    ],
}

FALLBACK_PATTERNS = [
    "{p0} and {p1}: An Interdisciplinary Study",
    "Integrated {p0} for {p1} in {p2}",
    "{p1} via {p0}: Methods and Implications",
]


def parse_title_phrases(project_title: str) -> list[str]:
    parts   = CONNECTIVES.split(project_title)
    phrases = [p.strip(" :-–—") for p in parts
               if len(p.strip(" :-–—")) > 3 and not CONNECTIVES.fullmatch(p.strip())]
    return phrases or [project_title]


def make_pub_title(project_title: str, domains: list, pub_index: int) -> str:
    primary = domains[0] if domains else ""
    content = get_content(primary)
    suffix  = random.choice(content.get("suffixes", FALLBACK_CONTENT["suffixes"]))
    phrases = parse_title_phrases(project_title)
    while len(phrases) < 3:
        phrases.append(phrases[-1])
    p0, p1, p2 = phrases[0].strip(), phrases[1].strip(), phrases[2].strip()
    patterns   = RECOMBINATION_PATTERNS.get(primary, FALLBACK_PATTERNS)
    title      = patterns[pub_index % len(patterns)].format(p0=p0, p1=p1, p2=p2, suffix=suffix)
    title      = re.sub(r"\s{2,}", " ", title).strip()
    return (title[0].upper() + title[1:]) if title else title

# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                              # 7) NUMERIC HELPERS       
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

def make_volume(year: int, max_year: int) -> int:
    base = random.randint(25, 55)
    return max(1, year - (max_year - base))


def make_issue(domain: str) -> int:
    first = domain.split("+")[0].strip()
    return random.randint(1, 4) if first in {"Arts & Humanities", "Mathematics"} else random.randint(1, 12)


def make_citation_count(year: int, content: dict, max_year: int) -> int:
    lo, hi     = content["cite_range"]
    age        = max(0, max_year - year)
    age_factor = min(1.0, 0.25 + age * 0.12)
    return random.randint(lo, max(lo + 1, int(hi * age_factor)))


def make_impact_factor(content: dict) -> float:
    lo, hi = content["if_range"]
    return round(random.uniform(lo, hi), 3)

# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                # 8) LOAD PROJECTS
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

print("Loading projects ...")
df = pd.read_csv(PROJECTS_CSV)

COL_PROJECT_ID = "project_id"
COL_GRANT_ID   = "grant_id"
COL_TITLE      = "project_title"
COL_DOMAIN     = "domain"
COL_SDATE      = "start_date"
COL_EDATE      = "end_date"
COL_PI         = "principal_investigator"
COL_COPI       = "co_investigators"

for c in [COL_PROJECT_ID, COL_GRANT_ID, COL_TITLE, COL_DOMAIN,
          COL_SDATE, COL_EDATE, COL_PI, COL_COPI]:
    if c not in df.columns:
        df[c] = ""

df[COL_PROJECT_ID] = df[COL_PROJECT_ID].astype(str).str.strip()
df[COL_GRANT_ID]   = df[COL_GRANT_ID].astype(str).str.strip()
df[COL_TITLE]      = df[COL_TITLE].astype(str).str.strip()
df[COL_DOMAIN]     = df[COL_DOMAIN].astype(str).str.strip()


df[COL_PI]   = df[COL_PI].apply(normalize_rid)
df[COL_COPI] = df[COL_COPI].apply(lambda x: split_ids(x) if not isinstance(x, list) else x)

print(f"  {len(df)} projects loaded")

# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                  # 9) LOAD GRANTS — vectorized year lookup
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

df_grants        = None
grant_year_lookup: dict[str, tuple[int | None, int | None]] = {}

try:
    df_grants = pd.read_csv(GRANTS_CSV)

    g_pid_col   = find_col(df_grants, ["project_id", "Project_ID"])
    g_start_col = find_col(df_grants, ["start_date", "Start_Date"])
    g_end_col   = find_col(df_grants, ["end_date",   "End_Date"])

    if g_pid_col:
        df_grants[g_pid_col] = df_grants[g_pid_col].astype(str).str.strip()

  
    # (grant_id key removed — project_id is the join key used later)
    if g_pid_col and g_start_col and g_end_col:
        tmp = df_grants[[g_pid_col, g_start_col, g_end_col]].copy()
        tmp["_y1"] = tmp[g_start_col].astype(str).str.extract(r"\b(\d{4})\b")[0].astype(float).astype("Int64")
        tmp["_y2"] = tmp[g_end_col  ].astype(str).str.extract(r"\b(\d{4})\b")[0].astype(float).astype("Int64")
        # last grant per project wins (projects have at most one grant here)
        for _, g in tmp.iterrows():
            pid = norm_str(g[g_pid_col])
            if pid:
                y1 = int(g["_y1"]) if pd.notna(g["_y1"]) else None
                y2 = int(g["_y2"]) if pd.notna(g["_y2"]) else None
                grant_year_lookup[pid] = (y1, y2)

    print(f"  {len(df_grants)} grants loaded")
    print(f"  {len(grant_year_lookup)} project→grant year mappings")

except Exception as e:
    print(f"  Grants file not loaded — grant year validation OFF\n  Reason: {e}")

# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                          # 10) LOAD RESEARCHER PROFILE
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

profile_ids: set[str] | None = None

try:
    prof    = pd.read_csv(PROFILE_CSV)
    rid_col = next((c for c in prof.columns if "r_id" in c.lower()), prof.columns[0])
    # FIX 1: normalize_rid returns "" not None; filter empties
    prof[rid_col] = prof[rid_col].apply(normalize_rid)
    profile_ids   = set(prof[rid_col][prof[rid_col] != ""])
    print(f"  {len(profile_ids)} researchers in profile")
except Exception as e:
    print(f"  Profile not loaded — author validation OFF\n  Reason: {e}")

# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                          # 11) LOAD TEMPORAL EVOLUTION
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

dev              = None
evo_year_by_rid: dict[str, list[int]] = {}   # rid → sorted list of available years
evo_domain_lookup: dict[tuple[str, int], str] = {}
EVO_YEAR_MIN: int | None = None
EVO_YEAR_MAX: int | None = None

try:
    dev = pd.read_csv(EVOLUTION_CSV)
    # FIX 1: normalize_rid returns ""
    dev["r_id"]                    = dev["r_id"].apply(normalize_rid)
    dev["year"]                    = dev["year"].astype(int)
    dev["current_dominant_domain"] = dev["current_dominant_domain"].astype(str).str.strip()
    dev = dev[dev["r_id"] != ""]

   
    for rid, grp in dev.groupby("r_id"):
        evo_year_by_rid[rid] = sorted(grp["year"].tolist())

    # OPT: vectorized dict build
    evo_domain_lookup = (
        dev.set_index(["r_id", "year"])["current_dominant_domain"].to_dict()
    )

    EVO_YEAR_MIN = int(dev["year"].min())
    EVO_YEAR_MAX = int(dev["year"].max())

    print(f"  {len(dev)} researcher-year evolution rows loaded")
    print(f"  Evolution years: {EVO_YEAR_MIN}–{EVO_YEAR_MAX}")

except Exception as e:
    print(f"  Evolution file not loaded — temporal author validation OFF\n  Reason: {e}")

# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                # 12) DERIVE GLOBAL YEAR RANGE DYNAMICALLY
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════


all_years = collect_years_from_df(df, ["Start_Date", "start_date"], ["End_Date", "end_date"])

if df_grants is not None:
    all_years += collect_years_from_df(
        df_grants, ["Start_Date", "start_date"], ["End_Date", "end_date"]
    )

if dev is not None and "year" in dev.columns:
    all_years += dev["year"].dropna().astype(int).tolist()


# Publications cannot appear in the future — ceiling is always today's year.
TODAY_YEAR = pd.Timestamp.today().year

if all_years:
    DEFAULT_YEAR_MIN = min(all_years)
    # Cap at TODAY_YEAR: projects/grants may end in 2028 but we cannot
    # publish in a year that hasn't happened yet.
    DEFAULT_YEAR_MAX = min(max(all_years), TODAY_YEAR)
else:
    DEFAULT_YEAR_MIN = EMERGENCY_YEAR_MIN
    DEFAULT_YEAR_MAX = min(EMERGENCY_YEAR_MAX, TODAY_YEAR)

print(f"\nDerived dataset year range: {DEFAULT_YEAR_MIN}–{DEFAULT_YEAR_MAX}")
print(f"Publication year ceiling capped at today: {TODAY_YEAR}")


EFFECTIVE_YEAR_MIN = max(DEFAULT_YEAR_MIN, EVO_YEAR_MIN) if EVO_YEAR_MIN is not None else DEFAULT_YEAR_MIN
print(f"Effective publication year floor (evolution-aware): {EFFECTIVE_YEAR_MIN}")


def _nearest_evo_year(rid: str, year: int) -> int | None:
    years = evo_year_by_rid.get(rid)
    if not years:
        return None
    idx = bisect.bisect_left(years, year)
    if idx == 0:
        return years[0]
    if idx >= len(years):
        return years[-1]
    lo, hi = years[idx - 1], years[idx]
    return hi if (hi - year) <= (year - lo) else lo


def filter_authors_by_publication_year(team: list[str], year: int) -> list[str]:
    """
    Keep authors present in profile and active in the evolution file
    for the publication year (nearest available year used).
 
    """
    valid = []
    for rid in team:
        if profile_ids is not None and rid not in profile_ids:
            continue
        if dev is not None:
            nearest = _nearest_evo_year(rid, year)
            if nearest is None:
                continue   # researcher has no evolution data at all
        valid.append(rid)
    return valid


def get_valid_publication_year_window(row) -> tuple[int | None, int | None]:
    """
    Publication year must fall inside:
      project active period ∩ grant active period.

    
    """
    project_id = norm_str(row.get(COL_PROJECT_ID, ""))

    proj_y1 = parse_year(row.get(COL_SDATE, "")) or DEFAULT_YEAR_MIN
    proj_y2 = parse_year(row.get(COL_EDATE, "")) or DEFAULT_YEAR_MAX

    y_low, y_high = proj_y1, proj_y2

   
    if project_id in grant_year_lookup:
        grant_y1, grant_y2 = grant_year_lookup[project_id]
        if grant_y1 is not None:
            y_low  = max(y_low,  grant_y1)
        if grant_y2 is not None:
            y_high = min(y_high, grant_y2)

   
    y_low  = max(y_low,  EFFECTIVE_YEAR_MIN)
    y_high = min(y_high, DEFAULT_YEAR_MAX)

    return (None, None) if y_low > y_high else (y_low, y_high)

# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                            # 13) GENERATE PUBLICATIONS
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

print("\nGenerating publications ...")

pub_rows: list[dict] = []

skipped_no_team             = 0
skipped_no_year_overlap     = 0
skipped_no_valid_author_year = 0

for _, row in df.iterrows():
    project_id = norm_str(row[COL_PROJECT_ID])
    grant_id   = norm_str(row[COL_GRANT_ID])
    proj_title = norm_str(row[COL_TITLE])
    domains    = split_domains(row[COL_DOMAIN])

    pi    = row[COL_PI]
    copis = row[COL_COPI] if isinstance(row[COL_COPI], list) else []

   
    team = list(dict.fromkeys(
        [r for r in ([pi] + [c for c in copis if c != pi]) if r]
    ))
    if profile_ids is not None:
        team = [t for t in team if t in profile_ids]

    if not team:
        skipped_no_team += 1
        continue

    y_low, y_high = get_valid_publication_year_window(row)
    if y_low is None:
        skipped_no_year_overlap += 1
        continue

    kpub = random.randint(MIN_PUB_PER_PROJECT, MAX_PUB_PER_PROJECT)

    for j in range(kpub):
        year       = random.randint(y_low, y_high)
        valid_team = filter_authors_by_publication_year(team, year)

        if not valid_team:
            skipped_no_valid_author_year += 1
            continue

       
        title        = make_pub_title(proj_title, domains, j)
        pub_domain   = assign_pub_domain(domains, j)
        content      = get_content(pub_domain)
        journal      = random.choice(content["journals"])
        keywords     = random.choice(content["keywords"])
        volume       = make_volume(year, DEFAULT_YEAR_MAX)
        issue        = make_issue(pub_domain)
        impact_factor  = make_impact_factor(content)
        citation_count = make_citation_count(year, content, DEFAULT_YEAR_MAX)

        if len(valid_team) == 1:
            authors = valid_team[:]
        else:
            k_auth = random.randint(2, len(valid_team))
            if pi and pi in valid_team and random.random() < 0.85:
                rest    = [a for a in valid_team if a != pi]
                authors = [pi] + random.sample(rest, min(k_auth - 1, len(rest)))
            else:
                authors = random.sample(valid_team, k_auth)

        pub_rows.append({
            "publication_id": "",          
            "title":          title,
            "author_ids":     ";".join(authors),
            "keywords":       keywords,
            "journal":        journal,
            "issue":          issue,
            "volume":         volume,
            "year":           year,
            "doi":            "",          # filled after ID is known
            "impact_factor":  impact_factor,
            "citation_count": citation_count,
            "domain":         pub_domain,
            "project_id":     project_id,
            "grant_id":       grant_id,

        })

# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                      # 14) SORT CHRONOLOGICALLY, ASSIGN IDs, SAVE
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

pub_df = pd.DataFrame(pub_rows, columns=[
    "publication_id", "title", "author_ids", "journal", "volume", "issue", "year",
    "doi", "keywords", "citation_count", "impact_factor",
    "domain", "project_id", "grant_id",
])


pub_df = (
    pub_df
    .sort_values(["year", "project_id"], ascending=[True, True])
    .reset_index(drop=True)
)

pub_df["publication_id"] = [f"{PUB_PREFIX}{i+1:06d}" for i in range(len(pub_df))]
pub_df["doi"]            = pub_df["publication_id"].apply(make_doi)

pub_df.to_csv(OUT_PUB_CSV, index=False, quoting=csv.QUOTE_MINIMAL)

# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                # 15) REPORT
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

print(f"\nDone! {len(pub_df)} publications saved to '{OUT_PUB_CSV}'")
print("\nSkipped summary:")
print(f"  No valid team:                      {skipped_no_team}")
print(f"  Project-grant years don't overlap:  {skipped_no_year_overlap}")
print(f"  No author active in selected year:  {skipped_no_valid_author_year}")

if not pub_df.empty:
    per_proj = pub_df.groupby("project_id").size()
    print(f"\nPublications per project — min: {per_proj.min()}  "
          f"max: {per_proj.max()}  mean: {per_proj.mean():.2f}")

    print("\nDomain distribution:")
    print(pub_df["domain"].value_counts().head(15).to_string())

    print("\nYear distribution:")
    print(pub_df["year"].value_counts().sort_index().to_string())

    print("\nGrant-linked publications:")
    print(pub_df["grant_id"].ne("").sum())

    # Verify IDs are chronological
    print(f"\nPUB-000001 year: {pub_df.loc[0, 'year']}  (should be earliest)")
    print(f"PUB-{len(pub_df):06d} year: {pub_df.loc[len(pub_df)-1, 'year']}  (should be latest)")

    print("\nSample publications:")
    try:
        display(pub_df.head(15))
    except NameError:
        print(pub_df.head(15))
else:
    print("No publications generated. Check project teams, date ranges, "
          "grant overlap, and researcher evolution coverage.")
