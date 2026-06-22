# -*- coding: utf-8 -*-
import bisect
import csv
import random
import re
from collections import defaultdict

import numpy as np
import pandas as pd

# ========================================================================================================================
                                              # 1. INPUT / OUTPUT FILES
# ======================================================================================================================== 

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

RESEARCHERS_CSV  = "researchers_profile.csv"
EVOLUTION_CSV    = "researcher_domain_evolution.csv"
OUT_PROJECTS_CSV = "Projects.csv"

# ========================================================================================================================
                                                # 2. BASIC SETTINGS
# ========================================================================================================================

PROJECT_TYPES = ["Internal", "External", "Consultancy"]
TYPE_WEIGHTS  = [0.30, 0.55, 0.15]

PROJECT_RATIO               = 0.50
PROJECT_HOLDOUT_RATE        = 0.10
MIN_COPI                    = 2
MAX_COPI                    = 4
DEFAULT_ACTIVE_END_YEAR     = 2028
MIN_PROJECT_DURATION_YEARS  = 1
AUTO_LIMIT_PROJECT_COUNT    = True
CROSS_DOMAIN_PROB           = 0.20

# ========================================================================================================================
                                  # 3. ROLE-BASED PROJECT LOAD CONSTRAINTS
# ========================================================================================================================

ROLE_MIN_PROJECTS = {
    "Research Scholar":     1,
    "Assistant Professor":  3,
    "Associate Professor":  3,
    "Professor":            3,
}
ROLE_MAX_PROJECTS = {
    "Research Scholar":     2,
    "Assistant Professor":  4,
    "Associate Professor":  4,
    "Professor":            4,
}
ROLE_MAX_PI = {
    "Research Scholar":     1,
    "Assistant Professor":  1,
    "Associate Professor":  2,
    "Professor":            3,
}

_DEFAULT_MIN_PROJECTS = 1
_DEFAULT_MAX_PROJECTS = 2
_DEFAULT_MAX_PI       = 1

def role_min_projects(role): return ROLE_MIN_PROJECTS.get(role, _DEFAULT_MIN_PROJECTS)
def role_max_projects(role): return ROLE_MAX_PROJECTS.get(role, _DEFAULT_MAX_PROJECTS)
def role_max_pi(role):       return ROLE_MAX_PI.get(role, _DEFAULT_MAX_PI)

# ========================================================================================================================
                                            # 4. DOMAIN KEYWORDS
# ========================================================================================================================

DOMAIN_KEYWORDS = {
    "Computer Science": [
        "machine learning", "deep learning", "graph neural networks",
        "natural language processing", "cybersecurity", "cloud computing",
        "computer vision", "federated learning", "data mining",
        "knowledge graphs", "edge computing", "explainable AI",
    ],
    "Engineering": [
        "structural health monitoring", "smart grid", "renewable energy",
        "robotics", "additive manufacturing", "control systems",
        "power electronics", "sensor networks", "thermal systems",
        "predictive maintenance", "microfabrication",
    ],
    "Mathematics": [
        "stochastic modelling", "optimization", "graph theory",
        "numerical methods", "topological data analysis",
        "Bayesian inference", "dynamical systems",
        "game theory", "statistical modelling",
    ],
    "Physics": [
        "quantum materials", "condensed matter", "photonics",
        "plasma physics", "spectroscopy", "semiconductor physics",
        "astrophysics", "quantum optics", "statistical physics",
    ],
    "Chemistry": [
        "catalysis", "green chemistry", "nanomaterials",
        "electrochemistry", "drug delivery", "polymer science",
        "water purification", "surface chemistry",
        "organic synthesis", "materials chemistry",
    ],
    "Biology": [
        "genomics", "microbiome", "molecular biology",
        "bioinformatics", "systems biology", "plant genetics",
        "cell signalling", "protein folding", "ecology",
        "microbiology",
    ],
    "Medical Sciences": [
        "clinical diagnosis", "biomarkers", "medical imaging",
        "public health", "drug discovery", "epidemiology",
        "cancer therapy", "clinical decision support",
        "neuroimaging", "telemedicine",
    ],
    "Agriculture & Environment": [
        "precision farming", "soil health", "crop modelling",
        "climate resilience", "remote sensing", "irrigation",
        "biodiversity conservation", "sustainable agriculture",
        "watershed management", "pest management",
    ],
    "Arts & Humanities": [
        "digital humanities", "cultural heritage", "translation studies",
        "literary analysis", "historical archives", "linguistics",
        "visual culture", "folklore studies", "manuscript preservation",
    ],
    "Social Sciences": [
        "public policy", "social networks", "governance",
        "education policy", "development studies", "urban studies",
        "social mobility", "digital society", "behavioural economics",
        "community development",
    ],
}

ALL_DOMAINS     = list(DOMAIN_KEYWORDS.keys())
ALL_DOMAINS_SET = set(ALL_DOMAINS)

# ========================================================================================================================
                                            # 5. DOMAIN RULES
# ========================================================================================================================

REALISTIC_BROAD_DOMAIN_NEIGHBORS = {
    "Computer Science":          {"Engineering","Mathematics","Medical Sciences",
                                  "Biology","Agriculture & Environment","Social Sciences"},
    "Engineering":               {"Computer Science","Physics","Chemistry",
                                  "Agriculture & Environment","Medical Sciences","Mathematics"},
    "Mathematics":               {"Computer Science","Physics","Engineering",
                                  "Social Sciences","Biology"},
    "Physics":                   {"Mathematics","Engineering","Chemistry","Computer Science"},
    "Chemistry":                 {"Biology","Medical Sciences","Physics",
                                  "Engineering","Agriculture & Environment"},
    "Biology":                   {"Medical Sciences","Chemistry",
                                  "Agriculture & Environment","Computer Science"},
    "Medical Sciences":          {"Biology","Chemistry","Computer Science",
                                  "Social Sciences","Engineering"},
    "Agriculture & Environment": {"Biology","Chemistry","Engineering",
                                  "Computer Science","Social Sciences"},
    "Arts & Humanities":         {"Social Sciences","Computer Science"},
    "Social Sciences":           {"Arts & Humanities","Computer Science","Medical Sciences",
                                  "Agriculture & Environment","Mathematics"},
}

BLOCKED_BROAD_DOMAIN_PAIRS = {
    tuple(sorted(("Arts & Humanities","Physics"))),
    tuple(sorted(("Arts & Humanities","Chemistry"))),
    tuple(sorted(("Arts & Humanities","Engineering"))),
    tuple(sorted(("Arts & Humanities","Medical Sciences"))),
    tuple(sorted(("Physics","Social Sciences"))),
    tuple(sorted(("Chemistry","Social Sciences"))),
}

DRIFT_AWARE_STAGES = {"exploratory","transitioning","shifted","strong_drift"}


def is_realistic_pair(d1, d2):
    if tuple(sorted((d1, d2))) in BLOCKED_BROAD_DOMAIN_PAIRS: return False
    return (d2 in REALISTIC_BROAD_DOMAIN_NEIGHBORS.get(d1, set())
            or d1 in REALISTIC_BROAD_DOMAIN_NEIGHBORS.get(d2, set()))


def canonical_domain(domain):
    if isinstance(domain, str) and domain in ALL_DOMAINS_SET: return domain
    if pd.isna(domain): return "Unknown"
    parts = [p.strip() for p in str(domain).split("+") if p.strip()]
    parts = [p for p in parts if p in ALL_DOMAINS_SET]
    if len(parts) == 1: return parts[0]
    if len(parts) >= 2: return "+".join(sorted(parts[:2]))
    return "Unknown"

# ========================================================================================================================
                                            # 6. TITLE GENERATION
# ========================================================================================================================

METHOD_BANK    = ["A Study on","Development of","Design and Analysis of",
                  "Computational Modelling of","Framework for","Assessment of",
                  "Optimization of","Integrated Approach for",
                  "Data-Driven Analysis of","Sustainable Strategies for"]
CONNECTOR_BANK = ["for","in","using","towards","with applications in"]
CONTEXT_BANK   = ["Indian Academic Institutions","Higher Education Systems",
                  "Rural and Urban Communities","Smart Research Ecosystems",
                  "Interdisciplinary Research Settings","Policy and Innovation Environments",
                  "Sustainable Development Contexts","Technology-Enabled Academic Systems"]


def generate_single_domain_title(domain):
    kws = DOMAIN_KEYWORDS[domain]
    kw1, kw2 = random.choice(kws), random.choice(kws)
    return random.choice([
        f"{random.choice(METHOD_BANK)} {kw1.title()} {random.choice(CONNECTOR_BANK)} {random.choice(CONTEXT_BANK)}",
        f"{random.choice(METHOD_BANK)} {kw1.title()} and {kw2.title()} in {random.choice(CONTEXT_BANK)}",
        f"{kw1.title()} Based Framework for {kw2.title()}",
        f"Integrated {kw1.title()} Approach for {kw2.title()} in {random.choice(CONTEXT_BANK)}",
        f"Data-Driven {kw1.title()} for {kw2.title()} Applications",
    ])


def generate_cross_domain_title(domain_pair):
    d1, d2   = domain_pair.split("+")
    kw1, kw2 = random.choice(DOMAIN_KEYWORDS[d1]), random.choice(DOMAIN_KEYWORDS[d2])
    return random.choice([
        f"{random.choice(METHOD_BANK)} {kw1.title()} and {kw2.title()} in {random.choice(CONTEXT_BANK)}",
        f"Interdisciplinary Framework Combining {kw1.title()} and {kw2.title()}",
        f"{kw1.title()} Enabled {kw2.title()} for {random.choice(CONTEXT_BANK)}",
        f"Cross-Domain Analysis of {kw1.title()} and {kw2.title()}",
        f"Integrated {kw1.title()} and {kw2.title()} Approach for {random.choice(CONTEXT_BANK)}",
    ])


def generate_project_title(domain):
    domain = canonical_domain(domain)
    if "+" in domain:            return generate_cross_domain_title(domain)
    if domain in DOMAIN_KEYWORDS: return generate_single_domain_title(domain)
    return "Interdisciplinary Research Project in Academic Innovation"


_KW_DOMAIN_INDEX = {}
for _dom, _kws in DOMAIN_KEYWORDS.items():
    for _kw in _kws:
        _KW_DOMAIN_INDEX[_kw.lower()] = _dom


def detect_domains(title):
    tl, scores = title.lower(), {}
    for kw, dom in _KW_DOMAIN_INDEX.items():
        if kw in tl: scores[dom] = scores.get(dom, 0) + 1
    if not scores: return "Unknown"
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    if len(ranked) == 1: return ranked[0][0]
    d1, s1 = ranked[0]; d2, s2 = ranked[1]
    if s2 >= 1 and s2 >= 0.40 * s1 and is_realistic_pair(d1, d2):
        return "+".join(sorted([d1, d2]))
    return d1

# ========================================================================================================================
                                                # 7. HELPERS
# ========================================================================================================================

def find_col(df, candidates, required=True):
    lm = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower().strip() in lm: return lm[cand.lower().strip()]
    if required: raise KeyError(f"None of these columns found: {candidates}")
    return None

def safe_str(x): return "" if pd.isna(x) else str(x).strip()

def normalize_rid(x):
    s = safe_str(x).upper()
    if s in {"","NAN","NONE","NULL"}: return ""
    m = re.match(r"^R(\d+)$", s)
    return "R" + m.group(1).zfill(4) if m else s

def parse_date_value(x, mode="start"):
    if pd.isna(x): return pd.NaT
    s = str(x).strip()
    if not s or s.lower() in {"nan","none","nat"}: return pd.NaT
    if re.fullmatch(r"\d{4}", s):
        return pd.Timestamp(f"{s}-01-01" if mode == "start" else f"{s}-12-31")
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def format_date(dt): return pd.Timestamp(dt).strftime("%d/%m/%Y")

def random_date_between(start_dt, end_dt):
    s, e = pd.Timestamp(start_dt), pd.Timestamp(end_dt)
    if pd.isna(s) or pd.isna(e) or e < s: return None
    return s + pd.Timedelta(days=random.randint(0, max((e - s).days, 0)))

def make_status(start_dt, end_dt):
    ey = pd.Timestamp(end_dt).year
    if ey < min(DEFAULT_ACTIVE_END_YEAR, evo_max_year):
        return random.choice(["Completed","Completed","Completed","Terminated"])
    if pd.Timestamp(start_dt).year > DEFAULT_ACTIVE_END_YEAR: return "Yet to Start"
    return random.choice(["Ongoing","Ongoing","Completed"])

# ========================================================================================================================
                                                    # 8. READ FILES
# ========================================================================================================================

print("Reading researcher profile...")
df_res = pd.read_csv(RESEARCHERS_CSV)
df_res.columns = df_res.columns.str.strip()

print("Reading researcher domain evolution...")
df_evo = pd.read_csv(EVOLUTION_CSV)
df_evo.columns = df_evo.columns.str.strip()

required_evo_cols = {"year","r_id","current_dominant_domain",
                     "target_drift_domain","drift_stage","jsd_drift_score"}
missing = required_evo_cols - set(df_evo.columns)
if missing: raise ValueError(f"Missing required columns in evolution file: {missing}")

R_ID_COL     = find_col(df_res, ["r_id","researcher_id"])
R_START_COL  = find_col(df_res, ["career_start_year","career_start_date"])
R_END_COL    = find_col(df_res, ["career_end_date","career_end_year"], required=False)
R_ACTIVE_COL = find_col(df_res, ["is_active","active"], required=False)
R_DOMAIN_COL = find_col(df_res, ["primary_domain","domain"], required=False)
R_ROLE_COL   = find_col(df_res, ["role","designation"], required=False)

df_res[R_ID_COL] = df_res[R_ID_COL].apply(normalize_rid)
df_evo["r_id"]   = df_evo["r_id"].apply(normalize_rid)
df_evo["year"]   = pd.to_numeric(df_evo["year"], errors="coerce")
df_evo           = df_evo.dropna(subset=["r_id","year"]).copy()
df_evo["year"]   = df_evo["year"].astype(int)
df_evo["current_dominant_domain"] = df_evo["current_dominant_domain"].astype(str).str.strip()
df_evo["target_drift_domain"]     = df_evo["target_drift_domain"].astype(str).str.strip()
df_evo["drift_stage"]             = df_evo["drift_stage"].astype(str).str.strip()

print(f"Researchers loaded:    {len(df_res)}")
print(f"Evolution rows loaded: {len(df_evo)}")
print(f"Evolution years:       {df_evo['year'].min()} to {df_evo['year'].max()}")

# ========================================================================================================================
                                        # 9. BUILD RESEARCHER LOOKUPS
# ========================================================================================================================

def parse_career_end(row):
    if R_ACTIVE_COL:
        is_active = safe_str(row[R_ACTIVE_COL]).lower() in {"1","true","yes","y"}
    if R_END_COL and pd.notna(row[R_END_COL]) and safe_str(row[R_END_COL]):
        end_dt = parse_date_value(row[R_END_COL], mode="end")
        if pd.notna(end_dt): return end_dt
    return pd.Timestamp(f"{DEFAULT_ACTIVE_END_YEAR}-12-31")


researcher_meta = {}
for _, row in df_res.iterrows():
    rid = normalize_rid(row[R_ID_COL])
    if not rid: continue
    start_dt = parse_date_value(row[R_START_COL], mode="start")
    if pd.isna(start_dt): continue
    end_dt = parse_career_end(row)
    if pd.isna(end_dt) or end_dt < start_dt: end_dt = start_dt
    role = safe_str(row[R_ROLE_COL]) if R_ROLE_COL else ""
    researcher_meta[rid] = {
        "start":        start_dt,
        "end":          end_dt,
        "primary_domain": safe_str(row[R_DOMAIN_COL]) if R_DOMAIN_COL else "",
        "role":         role,
        "max_projects": role_max_projects(role),
        "min_projects": role_min_projects(role),
        "max_pi":       role_max_pi(role),
    }

_career_span = {rid: (m["start"], m["end"]) for rid, m in researcher_meta.items()}

valid_researchers = list(researcher_meta.keys())
if not valid_researchers: raise ValueError("No valid researchers found.")

holdout_count = int(round(len(valid_researchers) * PROJECT_HOLDOUT_RATE))
held_out      = set(random.sample(valid_researchers, holdout_count)) if holdout_count else set()
project_eligible_researchers = [r for r in valid_researchers if r not in held_out]

print(f"Valid researchers:            {len(valid_researchers)}")
print(f"Held out from projects:       {len(held_out)}")
print(f"Project-eligible researchers: {len(project_eligible_researchers)}")

requested_count = int(round(len(df_res) * PROJECT_RATIO))
if AUTO_LIMIT_PROJECT_COUNT:
    eligible_slots = sum(researcher_meta[r]["max_projects"] for r in project_eligible_researchers)
    avg_team_size  = 1 + (MIN_COPI + MAX_COPI) / 2
    PROJECT_COUNT  = min(requested_count, int(eligible_slots // avg_team_size))
else:
    PROJECT_COUNT = requested_count

print(f"Requested projects:           {requested_count}")
print(f"Final projects to generate:   {PROJECT_COUNT}")

# ========================================================================================================================
                                            # 10. EVOLUTION LOOKUPS
# ========================================================================================================================

_evo_index = {}
for _, row in df_evo.iterrows():
    _evo_index[(row["r_id"], int(row["year"]))] = {
        "current":     canonical_domain(row["current_dominant_domain"]),
        "target":      canonical_domain(row.get("target_drift_domain", "")),
        "drift_stage": str(row.get("drift_stage","stable")).strip(),
    }

available_evo_years = sorted(df_evo["year"].unique().tolist())
evo_min_year        = int(df_evo["year"].min())
evo_max_year        = int(df_evo["year"].max())
evo_max_dt          = pd.Timestamp(f"{evo_max_year}-12-31")


def nearest_evolution_year(year):
    if not available_evo_years: return year
    idx = bisect.bisect_left(available_evo_years, year)
    if idx == 0: return available_evo_years[0]
    if idx >= len(available_evo_years): return available_evo_years[-1]
    lo, hi = available_evo_years[idx-1], available_evo_years[idx]
    return hi if (hi - year) < (year - lo) else lo


def get_researcher_evo_at_year(rid, year):
    """Returns evolution state {current, target, drift_stage} at nearest year."""
    entry = _evo_index.get((rid, nearest_evolution_year(year)))
    if entry: return entry
    pdom = canonical_domain(researcher_meta.get(rid,{}).get("primary_domain",""))
    return {"current": pdom, "target": "Unknown", "drift_stage": "stable"}


def acceptable_domains(rid, year):
    """
    Drift-aware domain set for a researcher at a given year.
    Always includes current domain.
    Also includes target domain if drift stage is
    exploratory / transitioning / shifted / strong_drift.
    """
    evo   = get_researcher_evo_at_year(rid, year)
    valid = set()
    if evo["current"] not in ("Unknown",""): valid.add(evo["current"])
    if evo["drift_stage"] in DRIFT_AWARE_STAGES and evo["target"] not in ("Unknown",""):
        valid.add(evo["target"])
    if not valid:
        fb = canonical_domain(researcher_meta.get(rid,{}).get("primary_domain",""))
        if fb not in ("Unknown",""): valid.add(fb)
    return valid


def get_year_domain_distribution(year):
    """Population-level domain distribution at a given year."""
    use_year = nearest_evolution_year(year)
    temp     = df_evo[df_evo["year"] == use_year]
    return {
        canonical_domain(d): int(c)
        for d, c in temp["current_dominant_domain"].value_counts().items()
        if canonical_domain(d) in ALL_DOMAINS_SET
    }


def choose_domain_for_year(year, force_cross=False):
    """
    Pick a project domain weighted by researcher distribution at `year`.
    force_cross=True → produce a cross-domain pair.
    """
    counts = get_year_domain_distribution(year)
    if not counts: counts = {d: 1 for d in ALL_DOMAINS}
    d1 = random.choices(list(counts.keys()), weights=list(counts.values()), k=1)[0]
    if not force_cross:
        return d1
    neighbors = [d for d in REALISTIC_BROAD_DOMAIN_NEIGHBORS.get(d1, set())
                 if d in ALL_DOMAINS_SET and is_realistic_pair(d1, d)]
    if not neighbors:
        neighbors = [d for d in ALL_DOMAINS if d != d1 and is_realistic_pair(d1, d)]
    if not neighbors: return d1
    return "+".join(sorted([d1, random.choice(neighbors)]))

# ========================================================================================================================
                                    # 11. TEAM SELECTION & DATE HELPERS
# ========================================================================================================================

project_count_by_rid = defaultdict(int)
pi_count_by_rid      = defaultdict(int)
_eligible_set        = set(project_eligible_researchers)


def career_overlap(team_ids):
    spans = [_career_span[rid] for rid in team_ids if rid in _career_span]
    if len(spans) != len(team_ids): return None, None
    vs = max(s[0] for s in spans)
    ve = min(s[1] for s in spans)
    return (vs, ve) if ve >= vs else (None, None)


def feasible_for_project(rid):
    if rid not in _eligible_set: return False
    return project_count_by_rid[rid] < researcher_meta.get(rid,{}).get("max_projects",_DEFAULT_MAX_PROJECTS)


def feasible_for_pi(rid):
    if not feasible_for_project(rid): return False
    return pi_count_by_rid[rid] < researcher_meta.get(rid,{}).get("max_pi",_DEFAULT_MAX_PI)


def domain_matches(researcher_domain, project_domain):
    if researcher_domain == "Unknown" or project_domain == "Unknown": return False
    return bool(set(project_domain.split("+")) & set(researcher_domain.split("+")))

# ========================================================================================================================
                      # 12. CORE GENERATION FUNCTIONS  (date → domain → team order)
# ========================================================================================================================

def generate_start_date(project_type):
    """
    STEP 1 — Pick a start date independently of domain/team.
    Sample two random eligible researchers, find their overlap,
    pick a start date within that window, then return the year.
    This anchors the project year before anything else is decided.
    """
    if project_type == "Internal":   dur = random.randint(1, 2)
    elif project_type == "External": dur = random.randint(2, 4)
    else:                            dur = random.randint(1, 3)

    # Try up to 20 random pairs to find a usable overlap window
    for _ in range(20):
        sample = random.sample(project_eligible_researchers,
                               min(3, len(project_eligible_researchers)))
        spans  = [_career_span[r] for r in sample if r in _career_span]
        if not spans: continue
        vs = max(s[0] for s in spans)
        ve = min(s[1] for s in spans)
        if ve < vs: continue

        latest_start = ve - pd.DateOffset(years=dur)
        if latest_start < vs: continue

        # Clamp to evo range
        clamped_start = max(vs, pd.Timestamp(f"{evo_min_year}-01-01"))
        clamped_end   = min(latest_start, evo_max_dt - pd.DateOffset(years=dur))
        if clamped_end < clamped_start: continue

        start_dt = random_date_between(clamped_start, clamped_end)
        if start_dt is None: continue

        end_dt = pd.Timestamp(start_dt) + pd.DateOffset(years=dur)
        if end_dt > evo_max_dt: end_dt = evo_max_dt
        if (end_dt - pd.Timestamp(start_dt)).days < 365 * MIN_PROJECT_DURATION_YEARS:
            continue

        return pd.Timestamp(start_dt), pd.Timestamp(end_dt), dur

    return None, None, None


def pick_team_at_year(project_domain, year, project_type, dur):
    """
    STEP 3 — Select PI and Co-Is whose domain state AT `year` fits
    the project domain. Career overlap must contain the project dates.
    All domain checks use the evolution state at `year`.
    """
    project_parts = set(project_domain.split("+"))

    # Cache acceptable domains at this specific year for all researchers
    dom_cache = {}
    def get_acceptable(rid):
        if rid not in dom_cache:
            dom_cache[rid] = acceptable_domains(rid, year)
        return dom_cache[rid]

    # PI candidates: drift-aware domain match at `year`
    pi_candidates = [
        r for r in project_eligible_researchers
        if feasible_for_pi(r) and bool(get_acceptable(r) & project_parts)
    ]
    if not pi_candidates: return None, None, None

    min_load      = min(project_count_by_rid[r] for r in pi_candidates)
    pi_candidates = [r for r in pi_candidates if project_count_by_rid[r] == min_load]
    pi            = random.choice(pi_candidates)

    # Co-I candidates: domain match or realistic neighbour at `year`
    candidates = []
    for rid in project_eligible_researchers:
        if rid == pi or not feasible_for_project(rid): continue
        r_acc = get_acceptable(rid)
        # primary match: any acceptable domain overlaps project domain
        if r_acc & project_parts:
            candidates.append((rid, 2))
        # secondary match: acceptable domain is a realistic neighbour
        elif any(
            d in ALL_DOMAINS_SET and any(is_realistic_pair(d, p) for p in project_parts)
            for d in r_acc
        ):
            candidates.append((rid, 1))

    if len(candidates) < MIN_COPI: return None, None, None

    # Ensure every domain part of a cross-domain project is covered
    selected, covered = [], set()
    covered |= (get_acceptable(pi) & project_parts)

    for pdom in project_parts:
        if pdom in covered: continue
        pdom_cands = [r for r, _ in candidates if pdom in get_acceptable(r)]
        if pdom_cands:
            ml     = min(project_count_by_rid[r] for r in pdom_cands)
            chosen = random.choice([r for r in pdom_cands if project_count_by_rid[r] == ml])
            if chosen not in selected:
                selected.append(chosen)
                covered.add(pdom)

    rem_min = max(0, MIN_COPI - len(selected))
    rem_max = MAX_COPI - len(selected)
    remaining = [r for r, _ in candidates if r not in selected]
    if len(remaining) < rem_min: return None, None, None

    remaining.sort(key=lambda r: (
        project_count_by_rid[r],
        -(2 if bool(get_acceptable(r) & project_parts) else 1)
    ))
    extra    = random.randint(rem_min, max(rem_min, min(rem_max, len(remaining))))
    selected += random.sample(remaining[:min(len(remaining), 30)], extra)
    selected  = list(dict.fromkeys(selected))[:MAX_COPI]
    if len(selected) < MIN_COPI: return None, None, None

    # Verify the full team's career overlap contains the project
    team        = [pi] + selected
    ov_s, ov_e  = career_overlap(team)
    if ov_s is None: return None, None, None

    # Generate final project dates within team's career overlap
    latest_start = ov_e - pd.DateOffset(years=dur)
    # Clamp to evo range
    earliest_start = max(ov_s, pd.Timestamp(f"{evo_min_year}-01-01"))
    if latest_start < earliest_start: return None, None, None

    start_dt = random_date_between(earliest_start, latest_start)
    if start_dt is None: return None, None, None

    # Re-check the start year matches what we expect
    if start_dt.year != year:
        # Allow ±1 year tolerance; otherwise reject
        if abs(start_dt.year - year) > 1:
            return None, None, None

    end_dt = pd.Timestamp(start_dt) + pd.DateOffset(years=dur)
    if end_dt > ov_e:      end_dt = ov_e
    if end_dt > evo_max_dt: end_dt = evo_max_dt
    if (end_dt - pd.Timestamp(start_dt)).days < 365 * MIN_PROJECT_DURATION_YEARS:
        return None, None, None

    return (pi, selected), (pd.Timestamp(start_dt), pd.Timestamp(end_dt))


# ========================================================================================================================
                                        # 13. GENERATE PROJECTS
# ========================================================================================================================

cross_domain_target  = int(round(PROJECT_COUNT * CROSS_DOMAIN_PROB))
single_domain_target = PROJECT_COUNT - cross_domain_target

rows, seen_titles    = [], set()
attempts             = 0
max_attempts         = PROJECT_COUNT * 10
cross_generated      = 0
single_generated     = 0

print(f"\nGenerating {PROJECT_COUNT} projects "
      f"({cross_domain_target} cross-domain, {single_domain_target} single-domain)...")
print("Flow: start_date → actual_year → domain → team (all at actual_year)\n")

while len(rows) < PROJECT_COUNT and attempts < max_attempts:
    attempts += 1

    remaining_cross  = cross_domain_target  - cross_generated
    remaining_single = single_domain_target - single_generated
    remaining_total  = PROJECT_COUNT - len(rows)

    if remaining_cross <= 0:    force_cross = False
    elif remaining_single <= 0: force_cross = True
    else:
        force_cross = random.random() < (remaining_cross / remaining_total)

    project_type = random.choices(PROJECT_TYPES, weights=TYPE_WEIGHTS, k=1)[0]

    # ── STEP 1: Generate start date to anchor the year ──────────────────
    start_dt, end_dt, dur = generate_start_date(project_type)
    if start_dt is None:
        continue

    actual_year = int(start_dt.year)   # ← THIS is the year used for everything below

    # ── STEP 2: Choose domain based on actual_year ───────────────────────
    domain = canonical_domain(choose_domain_for_year(actual_year, force_cross=force_cross))
    if domain == "Unknown":
        continue

    if "+" in domain:
        parts = domain.split("+")
        if len(parts) != 2 or not is_realistic_pair(parts[0], parts[1]):
            continue
    elif force_cross:
        continue

    # ── STEP 3 & 4: Pick team and validate — all at actual_year ─────────
    result = pick_team_at_year(domain, actual_year, project_type, dur)
    if result[0] is None:
        continue

    (pi, co_investigators), (final_start, final_end) = result

    # Title generation and domain detection
    title = generate_project_title(domain)
    if title in seen_titles:
        continue

    detected = canonical_domain(detect_domains(title))
    if detected == "Unknown":
        continue
    if "+" in detected:
        dp = detected.split("+")
        if len(dp) != 2 or not is_realistic_pair(dp[0], dp[1]):
            continue
    if not set(domain.split("+")) & set(detected.split("+")):
        continue

    rows.append({
        "project_id":             "",
        "grant_ID":               "",
        "project_title":          title,
        "domain":                 domain,
        "project_type":           project_type,
        "start_date":             format_date(final_start),
        "end_date":               format_date(final_end),
        "status":                 make_status(final_start, final_end),
        "principal_investigator": pi,
        "co_investigators":       ";".join(co_investigators),
    })
    seen_titles.add(title)
    project_count_by_rid[pi] += 1
    pi_count_by_rid[pi]      += 1
    for co in co_investigators: project_count_by_rid[co] += 1
    if "+" in domain: cross_generated  += 1
    else:             single_generated += 1

# ========================================================================================================================
                                          # 14. SAVE & VALIDATE
# ========================================================================================================================

df_projects = pd.DataFrame(rows)
df_projects["_sort_start"] = pd.to_datetime(df_projects["start_date"], dayfirst=True)
df_projects["_sort_end"]   = pd.to_datetime(df_projects["end_date"],   dayfirst=True)
df_projects = (df_projects
               .sort_values(["_sort_start","_sort_end"], ascending=[True,True])
               .reset_index(drop=True))
df_projects["project_id"] = [f"PROJ-{i+1:05d}" for i in range(len(df_projects))]
df_projects = df_projects.drop(columns=["_sort_start","_sort_end"])
df_projects.to_csv(OUT_PROJECTS_CSV, index=False, quoting=csv.QUOTE_MINIMAL)

print(f"Saved:                   {OUT_PROJECTS_CSV}")
print(f"Projects generated:      {len(df_projects)}")
print(f"Attempts made:           {attempts}")
print(f"Cross-domain generated:  {cross_generated}  (target {cross_domain_target})")
print(f"Single-domain generated: {single_generated}  (target {single_domain_target})")

if df_projects.empty: raise ValueError("No projects generated.")

start_yrs = pd.to_datetime(df_projects["start_date"], dayfirst=True).dt.year
end_yrs   = pd.to_datetime(df_projects["end_date"],   dayfirst=True).dt.year
print(f"\nProject date range: {start_yrs.min()} to {end_yrs.max()}")
print("\nProject type distribution:")
print(df_projects["project_type"].value_counts())
print("\nStatus distribution:")
print(df_projects["status"].value_counts())
print("\nDomain distribution (top 20):")
print(df_projects["domain"].value_counts().head(20))
print(f"\nCross-domain:  {df_projects['domain'].str.contains(r'+',regex=False).sum()}")
print(f"Single-domain: {(~df_projects['domain'].str.contains(r'+',regex=False)).sum()}")

# Role-based load validation
print("\nRole-based load validation:")
over_total, over_pi_list = [], []
for rid, count in project_count_by_rid.items():
    meta = researcher_meta.get(rid,{})
    if count > meta.get("max_projects",_DEFAULT_MAX_PROJECTS):
        over_total.append((rid, meta.get("role","?"), count))
for rid, count in pi_count_by_rid.items():
    meta = researcher_meta.get(rid,{})
    if count > meta.get("max_pi",_DEFAULT_MAX_PI):
        over_pi_list.append((rid, meta.get("role","?"), count))
if over_total:   print(f"  OVER-LIMIT (total): {over_total[:10]}")
else:            print("  All total project counts within role limits.")
if over_pi_list: print(f"  OVER-LIMIT (PI): {over_pi_list[:10]}")
else:            print("  All PI counts within role limits.")

# Temporal validation
_sv = pd.to_datetime(df_projects["start_date"], dayfirst=True)
_ev = pd.to_datetime(df_projects["end_date"],   dayfirst=True)
_pv = df_projects["principal_investigator"].apply(normalize_rid)
_cv = df_projects["co_investigators"]
invalid_ids = []
for i, proj_id in enumerate(df_projects["project_id"]):
    team = ([_pv.iat[i]]
            + [normalize_rid(x) for x in str(_cv.iat[i]).split(";") if normalize_rid(x)])
    ov_s, ov_e = career_overlap(team)
    if (ov_s is None
            or _sv.iat[i] < ov_s
            or _ev.iat[i] > ov_e
            or _ev.iat[i] > evo_max_dt):
        invalid_ids.append(proj_id)

print("\nTemporal validation:")
print(f"Invalid project date rows: {len(invalid_ids)}")
if invalid_ids: print("Invalid Project IDs sample:", invalid_ids[:20])
else:           print("All project dates within career overlap and evo year range.")

# Evolution consistency check — sample 20 projects
print("\nEvolution consistency check (sample 20):")
sample_df = df_projects.sample(min(20, len(df_projects)), random_state=42)
inconsistent = []
for _, row in sample_df.iterrows():
    yr    = pd.to_datetime(row["start_date"], dayfirst=True).year
    pi    = normalize_rid(row["principal_investigator"])
    pdom  = row["domain"]
    pi_ok = bool(acceptable_domains(pi, yr) & set(pdom.split("+")))
    if not pi_ok:
        inconsistent.append((row["project_id"], pi, yr, pdom,
                              acceptable_domains(pi, yr)))
if inconsistent:
    print(f"  Inconsistent PI-domain assignments: {len(inconsistent)}")
    for item in inconsistent[:5]: print(f"    {item}")
else:
    print("  All sampled PI domains consistent with evolution at project start year.")

print("\nSample projects:")
try:    display(df_projects.head(15))
except: print(df_projects.head(15))
