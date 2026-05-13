"""
grants.py
──────────────────
Reads:
  • generating_projects.csv - Reads the output of generate_projects.py. At this stage, grants are not yet mapped to their corresponding project IDs. The grant IDs are assigned to the respective projects in Grants.py. 
  • Researchers_Profile.csv

All original constraints preserved exactly:
  - PI must belong to at least one of the project's domains
  - Cross-domain projects must have at least one CoPI per uncovered domain
  - Max 2 total projects per researcher
  - Max 1 project as PI per researcher
  - Each project has 2 or 3 Co-Investigators
  - 20% of researchers are forced unused (never assigned)
Run:
  python grants.py
"""

import pandas as pd
import numpy as np
import random
import re
import itertools
from collections import Counter, defaultdict

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                              # 1) PATHS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
INPUT_PROJECTS_CSV = "generating_projects.csv"   # from generate_projects.py
RESEARCHERS_CSV    = "Researchers_Profile.csv"

OUT_PROJECTS_FINAL = "Projects.csv" # This file is used to generate complete project file 
OUT_GRANTS_FINAL   = "Grants.csv"

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                            # 2) COLUMN NAMES
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
COL_PROJECT_ID = "Project_ID"
COL_GRANT_ID   = "Grant_ID"
COL_TITLE      = "Project_Title"
COL_DOMAIN     = "Domain"
COL_TYPE       = "Project_Type"
COL_SDATE      = "Start_Date"
COL_EDATE      = "End_Date"
COL_STATUS     = "Status"

COL_AGENCY = "Agency"               # used internally; written to Grants_final only
COL_PI     = "Principal_Investigator"
COL_COPI   = "Co_Investigators"
COL_ALL    = "All_Researchers_In_Project"

RES_ID_COL     = "r_id"
RES_DOMAIN_COL = "primary_domain"

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                                  # 3) CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

FORCE_UNUSED_FRAC                 = 0.20   # 20% of researchers never assigned
MAX_TOTAL_PROJECTS_PER_RESEARCHER = 2
MAX_PI_PROJECTS_PER_RESEARCHER    = 1
COPI_CHOICES                      = [2, 3] # each project gets 2 or 3 CoPIs

GRANTS_COUNT     = 350                     # number of projects to fund
AMT_MIN, AMT_MAX = 5, 250                 # grant amount range (lakhs)
GRANT_PREFIX     = "GRANT-"

#═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                          # 4) STRING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
def norm_str(x) -> str:
    return "" if pd.isna(x) else str(x).strip()

def clean_id(x) -> str:
    x = norm_str(x)
    x = x.replace("\u00A0", " ").replace("\u200B", "")
    return re.sub(r"\s+", " ", x).strip()

def normalize_domain_name(d: str) -> str:
    return re.sub(r"\s+", " ", norm_str(d)).strip()

def split_project_domains(domain_cell: str) -> list:
    """Split domain cell on +, comma, semicolon, pipe. Deduplicate."""
    s = norm_str(domain_cell)
    if not s:
        return []
    parts = re.split(r"\+|,|;|\|", s)
    parts = [normalize_domain_name(p) for p in parts if normalize_domain_name(p)]
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def normalize_rid(x) -> str | None:
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
            seen.add(rid)
            out.append(rid)
    return out

def normalize_type(x) -> str:
    s = norm_str(x).lower()
    if "internal"  in s: return "internal"
    if "consult"   in s: return "consultancy"
    if "external"  in s: return "external"
    return s

def is_cross_domain(domain_str) -> int:
    return int(bool(re.search(r"[+,;|]", norm_str(domain_str))))

def domains_involved(domain_str) -> str:
    return ", ".join(split_project_domains(domain_str))

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                        # 5) AGENCY LOGIC
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
CANON = {
    "CS": "Computer Science",               "Computer Science": "Computer Science",
    "Engg": "Engineering",                  "Engineering": "Engineering",
    "Math": "Mathematics",                  "Mathematics": "Mathematics",
    "Physics": "Physics",                   "Chemistry": "Chemistry",
    "Biology": "Biology",
    "Medical": "Medical Sciences",          "Medical Sciences": "Medical Sciences",
    "Agri": "Agriculture & Environment",    "Agriculture": "Agriculture & Environment",
    "Agriculture & Environment": "Agriculture & Environment",
    "Arts": "Arts & Humanities",            "Arts & Humanities": "Arts & Humanities",
    "Social": "Social Sciences",            "Social Sciences": "Social Sciences",
}

AGENCY_TO_DOMAINS = {
    "DST-SERB":         {"Computer Science", "Engineering", "Mathematics", "Physics", "Chemistry"},
    "DST-SERB Synergy": {"Interdisciplinary"},
    "Industry":         {"Computer Science", "Engineering", "Medical Sciences", "Agriculture & Environment"},
    "Wellcome Trust":   {"Biology", "Medical Sciences"},
    "NSF Convergence":  {"Computer Science", "Engineering", "Mathematics", "Interdisciplinary"},
    "Gates Foundation": {"Medical Sciences", "Biology", "Agriculture & Environment"},
    "UKRI":             {"Computer Science", "Engineering", "Physics", "Social Sciences"},
    "ICMR":             {"Medical Sciences"},
    "DBT":              {"Biology", "Medical Sciences"},
    "ICAR":             {"Agriculture & Environment"},
    "ISRO":             {"Physics", "Engineering", "Computer Science"},
    "UGC":              {"Arts & Humanities", "Social Sciences"},
    "Horizon Europe":   {"Computer Science", "Engineering", "Physics", "Interdisciplinary"},
    "HHMI":             {"Biology", "Medical Sciences"},
    "CSIR":             {"Chemistry"},
}

AGENCY_PRIORITY = [
    "ICMR", "DBT", "ICAR", "ISRO", "CSIR",
    "UGC",
    "Wellcome Trust", "HHMI", "Gates Foundation",
    "NSF Convergence", "UKRI", "Horizon Europe",
    "DST-SERB", "DST-SERB Synergy", "Industry",
]

def allowed_agencies(domain_str: str) -> list:
    parts = split_project_domains(domain_str)
    parts = list(dict.fromkeys([CANON.get(p, p) for p in parts]))
    if not parts:
        return []
    is_inter = len(parts) >= 2
    labels   = set(parts)
    if is_inter:
        labels.add("Interdisciplinary")
    allowed = []
    for agency, ok in AGENCY_TO_DOMAINS.items():
        if agency == "DST-SERB Synergy":
            if is_inter:
                allowed.append(agency)
            continue
        if ok.intersection(labels):
            allowed.append(agency)
    return [a for a in AGENCY_PRIORITY if a in allowed]

def choose_agency(domain_str: str, proj_type: str) -> str:
    t = normalize_type(proj_type)
    if t == "internal":    return "Institutional Funding"
    if t == "consultancy": return "Industry"
    if t == "external":
        ag = allowed_agencies(domain_str)
        return ag[0] if ag else ""
    return ""

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                            # 6) LOAD RESEARCHERS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
print("Loading researcher profiles ...")
df_res = pd.read_csv(RESEARCHERS_CSV)

# Auto-detect column names flexibly
res_id_col = next(
    (c for c in df_res.columns if c.lower() == "r_id" or "r_id" in c.lower()),
    df_res.columns[0]
)
res_domain_col = next(
    (c for c in df_res.columns if "domain" in c.lower()),
    df_res.columns[1]
)

df_res[res_id_col]     = df_res[res_id_col].astype(str).apply(normalize_rid)
df_res[res_domain_col] = df_res[res_domain_col].astype(str).apply(normalize_domain_name)
df_res = df_res[df_res[res_id_col].notna() & (df_res[res_id_col] != "")]

all_researchers = sorted(set(df_res[res_id_col]))
domain_to_researchers: dict[str, list] = (
    df_res.groupby(res_domain_col)[res_id_col]
    .apply(lambda s: sorted(set(s)))
    .to_dict()
)
print(f"  {len(all_researchers)} researchers across {len(domain_to_researchers)} domains")
print(f"  Domains: {sorted(domain_to_researchers.keys())}")

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                        # 7) LOAD PROJECTS + DEDUP
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
print("\nLoading projects ...")
df = pd.read_csv(INPUT_PROJECTS_CSV)

# Ensure all required columns exist
for c in [COL_PROJECT_ID, COL_GRANT_ID, COL_TITLE, COL_DOMAIN,
          COL_TYPE, COL_SDATE, COL_EDATE, COL_STATUS]:
    if c not in df.columns:
        df[c] = ""
for c in [COL_PI, COL_COPI, COL_ALL, COL_AGENCY]:
    if c not in df.columns:
        df[c] = ""

df[COL_PROJECT_ID] = df[COL_PROJECT_ID].apply(clean_id)
df[COL_TITLE]      = df[COL_TITLE].apply(clean_id)

# Deduplicate on content
DEDUP_COLS = [COL_TITLE, COL_DOMAIN, COL_TYPE, COL_SDATE, COL_EDATE]
before = len(df)
df = df.drop_duplicates(subset=DEDUP_COLS, keep="first").reset_index(drop=True)
print(f"  {len(df)} projects after dedup (dropped {before - len(df)} duplicates)")

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                  # 8) FORCE 20% UNUSED RESEARCHERS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
num_unused         = int(FORCE_UNUSED_FRAC * len(all_researchers))
unused_researchers = (
    set(random.sample(all_researchers, num_unused)) if num_unused > 0 else set()
)
print(f"\n{len(unused_researchers)} researchers forced unused "
      f"({FORCE_UNUSED_FRAC*100:.0f}% of {len(all_researchers)})")

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                # 9) ASSIGN PI + CO-INVESTIGATORS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
print("\nAssigning PI and Co-Investigators ...")
total_count: dict[str, int] = defaultdict(int)
pi_count:    dict[str, int] = defaultdict(int)

def eligible(pool: list, exclude: set, for_pi: bool = False) -> list:
    """
    Filter pool to researchers who:
      - are not excluded
      - are not in the unused set
      - have not exceeded max total projects
      - (if for_pi) have not exceeded max PI projects
    Sorted by load (ascending) then ID for determinism.
    """
    out = []
    for r in pool:
        if r in exclude:                                               continue
        if r in unused_researchers:                                    continue
        if total_count[r] >= MAX_TOTAL_PROJECTS_PER_RESEARCHER:       continue
        if for_pi and pi_count[r] >= MAX_PI_PROJECTS_PER_RESEARCHER:  continue
        out.append(r)
    out.sort(key=lambda r: (total_count[r], r))
    return out

def choose_pi(domains: list) -> str | None:
    """Pick the lowest-load eligible PI from the union of domain pools."""
    pool  = sorted({r for d in domains for r in domain_to_researchers.get(d, [])})
    cands = eligible(pool, set(), for_pi=True)
    return cands[0] if cands else None

def pick_from_domain(domain: str, exclude: set) -> str | None:
    """Pick one eligible researcher from a specific domain pool."""
    cands = eligible(domain_to_researchers.get(domain, []), exclude)
    return cands[0] if cands else None

def pick_extra(domains: list, exclude: set, need_k: int) -> list:
    """Fill remaining CoPI slots from the union of domain pools."""
    pool   = sorted({r for d in domains for r in domain_to_researchers.get(d, [])})
    cands  = eligible(pool, exclude)
    chosen = []
    for r in cands:
        if len(chosen) >= need_k:
            break
        chosen.append(r)
        exclude.add(r)
    return chosen

failed_rows: list[int] = []

for i in range(len(df)):
    domains = split_project_domains(df.at[i, COL_DOMAIN])
    if not domains:
        failed_rows.append(i)
        continue

    k_copi = random.choice(COPI_CHOICES)

    # ── Choose PI ──────────────────────────────────────────────
    pi = choose_pi(domains)
    if pi is None:
        failed_rows.append(i)
        continue

    exclude = {pi}
    copis: list[str] = []

    # ── Cross-domain rule: cover each domain not held by PI ───
    if len(domains) >= 2:
        pi_domains = {d for d, rs in domain_to_researchers.items() if pi in rs}
        for d in domains:
            if d in pi_domains:
                continue            # PI already covers this domain
            r = pick_from_domain(d, exclude)
            if r is None:
                copis = []
                break
            copis.append(r)
            exclude.add(r)

        if not copis and len(domains) >= 2:
            failed_rows.append(i)
            continue

    # ── Trim if cross-domain produced more than k_copi ────────
    if len(copis) > k_copi:
        copis   = copis[:k_copi]
        exclude = {pi} | set(copis)

    # ── Fill remaining slots ───────────────────────────────────
    need = k_copi - len(copis)
    if need > 0:
        copis.extend(pick_extra(domains, exclude, need))

    # ── Fallback: relax to 2 if we couldn't reach 3 ───────────
    if len(copis) < 2 and k_copi == 3:
        k_copi = 2
        need   = k_copi - len(copis)
        if need > 0:
            copis.extend(pick_extra(domains, exclude, need))

    # ── Must have at least 2 CoPIs ────────────────────────────
    if len(copis) < 2:
        failed_rows.append(i)
        continue

    # Deduplicate and cap
    copis = list(dict.fromkeys([c for c in copis if c and c != pi]))[:k_copi]

    df.at[i, COL_PI]   = pi
    df.at[i, COL_COPI] = ";".join(copis)
    df.at[i, COL_ALL]  = ";".join([pi] + copis)

    pi_count[pi]    += 1
    total_count[pi] += 1
    for r in copis:
        total_count[r] += 1

# Drop rows where assignment was not possible
df = df.drop(index=failed_rows).reset_index(drop=True)
print(f"  Assignment complete.")
print(f"  Dropped (no valid PI/CoPI): {len(failed_rows)}")
print(f"  Projects with full assignment: {len(df)}")

# ── Assignment stats ───────────────────────────────────────────────────
assigned_res = [r for r in all_researchers if total_count[r] > 0]
pi_res       = [r for r in all_researchers if pi_count[r]    > 0]
print(f"  Researchers assigned to at least 1 project: {len(assigned_res)}")
print(f"  Researchers who served as PI: {len(pi_res)}")
print(f"  Researchers unused (forced): {len(unused_researchers)}")

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                            # 10) AGENCY ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
df[COL_AGENCY] = df.apply(
    lambda r: choose_agency(r[COL_DOMAIN], r[COL_TYPE]), axis=1
)

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                          # 11) ASSIGN GRANTS
#     Randomly select GRANTS_COUNT projects to fund.
#     Grant_ID is written back into the projects table
#     so both files share the same key.
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
print(f"\nAssigning {GRANTS_COUNT} grants ...")
N     = len(df)
kfund = min(max(GRANTS_COUNT, 0), N)

funded_idx = sorted(random.sample(range(N), kfund)) if kfund > 0 else []
grant_ids  = [f"{GRANT_PREFIX}{str(j+1).zfill(6)}" for j in range(len(funded_idx))]

# Write Grant_ID back to projects (the foreign-key link)
df[COL_GRANT_ID] = ""
for idx, gid in zip(funded_idx, grant_ids):
    df.at[idx, COL_GRANT_ID] = gid

# Build grants dataframe
funded = df[df[COL_GRANT_ID].ne("")].copy()
funded["amount_usd_lakhs"] = np.round(
    np.random.uniform(AMT_MIN, AMT_MAX, size=len(funded)), 2
)
funded["is_cross_domain"]  = funded[COL_DOMAIN].apply(is_cross_domain)
funded["domains_involved"] = funded[COL_DOMAIN].apply(domains_involved)

grants_final = pd.DataFrame({
    "grant_id":                  funded[COL_GRANT_ID],     # GRANT-000001 …
    "project_id":                funded[COL_PROJECT_ID],   # PROJ-00001 … ← shared key
    "title":                     funded[COL_TITLE],
    "agency":                    funded[COL_AGENCY],
    "amount_usd_lakhs":          funded["amount_usd_lakhs"],
    "is_cross_domain":           funded["is_cross_domain"],
    "domains_involved":          funded["domains_involved"],
    "start_date":                funded[COL_SDATE],
    "end_date":                  funded[COL_EDATE],
    "status":                    funded[COL_STATUS],
    "principal_investigator_id": funded[COL_PI],
    "co_investigators":          funded[COL_COPI],
})
print(f"  {len(grants_final)} grants assigned")
print(f"  Amount range: {grants_final['amount_usd_lakhs'].min():.2f} – "
      f"{grants_final['amount_usd_lakhs'].max():.2f} lakhs")
print(f"  Cross-domain grants: {grants_final['is_cross_domain'].sum()}")

# Agency distribution
print("\n  Agency distribution:")
print(grants_final["agency"].value_counts().to_string())

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                              # 12) SAVE ALL OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
KEEP_COLS = [
    COL_PROJECT_ID, COL_GRANT_ID, COL_TITLE, COL_DOMAIN, COL_TYPE,
    COL_SDATE, COL_EDATE, COL_PI, COL_COPI,
]
df[KEEP_COLS].to_csv(OUT_PROJECTS_FINAL, index=False)
grants_final.to_csv(OUT_GRANTS_FINAL, index=False)

