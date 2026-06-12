# -*- coding: utf-8 -*-
# ========================================================================================================================
                                              # grants.py
# ========================================================================================================================

import csv
import itertools
import random
import re
from collections import Counter

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ========================================================================================================================
                                                # 1. PATHS
# ========================================================================================================================

INPUT_PROJECTS_CSV = "Projects.csv"

OUT_PROJECTS_FINAL = "Project_with_grant.csv"
OUT_GRANTS_FINAL   = "Grants.csv"

# ========================================================================================================================
                                                  # 2. COLUMN NAMES
# ========================================================================================================================

COL_PROJECT_ID = "project_id"
COL_GRANT_ID   = "grant_id"
COL_TITLE      = "project_title"
COL_DOMAIN     = "domain"
COL_TYPE       = "project_type"
COL_SDATE      = "start_date"
COL_EDATE      = "end_date"
COL_STATUS     = "status"
COL_PI         = "principal_investigator"
COL_COPI       = "co_investigators"
COL_ALL        = "All_Researchers_In_Project"
COL_AGENCY     = "agency"

# ========================================================================================================================
                                                  # 3. SETTINGS
# ========================================================================================================================

AMT_MIN, AMT_MAX = 5, 250
GRANT_PREFIX     = "GRANT-"

GRANT_PROB_BY_TYPE = {
    "external":    0.80,
    "internal":    0.35,
    "consultancy": 0.60,
}

# ========================================================================================================================
                                                    # 4. HELPERS
# ========================================================================================================================

_RE_WHITESPACE   = re.compile(r"\s+")
_RE_SPLIT_IDS    = re.compile(r"[;,|\s]+")
_RE_SPLIT_DOMAIN = re.compile(r"\+|,|;|\|")
_RE_RID          = re.compile(r"^R(\d+)$")
_RE_YEAR_ONLY    = re.compile(r"^\d{4}$")


def norm_str(x):
    return "" if pd.isna(x) else str(x).strip()


def clean_id(x):
    x = norm_str(x).replace("\u00A0"," ").replace("\u200B","")
    return _RE_WHITESPACE.sub(" ", x).strip()


def normalize_rid(x):
    s = norm_str(x).upper()
    if s in {"","NAN","NONE","NULL"}: return ""
    m = _RE_RID.match(s)
    return "R" + m.group(1).zfill(4) if m else s


def split_ids(cell):
    s = norm_str(cell)
    if not s: return []
    out, seen = [], set()
    for p in _RE_SPLIT_IDS.split(s):
        rid = normalize_rid(p)
        if rid and rid not in seen:
            seen.add(rid); out.append(rid)
    return out


def normalize_domain_name(d):
    return _RE_WHITESPACE.sub(" ", norm_str(d)).strip()


def split_project_domains(domain_cell):
    s = norm_str(domain_cell)
    if not s: return []
    parts = [normalize_domain_name(p) for p in _RE_SPLIT_DOMAIN.split(s)]
    seen, out = set(), []
    for p in parts:
        if p and p not in seen:
            seen.add(p); out.append(p)
    return out[:2]


def normalize_type(x):
    s = norm_str(x).lower()
    if "internal"  in s: return "internal"
    if "consult"   in s: return "consultancy"
    if "external"  in s: return "external"
    return s


def is_cross_domain(domain_str):
    return int(bool(_RE_SPLIT_DOMAIN.search(norm_str(domain_str))))


def domains_involved(domain_str):
    return ", ".join(split_project_domains(domain_str))


def parse_date(date_str, mode="start"):
    if pd.isna(date_str): return None
    s = str(date_str).strip()
    if not s or s.lower() in {"nan","none","nat"}: return None
    if _RE_YEAR_ONLY.fullmatch(s):
        return pd.Timestamp(f"{s}-01-01" if mode == "start" else f"{s}-12-31")
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return None if pd.isna(dt) else pd.Timestamp(dt)


def format_date(dt):
    return pd.Timestamp(dt).strftime("%d/%m/%Y")


def make_grant_dates_inside_project(project_start, project_end):
    """
    Grant start: within first 20% of project duration.
    Grant end:   at least 50% of project duration; at most project end date.
    """
    ps = parse_date(project_start, mode="start")
    pe = parse_date(project_end,   mode="end")
    if ps is None or pe is None or pe < ps:
        return project_start, project_end
    total_days = (pe - ps).days
    if total_days <= 30:
        return format_date(ps), format_date(pe)

    # Grant start: within first 20% of project duration
    max_start_offset = max(0, int(total_days * 0.20))
    grant_start = ps + pd.Timedelta(days=random.randint(0, max_start_offset))

    # Grant end: at least 50% of total project duration from project start
    min_duration = max(30, int(total_days * 0.50))
    min_end      = ps + pd.Timedelta(days=min_duration)
    if min_end > pe:
        grant_end = pe
    else:
        grant_end = min_end + pd.Timedelta(days=random.randint(0, (pe - min_end).days))
    if grant_end > pe:
        grant_end = pe

    return format_date(grant_start), format_date(grant_end)

# ========================================================================================================================
                                              # 5. AGENCY LOGIC
# ========================================================================================================================

CANON = {
    "CS": "Computer Science", "Computer Science": "Computer Science",
    "Engg": "Engineering",    "Engineering": "Engineering",
    "Math": "Mathematics",    "Mathematics": "Mathematics",
    "Physics": "Physics",     "Chemistry": "Chemistry",
    "Biology": "Biology",
    "Medical": "Medical Sciences",    "Medical Sciences": "Medical Sciences",
    "Agri": "Agriculture & Environment",
    "Agriculture": "Agriculture & Environment",
    "Agriculture & Environment": "Agriculture & Environment",
    "Arts": "Arts & Humanities", "Arts & Humanities": "Arts & Humanities",
    "Social": "Social Sciences", "Social Sciences": "Social Sciences",
}

# ── Agency → eligible domain labels ─────────────────────────────────────────
# DST-SERB Synergy and NSF Convergence are ONLY for cross-domain projects.
# All other agencies follow domain matching as before.
AGENCY_TO_DOMAINS = {
    "DST-SERB":         {"Computer Science","Engineering","Mathematics","Physics","Chemistry"},
    "DST-SERB Synergy": {"Interdisciplinary"},   # cross-domain only — enforced in choose_agency
    "Industry":         {"Computer Science","Engineering","Medical Sciences","Agriculture & Environment"},
    "Wellcome Trust":   {"Biology","Medical Sciences"},
    "NSF Convergence":  {"Interdisciplinary"},    # cross-domain only — enforced in choose_agency
    "Gates Foundation": {"Medical Sciences","Biology","Agriculture & Environment"},
    "UKRI":             {"Computer Science","Engineering","Physics","Social Sciences"},
    "ICMR":             {"Medical Sciences"},
    "DBT":              {"Biology","Medical Sciences"},
    "ICAR":             {"Agriculture & Environment"},
    "ISRO":             {"Physics","Engineering","Computer Science"},
    "UGC":              {"Arts & Humanities","Social Sciences"},
    "Horizon Europe":   {"Computer Science","Engineering","Physics"},
    "HHMI":             {"Biology","Medical Sciences"},
    "CSIR":             {"Chemistry"},
}

# Priority order for external agency selection (most specific first)
AGENCY_PRIORITY = [
    "ICMR","DBT","ICAR","ISRO","CSIR","UGC",
    "Wellcome Trust","HHMI","Gates Foundation",
    "NSF Convergence","UKRI","Horizon Europe",
    "DST-SERB Synergy","DST-SERB","Industry",
]


def allowed_agencies(domain_str: str) -> list[str]:
    parts    = list(dict.fromkeys([CANON.get(p, p) for p in split_project_domains(domain_str)]))
    is_inter = len(parts) >= 2      # cross-domain project
    labels   = set(parts)
    if is_inter:
        labels.add("Interdisciplinary")

    allowed = []
    for agency in AGENCY_PRIORITY:
        ok = AGENCY_TO_DOMAINS.get(agency, set())

        # DST-SERB Synergy: cross-domain projects only
        if agency == "DST-SERB Synergy":
            if is_inter:
                allowed.append(agency)
            continue

        # NSF Convergence: cross-domain projects only
        if agency == "NSF Convergence":
            if is_inter:
                allowed.append(agency)
            continue

        # All other agencies: match on domain labels
        if ok & labels:
            allowed.append(agency)

    return allowed


def choose_agency(domain_str: str, proj_type: str) -> str:
    t = normalize_type(proj_type)
    if t == "internal":    return "Institutional Funding"
    if t == "consultancy": return "Industry"
    if t == "external":
        ag = allowed_agencies(domain_str)
        return ag[0] if ag else "DST-SERB"
    return "DST-SERB"

# ========================================================================================================================
                                            # 6. LOAD PROJECTS
# ========================================================================================================================

print("Loading projects ...")
df = pd.read_csv(INPUT_PROJECTS_CSV)

required_cols = [COL_PROJECT_ID, COL_TITLE, COL_DOMAIN, COL_TYPE,
                 COL_SDATE, COL_EDATE, COL_STATUS, COL_PI, COL_COPI]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(
        f"Project file must already contain PI and Co-PI columns. "
        f"Missing: {missing}"
    )

for col in (COL_GRANT_ID, COL_AGENCY, COL_ALL):
    if col not in df.columns: df[col] = ""

df[COL_PROJECT_ID] = df[COL_PROJECT_ID].apply(clean_id)
df[COL_TITLE]      = df[COL_TITLE].apply(clean_id)
df[COL_DOMAIN]     = df[COL_DOMAIN].apply(normalize_domain_name)
df[COL_TYPE]       = df[COL_TYPE].astype(str).str.strip()
df[COL_PI]         = df[COL_PI].apply(normalize_rid)
df[COL_COPI]       = df[COL_COPI].astype(str).str.strip()

before = len(df)
df = df.drop_duplicates(subset=[COL_PROJECT_ID], keep="first").reset_index(drop=True)
print(f"Projects loaded:                           {before}")
print(f"After project_id deduplication:            {len(df)}")

valid_team_mask = (df[COL_PI].notna() & df[COL_PI].ne("")
                   & df[COL_COPI].notna() & df[COL_COPI].ne(""))
before_team = len(df)
df = df[valid_team_mask].reset_index(drop=True)
print(f"After team validation:                     {len(df)}")
print(f"Dropped (no PI/CoPI):                      {before_team - len(df)}")

df[COL_ALL]    = df.apply(lambda r: ";".join([x for x in [r[COL_PI]] + split_ids(r[COL_COPI]) if x]), axis=1)
df[COL_AGENCY] = df.apply(lambda r: choose_agency(r[COL_DOMAIN], r[COL_TYPE]), axis=1)

print(f"\nCross-domain projects: {df[COL_DOMAIN].apply(is_cross_domain).sum()}")
print("Agency distribution (pre-grant assignment):")
print(df[COL_AGENCY].value_counts().to_string())

# ========================================================================================================================
                                            # 7. ASSIGN GRANTS
# ========================================================================================================================

print("\nAssigning grants based on project type probabilities ...")

df[COL_GRANT_ID] = ""
funded_rows = []

for _, row in df.iterrows():
    prob = GRANT_PROB_BY_TYPE.get(normalize_type(row[COL_TYPE]), 0.50)
    if random.random() < prob:
        gs, ge = make_grant_dates_inside_project(row[COL_SDATE], row[COL_EDATE])
        funded_rows.append({
            "project_id":       row[COL_PROJECT_ID],
            "start_date":       gs,
            "end_date":         ge,
            "grant_start_dt":   parse_date(gs,       mode="start"),
            "project_start_dt": parse_date(row[COL_SDATE], mode="start"),
        })

funded_order = pd.DataFrame(funded_rows)

pid_to_pos = {pid: i for i, pid in enumerate(df[COL_PROJECT_ID])}
grant_date_lookup = {}

if not funded_order.empty:
    funded_order = funded_order.sort_values(
        by=["grant_start_dt","project_start_dt","project_id"],
        ascending=[True,True,True],
    ).reset_index(drop=True)

    for position, (_, row) in enumerate(funded_order.iterrows(), start=1):
        gid = f"{GRANT_PREFIX}{position:06d}"
        pid = row["project_id"]
        pos = pid_to_pos.get(pid)
        if pos is not None:
            df.loc[pos, COL_GRANT_ID] = gid
        grant_date_lookup[pid] = (row["start_date"], row["end_date"])

print(f"Grants assigned: {len(funded_order)}")

# ========================================================================================================================
                                            # 8. BUILD GRANTS DATAFRAME
# ========================================================================================================================

funded = df[df[COL_GRANT_ID].ne("")].copy()
funded["amount_in_lakhs"]  = np.round(np.random.uniform(AMT_MIN, AMT_MAX, size=len(funded)), 2)
funded["is_cross_domain"]  = funded[COL_DOMAIN].apply(is_cross_domain)
funded["domains_involved"] = funded[COL_DOMAIN].apply(domains_involved)
funded["start_date"]       = funded[COL_PROJECT_ID].map(lambda p: grant_date_lookup.get(p,("",""))[0])
funded["end_date"]         = funded[COL_PROJECT_ID].map(lambda p: grant_date_lookup.get(p,("",""))[1])

grants_final = pd.DataFrame({
    "grant_id":              funded[COL_GRANT_ID],
    "project_id":            funded[COL_PROJECT_ID],
    "title":                 funded[COL_TITLE],
    "agency":                funded[COL_AGENCY],
    "amount_in_lakhs":       funded["amount_in_lakhs"],
    "is_cross_domain":       funded["is_cross_domain"],
    "domains_involved":      funded["domains_involved"],
    "start_date":            funded["start_date"],
    "end_date":              funded["end_date"],
    "status":                funded[COL_STATUS],
    "principal_investigator": funded[COL_PI],
    "co_investigators":      funded[COL_COPI],
})

if not grants_final.empty:
    grants_final["_order"] = (
        grants_final["grant_id"].str.extract(r"(\d+)")[0].astype(int)
    )
    grants_final = (grants_final.sort_values("_order")
                                .drop(columns=["_order"])
                                .reset_index(drop=True))

print(f"Grants in final table: {len(grants_final)}")
if not grants_final.empty:
    print(f"Amount range: {grants_final['amount_in_lakhs'].min():.2f} – "
          f"{grants_final['amount_in_lakhs'].max():.2f} lakhs")
    print(f"Cross-domain grants: {grants_final['is_cross_domain'].sum()}")
    print("\nAgency distribution:")
    print(grants_final["agency"].value_counts().to_string())

# Agency constraint validation for cross-domain
if not grants_final.empty:
    cross_grants = grants_final[grants_final["is_cross_domain"] == 1]
    non_cross_grants = grants_final[grants_final["is_cross_domain"] == 0]
    # Verify DST-SERB Synergy and NSF Convergence only appear in cross-domain
    exclusive_agencies = {"DST-SERB Synergy", "NSF Convergence"}
    violations = non_cross_grants[non_cross_grants["agency"].isin(exclusive_agencies)]
    print(f"\nAgency constraint validation:")
    print(f"  Cross-domain grants:               {len(cross_grants)}")
    print(f"  DST-SERB Synergy / NSF Convergence used in cross-domain grants: "
          f"{cross_grants['agency'].isin(exclusive_agencies).sum()}")
    if len(violations):
        print(f"  VIOLATIONS (exclusive agency on single-domain): {len(violations)}")
    else:
        print("  No violations — DST-SERB Synergy and NSF Convergence "
              "only assigned to cross-domain projects.")

# ========================================================================================================================
                                    # 9. BUILD PROJECT COLLABORATION EDGELIST
# ========================================================================================================================

print("\nBuilding project collaboration edgelist ...")
edges = Counter()

for row in df.itertuples(index=False):
    pi    = normalize_rid(getattr(row, COL_PI, ""))
    copis = split_ids(getattr(row, COL_COPI, ""))
    team  = list(dict.fromkeys([r for r in ([pi] if pi else []) + copis if r]))
    for a, b in itertools.combinations(sorted(team), 2):
        edges[(a, b)] += 1

edge_df = pd.DataFrame(
    [{"Source": a, "Target": b, "Weight": int(w)} for (a, b), w in edges.items()],
    columns=["Source","Target","Weight"],
)
if not edge_df.empty:
    edge_df = edge_df.sort_values(["Weight","Source","Target"],
                                  ascending=[False,True,True]).reset_index(drop=True)

print(f"Unique project collaboration edges: {len(edge_df)}")
if not edge_df.empty:
    print(f"Max edge weight: {edge_df['Weight'].max()}")
    print(f"Avg edge weight: {edge_df['Weight'].mean():.2f}")

# ========================================================================================================================
                                          # 10. TEMPORAL VALIDATION
# ========================================================================================================================

def validate_grant_inside_project(ps, pe, gs, ge):
    if any(x is None for x in (ps, pe, gs, ge)): return False
    return ps <= gs <= ge <= pe


project_date_lookup = dict(zip(
    df[COL_PROJECT_ID],
    zip(df[COL_SDATE].apply(lambda x: parse_date(x, mode="start")),
        df[COL_EDATE].apply(lambda x: parse_date(x, mode="end"))),
))

invalid_grants = []
if not grants_final.empty:
    for row in grants_final.itertuples(index=False):
        pid   = row.project_id
        dates = project_date_lookup.get(pid)
        if dates is None: invalid_grants.append(row.grant_id); continue
        ps, pe = dates
        gs = parse_date(row.start_date, mode="start")
        ge = parse_date(row.end_date,   mode="start")
        if not validate_grant_inside_project(ps, pe, gs, ge):
            invalid_grants.append(row.grant_id)

print("\nTemporal validation:")
print(f"Invalid grant date rows: {len(invalid_grants)}")
if invalid_grants:  print("Invalid Grant IDs sample:", invalid_grants[:20])
else:               print("All grant dates fall inside their project periods.")

# ========================================================================================================================
                                              # 11. SAVE OUTPUTS
# ========================================================================================================================

KEEP_COLS = [COL_PROJECT_ID, COL_GRANT_ID, COL_TITLE, COL_DOMAIN,
             COL_TYPE, COL_SDATE, COL_EDATE, COL_STATUS, COL_PI, COL_COPI]

df[KEEP_COLS].to_csv(OUT_PROJECTS_FINAL, index=False, quoting=csv.QUOTE_MINIMAL)
grants_final.to_csv(OUT_GRANTS_FINAL,    index=False, quoting=csv.QUOTE_MINIMAL)

print(f"\n{'─'*60}")
print("Files written:")
print(f"  {OUT_PROJECTS_FINAL:<38} {len(df)} rows")
print(f"  {OUT_GRANTS_FINAL:<38} {len(grants_final)} rows")
print(f"{'─'*60}")
print(f"\nKey link: {OUT_PROJECTS_FINAL}.project_id == {OUT_GRANTS_FINAL}.project_id")

print("\nFirst 10 grants:")
try:    display(grants_final.head(10))
except: print(grants_final.head(10))
