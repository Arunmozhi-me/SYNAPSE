# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
from itertools import combinations
from collections import defaultdict

# ====================================================================================================================================================================================
                                                                          # 1. CONFIG
# ====================================================================================================================================================================================      
#SEED = 42
#rng = np.random.default_rng(SEED)

PROFILES_FILE = "/content/researchers_profile.csv"
PROJECTS_FILE = "/content/Projects.csv"
GRANTS_FILE   = "/content/Grants.csv"

OUTPUT_FILE   = "interactions.csv"

# Informal interaction controls
INFORMAL_DEPT_LINKS_PER_RESEARCHER        = 2
INFORMAL_DOMAIN_LINKS_PER_RESEARCHER      = 1
INFORMAL_CROSS_DOMAIN_LINKS_PER_RESEARCHER = 1
INFORMAL_WINDOW_PROB                      = 0.08
MIN_OVERLAP_DAYS                          = 14

# Formal meeting scope controls
FULL_TEAM_PROB = 0.45
SUBGROUP_PROB  = 0.35
PAIR_ONLY_PROB = 0.20

# ====================================================================================================================================================================================
                                                                          # 2. HELPERS
# ====================================================================================================================================================================================
def find_col(df, candidates, required=True):
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in cols_lower:
            return cols_lower[key]
    if required:
        raise KeyError(f"None of these columns found: {candidates}")
    return None


def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def parse_date(x, mode="start"):
    if pd.isna(x):
        return pd.NaT
    x = str(x).strip()
    if x == "" or x.lower() in {"nan", "none", "nat"}:
        return pd.NaT

    # year only
    if re.fullmatch(r"\d{4}", x):
        return pd.Timestamp(f"{x}-01-01") if mode == "start" else pd.Timestamp(f"{x}-12-31")

    # general parse
    dt = pd.to_datetime(x, errors="coerce", dayfirst=False)
    return dt


def parse_researcher_ids(x):
    if pd.isna(x):
        return []
    x = str(x)
    ids = re.findall(r"R\d+", x, flags=re.IGNORECASE)
    ids = [i.upper() for i in ids]
    return list(dict.fromkeys(ids))


def overlap_interval(start1, end1, start2, end2, start3=None, end3=None):
    starts = [start1, start2]
    ends   = [end1, end2]

    if start3 is not None and pd.notna(start3):
        starts.append(start3)
    if end3 is not None and pd.notna(end3):
        ends.append(end3)

    ov_start = max(starts)
    ov_end   = min(ends)

    if ov_end < ov_start:
        return None, None
    return ov_start, ov_end


def get_fortnight_windows(start_date, end_date):
    start_date = pd.Timestamp(start_date).normalize()
    end_date   = pd.Timestamp(end_date).normalize()

    windows = []
    cur = pd.Timestamp(year=start_date.year, month=start_date.month, day=1)

    while cur <= end_date:
        month_start = cur
        month_end = month_start + pd.offsets.MonthEnd(0)

        first_start = month_start
        first_end   = min(month_start + pd.Timedelta(days=13), month_end)

        second_start = month_start + pd.Timedelta(days=14)
        second_end   = month_end

        halves = [
            ("F1", first_start, first_end),
            ("F2", second_start, second_end)
        ]

        for label, w_start, w_end in halves:
            if w_start <= end_date and w_end >= start_date:
                actual_start = max(w_start, start_date)
                actual_end   = min(w_end, end_date)
                windows.append({
                    "fortnight_id":    f"{w_start.year}-{w_start.month:02d}-{label}",
                    "fortnight_start": actual_start,
                    "fortnight_end":   actual_end
                })

        cur = month_start + pd.offsets.MonthBegin(1)

    return windows


def role_relation(role_a, role_b, years_exp_a=None, years_exp_b=None):
    if role_a == "PI" and role_b == "PI":
        return "PI-PI"

    if (role_a == "PI" and role_b == "CoI") or (role_a == "CoI" and role_b == "PI"):
        return "PI-CoI"

    if years_exp_a is not None and years_exp_b is not None:
        if abs(years_exp_a - years_exp_b) >= 8:
            return "mentor-mentee"

    return "peer"


def choose_interaction_date(fortnight_start, fortnight_end):
    days = (fortnight_end - fortnight_start).days
    offset = int(rng.integers(0, days + 1)) if days > 0 else 0
    return (fortnight_start + pd.Timedelta(days=offset)).normalize()


def choose_meeting_scope(participants):
    n = len(participants)
    if n <= 2:
        return sorted(participants)

    r = rng.random()
    if r < FULL_TEAM_PROB:
        return sorted(participants)
    elif r < FULL_TEAM_PROB + SUBGROUP_PROB:
        subgroup_size = int(rng.integers(2, n))
        subgroup = list(rng.choice(participants, size=subgroup_size, replace=False))
        return sorted(subgroup)
    else:
        pair = list(rng.choice(participants, size=2, replace=False))
        return sorted(pair)


def get_progress_phase(progress):
    if pd.isna(progress):
        return "independent"
    if progress < 0.20:
        return "early"
    elif progress < 0.80:
        return "middle"
    return "late"


# =========================================================================================================================================================================================================================
                                                                      # 3. INTERACTION TYPE RULES
# =========================================================================================================================================================================================================================
def project_type_by_phase(progress, relation):
    if progress < 0.20:
        choices = [
            "kickoff_meeting",
            "planning_discussion",
            "brainstorming"
        ]
    elif progress < 0.80:
        if relation == "PI-CoI":
            choices = [
                "technical_discussion",
                "coordination",
                "mentoring",
                "review_meeting"
            ]
        else:
            choices = [
                "technical_discussion",
                "coordination",
                "knowledge_exchange",
                "review_meeting"
            ]
    else:
        choices = [
            "review_meeting",
            "manuscript_preparation",
            "closure_discussion",
            "submission",
            "consolidation",
            "validation"
        ]
    return rng.choice(choices)


def grant_type_by_phase(progress):
    if progress < 0.20:
        choices = [
            "proposal_planning",
            "budget_discussion",
            "kickoff_meeting"
        ]
    elif progress < 0.80:
        choices = [
            "grant_coordination",
            "progress_review",
            "data_discussion",
            "review_meeting"
        ]
    else:
        choices = [
            "reporting",
            "paper_drafting",
            "final_review",
            "closure_discussion",
            "submission",
            "consolidation",
            "validation"
        ]
    return rng.choice(choices)


def informal_type(meta1, meta2):
    y1 = meta1.get("years_exp", 0)
    y2 = meta2.get("years_exp", 0)

    if abs(y1 - y2) >= 8:
        choices = [
            "mentoring",
            "knowledge_exchange",
            "seminar_discussion"
        ]
    else:
        choices = [
            "informal_meeting",
            "casual_discussion",
            "peer_discussion",
            "knowledge_exchange",
            "seminar_discussion"
        ]
    return rng.choice(choices)


# ============================================================
# OPTIONAL LABEL MAP FOR FUTURE ANALYSIS
# ============================================================
EXPECTED_PHASE_BY_TYPE = {
    "kickoff_meeting":       "early",
    "planning_discussion":   "early",
    "brainstorming":         "early",
    "proposal_planning":     "early",
    "budget_discussion":     "early",

    "technical_discussion":  "middle",
    "coordination":          "middle",
    "mentoring":             "middle",
    "knowledge_exchange":    "middle",
    "grant_coordination":    "middle",
    "progress_review":       "middle",
    "data_discussion":       "middle",
    "review_meeting":        "middle",

    "manuscript_preparation": "late",
    "closure_discussion":    "late",
    "submission":            "late",
    "consolidation":         "late",
    "validation":            "late",
    "reporting":             "late",
    "paper_drafting":        "late",
    "final_review":          "late",

    "informal_meeting":      "independent",
    "casual_discussion":     "independent",
    "peer_discussion":       "independent",
    "seminar_discussion":    "independent"
}


def future_progress_label(progress, interaction_type, source_layer):
    if source_layer == "informal":
        return "independent"

    expected_phase = EXPECTED_PHASE_BY_TYPE.get(interaction_type, None)
    actual_phase   = get_progress_phase(progress)

    order = {"early": 1, "middle": 2, "late": 3}

    if expected_phase not in order or actual_phase not in order:
        return "unknown"

    if order[actual_phase] < order[expected_phase]:
        return "before_time"
    elif order[actual_phase] == order[expected_phase]:
        return "on_time"
    else:
        return "slow"


# ====================================================================================================================================================================================
                                                                            # 4. LOAD DATA
# ====================================================================================================================================================================================
profiles = pd.read_csv(PROFILES_FILE)
projects = pd.read_csv(PROJECTS_FILE)
grants   = pd.read_csv(GRANTS_FILE)

profiles.columns = [c.strip() for c in profiles.columns]
projects.columns = [c.strip() for c in projects.columns]
grants.columns   = [c.strip() for c in grants.columns]

# ====================================================================================================================================================================================
                                                                    # 5. DETECT COLUMNS
# ====================================================================================================================================================================================
P_RID   = find_col(profiles, ["r_id", "researcher_id"])
P_START = find_col(profiles, ["career_start_year", "career_start_date"])
P_END   = find_col(profiles, ["career_end_date", "career_end_year"], required=False)
P_ACT   = find_col(profiles, ["is_active"], required=False)
P_DOM   = find_col(profiles, ["primary_domain", "domain"], required=False)
P_DEPT  = find_col(profiles, ["d_name", "department"], required=False)
P_YEXP  = find_col(profiles, ["years_exp"], required=False)

PR_ID    = find_col(projects, ["Project_ID", "project_id"])
PR_START = find_col(projects, ["Start_Date", "start_date"])
PR_END   = find_col(projects, ["End_Date", "end_date"])
PR_PI    = find_col(projects, ["Principal_Investigator", "principal_investigator"])
PR_COI   = find_col(projects, ["Co_Investigators", "co_investigators"], required=False)

G_ID    = find_col(grants, ["grant_id"])
G_START = find_col(grants, ["start_date"])
G_END   = find_col(grants, ["end_date"])
G_PI    = find_col(grants, ["principal_investigator_id", "principal_investigator"])
G_COI   = find_col(grants, ["co_investigators"], required=False)

# ============================================================
# GLOBAL END DATE
# ============================================================
project_end_dates = projects[PR_END].apply(lambda x: parse_date(x, mode="end"))
grant_end_dates   = grants[G_END].apply(lambda x: parse_date(x, mode="end"))

max_project_end = project_end_dates.dropna().max() if project_end_dates.notna().any() else pd.Timestamp("2025-12-31")
max_grant_end   = grant_end_dates.dropna().max()   if grant_end_dates.notna().any()   else pd.Timestamp("2025-12-31")
GLOBAL_END = max(max_project_end, max_grant_end)

# ====================================================================================================================================================================================
                                                                              # 6. RESEARCHER METADATA
# ====================================================================================================================================================================================
researcher_meta = {}

for _, row in profiles.iterrows():
    rid = safe_str(row[P_RID]).upper()
    if rid == "":
        continue

    start_dt = parse_date(row[P_START], mode="start")
    end_dt   = parse_date(row[P_END], mode="end") if P_END else pd.NaT

    if pd.isna(start_dt):
        continue

    is_active = True
    if P_ACT is not None:
        val = safe_str(row[P_ACT]).lower()
        is_active = val in {"1", "true", "yes", "y"}

    if pd.isna(end_dt):
        end_dt = GLOBAL_END

    if end_dt < start_dt:
        end_dt = start_dt

    years_exp = 0.0
    if P_YEXP is not None and pd.notna(row[P_YEXP]):
        try:
            years_exp = float(row[P_YEXP])
        except:
            years_exp = 0.0

    researcher_meta[rid] = {
        "start":     start_dt,
        "end":       end_dt,
        "domain":    safe_str(row[P_DOM])  if P_DOM  else "",
        "dept":      safe_str(row[P_DEPT]) if P_DEPT else "",
        "years_exp": years_exp
    }

valid_researchers = set(researcher_meta.keys())

# ====================================================================================================================================================================================
                                                                      # 7. GENERATION
# ====================================================================================================================================================================================
rows = []
interaction_counter = 1
event_counter       = 1


def add_formal_events(df, source_layer, id_col, start_col, end_col, pi_col, coi_col):
    global interaction_counter, event_counter

    for _, row in df.iterrows():
        src_id    = safe_str(row[id_col])
        ctx_start = parse_date(row[start_col], mode="start")
        ctx_end   = parse_date(row[end_col],   mode="end")

        if pd.isna(ctx_start) or pd.isna(ctx_end) or ctx_end < ctx_start:
            continue

        pi_ids  = [r for r in parse_researcher_ids(row[pi_col])  if r in valid_researchers]
        coi_ids = [r for r in parse_researcher_ids(row[coi_col]) if r in valid_researchers] if coi_col else []

        participants = list(dict.fromkeys(pi_ids + coi_ids))
        if len(participants) < 2:
            continue

        role_map = {}
        for r in pi_ids:
            role_map[r] = "PI"
        for r in coi_ids:
            if r not in role_map:
                role_map[r] = "CoI"

        windows    = get_fortnight_windows(ctx_start, ctx_end)
        total_days = max((ctx_end - ctx_start).days, 1)

        for w in windows:
            fw_start = w["fortnight_start"]
            fw_end   = w["fortnight_end"]

            progress = max(0, min(1, (fw_start - ctx_start).days / total_days))

            base_p = 0.32 if source_layer == "project" else 0.26
            if progress < 0.15 or progress > 0.85:
                base_p += 0.05

            if rng.random() > min(base_p, 0.90):
                continue

            attendees = choose_meeting_scope(participants)
            if len(attendees) < 2:
                continue

            valid_attendees = []
            for r in attendees:
                m = researcher_meta[r]
                ov_start, _ = overlap_interval(m["start"], m["end"], fw_start, fw_end)
                if ov_start is not None:
                    valid_attendees.append(r)

            valid_attendees = sorted(set(valid_attendees))
            if len(valid_attendees) < 2:
                continue

            interaction_date = choose_interaction_date(fw_start, fw_end)

            # one actual meeting event
            event_id = f"EVT{event_counter:07d}"
            event_counter += 1

            # choose one interaction type for the whole event
            if source_layer == "project":
                if len(valid_attendees) >= 2:
                    r_a, r_b = valid_attendees[0], valid_attendees[1]
                    rel = role_relation(
                        role_map.get(r_a, "CoI"),
                        role_map.get(r_b, "CoI"),
                        researcher_meta[r_a]["years_exp"],
                        researcher_meta[r_b]["years_exp"]
                    )
                else:
                    rel = "peer"
                i_type = project_type_by_phase(progress, rel)
            else:
                i_type = grant_type_by_phase(progress)

            for r1, r2 in combinations(valid_attendees, 2):
                rows.append({
                    "interaction_id":   f"INT{interaction_counter:07d}",
                    "event_id":         event_id,
                    "r_id1":            r1,
                    "r_id2":            r2,
                    "fortnight_start":  fw_start,
                    "interaction_date": interaction_date,
                    "progress":         round(float(progress), 4),
                    "interaction_type": i_type,
                    "source_layer":     source_layer,
                    "source_id":        src_id
                })
                interaction_counter += 1


def add_informal_events():
    global interaction_counter, event_counter

    by_dept   = defaultdict(list)
    by_domain = defaultdict(list)
    all_researchers = list(researcher_meta.keys())

    for rid, meta in researcher_meta.items():
        if meta["dept"]:
            by_dept[meta["dept"]].append(rid)
        if meta["domain"]:
            by_domain[meta["domain"]].append(rid)

    informal_pairs = set()

    for rid, meta in researcher_meta.items():
        # same department
        dept_pool = [x for x in by_dept.get(meta["dept"], []) if x != rid]
        rng.shuffle(dept_pool)
        for other in dept_pool[:INFORMAL_DEPT_LINKS_PER_RESEARCHER]:
            informal_pairs.add(tuple(sorted((rid, other))))

        # same domain
        dom_pool  = [x for x in by_domain.get(meta["domain"], []) if x != rid]
        rng.shuffle(dom_pool)
        added_dom = 0
        for other in dom_pool:
            pair = tuple(sorted((rid, other)))
            if pair in informal_pairs:
                continue
            informal_pairs.add(pair)
            added_dom += 1
            if added_dom >= INFORMAL_DOMAIN_LINKS_PER_RESEARCHER:
                break

        # cross domain
        cross_pool = [
            x for x in all_researchers
            if x != rid and researcher_meta[x].get("domain", "") != meta.get("domain", "")
        ]
        rng.shuffle(cross_pool)
        added_cross = 0
        for other in cross_pool:
            pair = tuple(sorted((rid, other)))
            if pair in informal_pairs:
                continue
            informal_pairs.add(pair)
            added_cross += 1
            if added_cross >= INFORMAL_CROSS_DOMAIN_LINKS_PER_RESEARCHER:
                break

    for r1, r2 in informal_pairs:
        m1 = researcher_meta[r1]
        m2 = researcher_meta[r2]

        ov_start, ov_end = overlap_interval(m1["start"], m1["end"], m2["start"], m2["end"])
        if ov_start is None:
            continue

        if (ov_end - ov_start).days < MIN_OVERLAP_DAYS:
            continue

        windows = get_fortnight_windows(ov_start, ov_end)

        for w in windows:
            if rng.random() > INFORMAL_WINDOW_PROB:
                continue

            fw_start         = w["fortnight_start"]
            fw_end           = w["fortnight_end"]
            interaction_date = choose_interaction_date(fw_start, fw_end)
            i_type           = informal_type(m1, m2)

            event_id = f"EVT{event_counter:07d}"
            event_counter += 1

            rows.append({
                "interaction_id":   f"INT{interaction_counter:07d}",
                "event_id":         event_id,
                "r_id1":            r1,
                "r_id2":            r2,
                "fortnight_start":  fw_start,
                "interaction_date": interaction_date,
                "progress":         np.nan,
                "interaction_type": i_type,
                "source_layer":     "informal",
                "source_id":        "NA"
            })
            interaction_counter += 1


# ====================================================================================================================================================================================
                                                                          # 8.RUN GENERATION
# ====================================================================================================================================================================================
add_formal_events(projects, "project", PR_ID, PR_START, PR_END, PR_PI, PR_COI)
add_formal_events(grants,   "grant",   G_ID,  G_START,  G_END,  G_PI,  G_COI)
add_informal_events()

interactions = pd.DataFrame(rows)

if interactions.empty:
    raise ValueError("No interaction rows were generated. Check input files and column names.")

# sort ids in pair columns for consistency
interactions[["r_id1", "r_id2"]] = np.sort(interactions[["r_id1", "r_id2"]], axis=1)

# exact deduplication
interactions = interactions.drop_duplicates(
    subset=[
        "event_id",
        "r_id1",
        "r_id2",
        "interaction_date",
        "interaction_type",
        "source_layer",
        "source_id"
    ]
).reset_index(drop=True)

interactions = interactions.sort_values(
    by=["interaction_date", "event_id", "r_id1", "r_id2"]
).reset_index(drop=True)

final_columns = [
    "interaction_id",
    "event_id",
    "r_id1",
    "r_id2",
    "fortnight_start",
    "interaction_date",
    "progress",
    "interaction_type",
    "source_layer",
    "source_id"
]

interactions = interactions[final_columns].copy()
interactions.to_csv(OUTPUT_FILE, index=False)

print(f"Saved: {OUTPUT_FILE}")
print(f"Rows:  {len(interactions):,}")
print("\nHead:")
print(interactions.head(10))

print("\nSource-layer counts:")
print(interactions["source_layer"].value_counts(dropna=False))

# optional preview for future trend labels
preview = interactions.copy()
preview["future_trend_label"] = preview.apply(
    lambda x: future_progress_label(x["progress"], x["interaction_type"], x["source_layer"]),
    axis=1
)

print("\nPreview with future trend labels:")
print(preview.head(15))
