# -*- coding: utf-8 -*-
import bisect
import re
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd


# ========================================================================================================================
                                                    # 1. CONFIG
# ========================================================================================================================

SEED = 42
rng = np.random.default_rng(SEED)

PROFILES_FILE  = "researchers_profile.csv"
EVOLUTION_FILE = "researcher_domain_evolution.csv"
PROJECTS_FILE  = "Project_with_grant.csv"
GRANTS_FILE    = "Grants.csv"
OUTPUT_FILE    = "interactions.csv"

TODAY = pd.Timestamp.today().normalize()


# ========================================================================================================================
                                  # 1.1 SIMPLE FORMAL INTERACTION CONTROLS
# ========================================================================================================================

FORMAL_PROJECT_WINDOW_PROB = 0.30
FORMAL_GRANT_WINDOW_PROB   = 0.25

FULL_TEAM_PROB = 0.65
SUBGROUP_PROB  = 0.30
PAIR_ONLY_PROB = 0.05


# ========================================================================================================================
                                # 1.2 SIMPLE INFORMAL INTERACTION CONTROLS
# ========================================================================================================================

GROUP_MIN_SIZE = 4
GROUP_MAX_SIZE = 6

# Simple group rates, not clustering targets
DEPT_GROUP_RATE   = 0.15
DOMAIN_GROUP_RATE = 0.05

# Probability that an informal pair interacts in a fortnight
INFORMAL_WINDOW_PROB     = 0.03
PRE_PROJECT_WINDOW_PROB  = 0.01

MIN_OVERLAP_DAYS = 14


# ========================================================================================================================
                                              # 2. BASIC HELPERS
# ========================================================================================================================

def find_col(df, candidates, required=True):
    cols_lower = {c.lower().strip(): c for c in df.columns}

    for cand in candidates:
        key = cand.lower().strip()
        if key in cols_lower:
            return cols_lower[key]

    if required:
        raise KeyError(f"None of these columns found: {candidates}")

    return None


def safe_str(x) -> str:
    return "" if pd.isna(x) else str(x).strip()


def normalize_rid(x) -> str:
    s = safe_str(x).upper()

    if s in {"", "NAN", "NONE"}:
        return ""

    m = re.match(r"^R(\d+)$", s)

    if m:
        return "R" + m.group(1).zfill(4)

    return s


def parse_date(x, mode="start"):
    if pd.isna(x):
        return pd.NaT

    x = str(x).strip()

    if not x or x.lower() in {"nan", "none", "nat"}:
        return pd.NaT

    if re.fullmatch(r"\d{4}", x):
        if mode == "start":
            return pd.Timestamp(f"{x}-01-01")
        else:
            return pd.Timestamp(f"{x}-12-31")

    return pd.to_datetime(x, errors="coerce", dayfirst=True)


def parse_researcher_ids(x) -> list:
    if pd.isna(x):
        return []

    ids = re.findall(r"R\d+", str(x), flags=re.IGNORECASE)
    ids = [normalize_rid(i) for i in ids]
    ids = [i for i in ids if i]

    return list(dict.fromkeys(ids))


def overlap_interval(start1, end1, start2, end2):
    ov_start = max(start1, start2)
    ov_end = min(end1, end2)

    if ov_end >= ov_start:
        return ov_start, ov_end

    return None, None


def get_fortnight_windows(start_date, end_date) -> list:
    start_date = pd.Timestamp(start_date).normalize()
    end_date = pd.Timestamp(end_date).normalize()

    windows = []
    cur = pd.Timestamp(year=start_date.year, month=start_date.month, day=1)

    while cur <= end_date:
        month_end = cur + pd.offsets.MonthEnd(0)

        halves = [
            (
                "F1",
                cur,
                min(cur + pd.Timedelta(days=13), month_end),
            ),
            (
                "F2",
                cur + pd.Timedelta(days=14),
                month_end,
            ),
        ]

        for label, w_start, w_end in halves:
            if w_start <= end_date and w_end >= start_date:
                windows.append({
                    "fortnight_id": f"{w_start.year}-{w_start.month:02d}-{label}",
                    "fortnight_start": max(w_start, start_date),
                    "fortnight_end": min(w_end, end_date),
                })

        cur = cur + pd.offsets.MonthBegin(1)

    return windows


def choose_interaction_date(fw_start, fw_end):
    days = (fw_end - fw_start).days
    offset = int(rng.integers(0, days + 1)) if days > 0 else 0

    return (fw_start + pd.Timedelta(days=offset)).normalize()


# ========================================================================================================================
                                          # 3. INTERACTION TYPE HELPERS
# ========================================================================================================================

def role_relation(role_a, role_b, years_exp_a=None, years_exp_b=None) -> str:
    if role_a == "PI" and role_b == "PI":
        return "PI-PI"

    if {role_a, role_b} == {"PI", "CoI"}:
        return "PI-CoI"

    if years_exp_a is not None and years_exp_b is not None:
        if abs(years_exp_a - years_exp_b) >= 8:
            return "mentor-mentee"

    return "peer"


def choose_meeting_scope(participants: list) -> list:
    n = len(participants)

    if n <= 2:
        return sorted(participants)

    r = rng.random()

    if r < FULL_TEAM_PROB:
        return sorted(participants)

    elif r < FULL_TEAM_PROB + SUBGROUP_PROB:
        size = int(rng.integers(2, n))
        return sorted(
            rng.choice(participants, size=size, replace=False).tolist()
        )

    else:
        return sorted(
            rng.choice(participants, size=2, replace=False).tolist()
        )


def project_type_by_phase(progress, relation) -> str:
    if progress < 0.20:
        return rng.choice([
            "kickoff_meeting",
            "planning_discussion",
            "brainstorming",
        ])

    if progress < 0.80:
        if relation == "PI-CoI":
            return rng.choice([
                "technical_discussion",
                "coordination",
                "mentoring",
                "review_meeting",
            ])

        return rng.choice([
            "technical_discussion",
            "coordination",
            "knowledge_exchange",
            "review_meeting",
        ])

    return rng.choice([
        "review_meeting",
        "manuscript_preparation",
        "closure_discussion",
        "submission",
        "consolidation",
        "validation",
    ])


def grant_type_by_phase(progress) -> str:
    if progress < 0.20:
        return rng.choice([
            "proposal_planning",
            "budget_discussion",
            "kickoff_meeting",
        ])

    if progress < 0.80:
        return rng.choice([
            "grant_coordination",
            "progress_review",
            "data_discussion",
            "review_meeting",
        ])

    return rng.choice([
        "reporting",
        "paper_drafting",
        "final_review",
        "closure_discussion",
        "submission",
        "consolidation",
        "validation",
    ])


def informal_type(meta1, meta2) -> str:
    y1 = meta1.get("years_exp", 0)
    y2 = meta2.get("years_exp", 0)

    if abs(y1 - y2) >= 8:
        return rng.choice([
            "mentoring",
            "knowledge_exchange",
            "seminar_discussion",
        ])

    return rng.choice([
        "informal_meeting",
        "casual_discussion",
        "peer_discussion",
        "knowledge_exchange",
        "seminar_discussion",
    ])


# ========================================================================================================================
                                                # 4. LOAD DATA
# ========================================================================================================================

print("Loading data...")

profiles  = pd.read_csv(PROFILES_FILE)
projects  = pd.read_csv(PROJECTS_FILE)
grants    = pd.read_csv(GRANTS_FILE)
evolution = pd.read_csv(EVOLUTION_FILE)

for df in (profiles, projects, grants, evolution):
    df.columns = df.columns.str.strip()

print(f"  Profiles  : {len(profiles):,}")
print(f"  Projects  : {len(projects):,}")
print(f"  Grants    : {len(grants):,}")
print(f"  Evolution : {len(evolution):,}")


# ========================================================================================================================
                                              # 5. DETECT COLUMNS
# ========================================================================================================================

P_RID   = find_col(profiles, ["r_id", "researcher_id"])
P_START = find_col(profiles, ["career_start_year", "career_start_date"])
P_END   = find_col(profiles, ["career_end_date", "career_end_year"], required=False)
P_ACT   = find_col(profiles, ["is_active"], required=False)
P_DOM   = find_col(profiles, ["primary_domain", "domain"], required=False)
P_DEPT  = find_col(profiles, ["d_name", "department"], required=False)
P_YEXP  = find_col(profiles, ["years_exp", "years_Exp"], required=False)

PR_ID    = find_col(projects, ["project_id", "Project_ID"])
PR_START = find_col(projects, ["start_date", "Start_Date"])
PR_END   = find_col(projects, ["end_date", "End_Date"])
PR_PI    = find_col(projects, ["principal_investigator", "Principal_Investigator"])
PR_COI   = find_col(projects, ["co_investigators", "Co_Investigators"], required=False)

G_ID    = find_col(grants, ["grant_id", "Grant_ID"])
G_START = find_col(grants, ["start_date", "Start_Date"])
G_END   = find_col(grants, ["end_date", "End_Date"])
G_PI    = find_col(
    grants,
    [
        "principal_investigator_id",
        "principal_investigator",
        "Principal_Investigator",
    ],
)
G_COI = find_col(grants, ["co_investigators", "Co_Investigators"], required=False)

E_RID    = find_col(evolution, ["r_id", "researcher_id"])
E_YEAR   = find_col(evolution, ["year"])
E_DOMAIN = find_col(
    evolution,
    ["current_dominant_domain", "dominant_domain", "domain"],
)


# ========================================================================================================================
                                            # 6. CLEAN EVOLUTION FILE
# ========================================================================================================================

evolution[E_RID] = evolution[E_RID].astype(str).apply(normalize_rid)
evolution[E_YEAR] = pd.to_numeric(evolution[E_YEAR], errors="coerce").astype("Int64")
evolution[E_DOMAIN] = evolution[E_DOMAIN].astype(str).str.strip()

evolution = evolution.dropna(subset=[E_RID, E_YEAR]).copy()
evolution[E_YEAR] = evolution[E_YEAR].astype(int)

domain_by_rid_year = evolution.set_index([E_RID, E_YEAR])[E_DOMAIN].to_dict()
available_years = sorted(evolution[E_YEAR].unique().tolist())


def _nearest_year(year: int) -> int:
    if not available_years:
        return year

    idx = bisect.bisect_left(available_years, year)

    if idx == 0:
        return available_years[0]

    if idx >= len(available_years):
        return available_years[-1]

    lo, hi = available_years[idx - 1], available_years[idx]

    return hi if (hi - year) <= (year - lo) else lo


def get_temporal_domain(rid: str, date_value, fallback: str = "") -> str:
    rid = normalize_rid(rid)

    if not rid:
        return fallback

    dt = parse_date(date_value, mode="start")

    if pd.isna(dt):
        return fallback

    year = _nearest_year(int(dt.year))

    return domain_by_rid_year.get((rid, year), fallback)


# ========================================================================================================================
                                              # 7. GLOBAL TEMPORAL LIMITS
# ========================================================================================================================

GLOBAL_END = TODAY

print(f"\nInteractions generated up to: {GLOBAL_END.date()}")

_proj_starts = projects[PR_START].apply(
    lambda x: parse_date(x, mode="start")
).dropna()

FIRST_PROJECT_START = (
    _proj_starts.min()
    if not _proj_starts.empty
    else GLOBAL_END
)

print(f"First project start date:     {FIRST_PROJECT_START.date()}")


# ========================================================================================================================
                                              # 8. RESEARCHER METADATA
# ========================================================================================================================

researcher_meta = {}

for _, row in profiles.iterrows():
    rid = normalize_rid(row[P_RID])

    if not rid:
        continue

    start_dt = parse_date(row[P_START], mode="start")

    if pd.isna(start_dt):
        continue

    end_dt = parse_date(row[P_END], mode="end") if P_END else pd.NaT

    is_active = True

    if P_ACT is not None:
        is_active = safe_str(row.get(P_ACT, "")).lower() in {
            "1",
            "true",
            "yes",
            "y",
        }

    if pd.isna(end_dt):
        end_dt = GLOBAL_END

    if end_dt < start_dt:
        end_dt = start_dt

    years_exp = 0.0

    if P_YEXP is not None and pd.notna(row.get(P_YEXP)):
        try:
            years_exp = float(row[P_YEXP])
        except Exception:
            years_exp = 0.0

    researcher_meta[rid] = {
        "start": start_dt,
        "end": end_dt,
        "domain": safe_str(row[P_DOM]) if P_DOM else "",
        "dept": safe_str(row[P_DEPT]) if P_DEPT else "",
        "years_exp": years_exp,
        "is_active": is_active,
    }

valid_researchers = set(researcher_meta.keys())

print(f"Valid researchers: {len(valid_researchers):,}")


# ========================================================================================================================
                                            # 9. ROW APPEND FUNCTION
# ========================================================================================================================

rows = []
event_counter = 1


def add_interaction_row(
    event_id,
    r1,
    r2,
    fw_start,
    interaction_date,
    progress,
    interaction_type,
    source_layer,
    source_id,
):
    r1 = normalize_rid(r1)
    r2 = normalize_rid(r2)

    if not r1 or not r2 or r1 == r2:
        return

    if r1 not in valid_researchers or r2 not in valid_researchers:
        return

    r1, r2 = sorted([r1, r2])

    d1 = get_temporal_domain(
        r1,
        interaction_date,
        researcher_meta.get(r1, {}).get("domain", ""),
    )

    d2 = get_temporal_domain(
        r2,
        interaction_date,
        researcher_meta.get(r2, {}).get("domain", ""),
    )

    rows.append({
        "interaction_id": "",
        "event_id": event_id,
        "r_id1": r1,
        "r_id2": r2,
        "r_id1_domain_at_interaction": d1,
        "r_id2_domain_at_interaction": d2,
        "fortnight_start": fw_start,
        "interaction_date": interaction_date,
        "interaction_year": int(pd.Timestamp(interaction_date).year),
        "progress": progress,
        "interaction_type": interaction_type,
        "source_layer": source_layer,
        "source_id": source_id,
    })


# ========================================================================================================================
                                              # 10. BUILD FORMAL PAIRS
# ========================================================================================================================

def build_formal_pairs(df, pi_col, coi_col) -> set:
    pairs = set()

    for _, row in df.iterrows():
        pi_ids = [
            r for r in parse_researcher_ids(row[pi_col])
            if r in valid_researchers
        ]

        coi_ids = (
            [
                r for r in parse_researcher_ids(row[coi_col])
                if r in valid_researchers
            ]
            if coi_col
            else []
        )

        participants = list(dict.fromkeys(pi_ids + coi_ids))

        if len(participants) < 2:
            continue

        for r1, r2 in combinations(participants, 2):
            pairs.add(tuple(sorted((r1, r2))))

    return pairs


# ========================================================================================================================
                                      # 11. FORMAL PROJECT / GRANT INTERACTIONS
# ========================================================================================================================

def add_formal_events(df, source_layer, id_col, start_col, end_col, pi_col, coi_col):
    global event_counter

    formal_rows_added = 0

    for _, row in df.iterrows():
        src_id = safe_str(row[id_col])

        ctx_start = parse_date(row[start_col], mode="start")
        ctx_end = parse_date(row[end_col], mode="end")

        if pd.isna(ctx_start) or pd.isna(ctx_end) or ctx_end < ctx_start:
            continue

        pi_ids = [
            r for r in parse_researcher_ids(row[pi_col])
            if r in valid_researchers
        ]

        coi_ids = (
            [
                r for r in parse_researcher_ids(row[coi_col])
                if r in valid_researchers
            ]
            if coi_col
            else []
        )

        participants = list(dict.fromkeys(pi_ids + coi_ids))

        if len(participants) < 2:
            continue

        role_map = {r: "PI" for r in pi_ids}

        for r in coi_ids:
            if r not in role_map:
                role_map[r] = "CoI"

        windows = get_fortnight_windows(ctx_start, ctx_end)
        total_days = max((ctx_end - ctx_start).days, 1)

        for w in windows:
            fw_start = w["fortnight_start"]
            fw_end = w["fortnight_end"]

            progress = max(
                0.0,
                min(1.0, (fw_start - ctx_start).days / total_days),
            )

            if source_layer == "project":
                base_p = FORMAL_PROJECT_WINDOW_PROB
            else:
                base_p = FORMAL_GRANT_WINDOW_PROB

            # Slightly more activity at beginning and end of formal work
            if progress < 0.15 or progress > 0.85:
                base_p += 0.05

            if rng.random() > min(base_p, 0.90):
                continue

            attendees = choose_meeting_scope(participants)

            if len(attendees) < 2:
                continue

            valid_attendees = sorted({
                r for r in attendees
                if overlap_interval(
                    researcher_meta[r]["start"],
                    researcher_meta[r]["end"],
                    fw_start,
                    fw_end,
                )[0] is not None
            })

            if len(valid_attendees) < 2:
                continue

            interaction_date = choose_interaction_date(fw_start, fw_end)

            event_id = f"EVT{event_counter:07d}"
            event_counter += 1

            r_a, r_b = valid_attendees[0], valid_attendees[1]

            if source_layer == "project":
                rel = role_relation(
                    role_map.get(r_a, "CoI"),
                    role_map.get(r_b, "CoI"),
                    researcher_meta[r_a]["years_exp"],
                    researcher_meta[r_b]["years_exp"],
                )

                i_type = project_type_by_phase(progress, rel)

            else:
                i_type = grant_type_by_phase(progress)

            for r1, r2 in combinations(valid_attendees, 2):
                add_interaction_row(
                    event_id=event_id,
                    r1=r1,
                    r2=r2,
                    fw_start=fw_start,
                    interaction_date=interaction_date,
                    progress=round(float(progress), 4),
                    interaction_type=i_type,
                    source_layer=source_layer,
                    source_id=src_id,
                )

                formal_rows_added += 1

    print(f"  Formal {source_layer} rows added: {formal_rows_added:,}")


# ========================================================================================================================
                                        # 12. SIMPLE INFORMAL INTERACTIONS
# ========================================================================================================================

def make_academic_groups(members, group_rate):
    """
    Create small academic groups.
    No explicit clustering coefficient is imposed.
    Triangles can appear naturally because group members interact repeatedly.
    """
    members = list(dict.fromkeys(members))

    if len(members) < GROUP_MIN_SIZE:
        return []

    n_groups = max(1, int(len(members) * group_rate))
    groups = []

    for _ in range(n_groups):
        size = int(
            rng.integers(
                GROUP_MIN_SIZE,
                min(GROUP_MAX_SIZE, len(members)) + 1,
            )
        )

        group = rng.choice(
            members,
            size=size,
            replace=False,
        ).tolist()

        group = sorted(list(dict.fromkeys(group)))

        if len(group) >= GROUP_MIN_SIZE:
            groups.append(group)

    return groups


def add_informal_events(formal_pairs: set):
    global event_counter

    print("\nGenerating simple informal interactions...")

    by_dept = defaultdict(list)
    by_domain = defaultdict(list)

    for rid, meta in researcher_meta.items():
        if meta["dept"]:
            by_dept[meta["dept"]].append(rid)

        if meta["domain"]:
            by_domain[meta["domain"]].append(rid)

    informal_pairs = set()

    # Department-based informal groups
    dept_group_count = 0

    for dept, members in by_dept.items():
        groups = make_academic_groups(
            members=members,
            group_rate=DEPT_GROUP_RATE,
        )

        dept_group_count += len(groups)

        for group in groups:
            for r1, r2 in combinations(group, 2):
                pair = tuple(sorted((r1, r2)))

                if pair in formal_pairs:
                    continue

                informal_pairs.add(pair)

    # Domain-based informal groups
    domain_group_count = 0

    for domain, members in by_domain.items():
        groups = make_academic_groups(
            members=members,
            group_rate=DOMAIN_GROUP_RATE,
        )

        domain_group_count += len(groups)

        for group in groups:
            for r1, r2 in combinations(group, 2):
                pair = tuple(sorted((r1, r2)))

                if pair in formal_pairs:
                    continue

                informal_pairs.add(pair)

    print(f"  Department groups created : {dept_group_count:,}")
    print(f"  Domain groups created     : {domain_group_count:,}")
    print(f"  Informal unique pairs     : {len(informal_pairs):,}")

    # Temporal expansion
    print("  Generating temporal informal rows...")

    generated_rows = 0

    for r1, r2 in informal_pairs:
        m1 = researcher_meta[r1]
        m2 = researcher_meta[r2]

        ov_start, ov_end = overlap_interval(
            m1["start"],
            m1["end"],
            m2["start"],
            m2["end"],
        )

        if ov_start is None:
            continue

        if (ov_end - ov_start).days < MIN_OVERLAP_DAYS:
            continue

        sub_periods = []

        pre_end = FIRST_PROJECT_START - pd.Timedelta(days=1)

        if ov_start <= pre_end:
            sub_periods.append((
                ov_start,
                min(ov_end, pre_end),
                PRE_PROJECT_WINDOW_PROB,
            ))

        if ov_end >= FIRST_PROJECT_START:
            sub_periods.append((
                max(ov_start, FIRST_PROJECT_START),
                ov_end,
                INFORMAL_WINDOW_PROB,
            ))

        for sp_start, sp_end, win_prob in sub_periods:
            if (sp_end - sp_start).days < 0:
                continue

            for w in get_fortnight_windows(sp_start, sp_end):
                if rng.random() > win_prob:
                    continue

                fw_start = w["fortnight_start"]
                fw_end = w["fortnight_end"]

                interaction_date = choose_interaction_date(fw_start, fw_end)
                i_type = informal_type(m1, m2)

                event_id = f"EVT{event_counter:07d}"
                event_counter += 1

                add_interaction_row(
                    event_id=event_id,
                    r1=r1,
                    r2=r2,
                    fw_start=fw_start,
                    interaction_date=interaction_date,
                    progress=np.nan,
                    interaction_type=i_type,
                    source_layer="informal",
                    source_id="NA",
                )

                generated_rows += 1

    print(f"  Informal temporal rows generated: {generated_rows:,}")


# ========================================================================================================================
                                                  # 13. RUN GENERATION
# ========================================================================================================================

print("\nBuilding formal pairs set...")

formal_pairs = (
    build_formal_pairs(projects, PR_PI, PR_COI)
    | build_formal_pairs(grants, G_PI, G_COI)
)

print(f"Unique formal pairs from project + grant: {len(formal_pairs):,}")

print("\nGenerating formal project interactions...")

add_formal_events(
    df=projects,
    source_layer="project",
    id_col=PR_ID,
    start_col=PR_START,
    end_col=PR_END,
    pi_col=PR_PI,
    coi_col=PR_COI,
)

print("\nGenerating formal grant interactions...")

add_formal_events(
    df=grants,
    source_layer="grant",
    id_col=G_ID,
    start_col=G_START,
    end_col=G_END,
    pi_col=G_PI,
    coi_col=G_COI,
)

add_informal_events(formal_pairs)

print(f"\nTotal raw rows generated: {len(rows):,}")

if not rows:
    raise ValueError("No rows generated. Check input files and date ranges.")


# ========================================================================================================================
                                              # 14. POST-PROCESSING
# ========================================================================================================================

interactions = pd.DataFrame(rows)

interactions[["r_id1", "r_id2"]] = np.sort(
    interactions[["r_id1", "r_id2"]].values,
    axis=1,
)

interactions = interactions.sort_values(
    by=[
        "interaction_date",
        "event_id",
        "r_id1",
        "r_id2",
    ]
).reset_index(drop=True)

interactions["interaction_id"] = [
    f"INT{i + 1:07d}"
    for i in range(len(interactions))
]

final_columns = [
    "interaction_id",
    "event_id",
    "r_id1",
    "r_id2",
    "r_id1_domain_at_interaction",
    "r_id2_domain_at_interaction",
    "fortnight_start",
    "interaction_date",
    "interaction_year",
    "progress",
    "interaction_type",
    "source_layer",
    "source_id",
]

interactions = interactions[final_columns].copy()

interactions.to_csv(OUTPUT_FILE, index=False)


# ========================================================================================================================
                                                  # 15. SUMMARY
# ========================================================================================================================

print(f"\nSaved: {OUTPUT_FILE}")
print(f"Total rows: {len(interactions):,}")

print("\nSource-layer row counts:")
print(interactions["source_layer"].value_counts(dropna=False))

print("\nInteraction year distribution:")
print(interactions["interaction_year"].value_counts().sort_index())

pre = interactions[
    pd.to_datetime(interactions["interaction_date"]) < FIRST_PROJECT_START
]

post = interactions[
    pd.to_datetime(interactions["interaction_date"]) >= FIRST_PROJECT_START
]

print(f"\nPre-first-project interactions ({FIRST_PROJECT_START.date()}): {len(pre):,}")
print(f"  Layer breakdown: {pre['source_layer'].value_counts().to_dict()}")

print(f"\nPost-first-project interactions: {len(post):,}")
print(f"  Layer breakdown: {post['source_layer'].value_counts().to_dict()}")

all_pairs = set(
    zip(
        interactions["r_id1"].tolist(),
        interactions["r_id2"].tolist(),
    )
)

print(f"\nUnique interaction pairs: {len(all_pairs):,}")

informal_pairs_final = set(
    zip(
        interactions.loc[
            interactions["source_layer"] == "informal",
            "r_id1",
        ].tolist(),
        interactions.loc[
            interactions["source_layer"] == "informal",
            "r_id2",
        ].tolist(),
    )
)

print(f"Informal unique pairs: {len(informal_pairs_final):,}")

print(
    f"\nINT0000001 date: "
    f"{pd.to_datetime(interactions.loc[0, 'interaction_date']).date()}"
)

print(
    f"INT{len(interactions):07d} date: "
    f"{pd.to_datetime(interactions.loc[len(interactions) - 1, 'interaction_date']).date()}"
)

print("\nHead:")
print(interactions.head(10))
