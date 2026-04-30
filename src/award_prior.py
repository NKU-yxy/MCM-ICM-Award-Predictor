"""
Official COMAP MCM/ICM award priors.

The model uses contest-level official statistics rather than optimistic
hand-written problem priors. 2026 results are not filled until COMAP publishes
them; unknown years fall back to the latest official year available here.
"""

from __future__ import annotations

from typing import Dict


AWARD_ORDER = ["O", "F", "M", "H", "S"]

# Raw percentages over all participating teams, from COMAP official result PDFs.
AWARD_STATS_BY_YEAR_CONTEST = {
    2025: {
        "MCM": {
            "teams": 21092,
            "problem_counts": {"A": 2082, "B": 6437, "C": 12573},
            "counts": {"O": 30, "F": 343, "M": 1346, "H": 4921, "S": 13878},
            "other_counts": {"U": 31, "DQ": 514, "NJ": 29},
            "source": "2025_MCM_Problem_C_Results.pdf",
        },
        "ICM": {
            "teams": 6364,
            "problem_counts": {"D": 902, "E": 3323, "F": 2139},
            "counts": {"O": 13, "F": 151, "M": 426, "H": 1444, "S": 4203},
            "other_counts": {"U": 6, "DQ": 113, "NJ": 8},
            "source": "2025_ICM_Problem_F_Results.pdf",
        },
    },
    2024: {
        "MCM": {
            "teams": 18525,
            "problem_counts": {"A": 5698, "B": 2643, "C": 10184},
            "counts": {"O": 20, "F": 319, "M": 1174, "H": 4290, "S": 12122},
            "other_counts": {"U": 49, "DQ": 540, "NJ": 11},
            "source": "2024_MCM_Problem_C_Results.pdf",
        },
        "ICM": {
            "teams": 10388,
            "problem_counts": {"D": 1971, "E": 5436, "F": 2981},
            "counts": {"O": 15, "F": 196, "M": 751, "H": 2380, "S": 6059},
            "other_counts": {"U": 79, "DQ": 895, "NJ": 13},
            "source": "2024_ICM_Problem_F_Results.pdf",
        },
    },
}


PROBLEM_TYPE_PROFILES = {
    "A": {
        "name": "MCM Problem A",
        "description": "Continuous / physical / analytical modeling",
        "key_indicators": {
            "formula_weight": 1.3,
            "figure_weight": 1.0,
            "data_analysis_weight": 0.8,
            "model_complexity_weight": 1.4,
            "sensitivity_weight": 1.3,
        },
        "preferred_keywords": [
            "model",
            "differential equation",
            "optimization",
            "simulation",
            "assumption",
            "sensitivity analysis",
            "validation",
        ],
    },
    "B": {
        "name": "MCM Problem B",
        "description": "Discrete / optimization / operational modeling",
        "key_indicators": {
            "formula_weight": 1.1,
            "figure_weight": 1.1,
            "data_analysis_weight": 0.9,
            "model_complexity_weight": 1.3,
            "sensitivity_weight": 1.2,
        },
        "preferred_keywords": [
            "optimization",
            "algorithm",
            "constraint",
            "simulation",
            "decision",
            "cost",
            "sensitivity analysis",
        ],
    },
    "C": {
        "name": "MCM Problem C",
        "description": "Data insights / statistics / prediction",
        "key_indicators": {
            "formula_weight": 0.9,
            "figure_weight": 1.4,
            "data_analysis_weight": 1.5,
            "model_complexity_weight": 1.0,
            "sensitivity_weight": 1.0,
        },
        "preferred_keywords": [
            "data",
            "regression",
            "prediction",
            "feature",
            "statistical",
            "visualization",
            "validation",
        ],
    },
    "D": {
        "name": "ICM Problem D",
        "description": "Network science / operations research",
        "key_indicators": {
            "formula_weight": 1.2,
            "figure_weight": 1.1,
            "data_analysis_weight": 1.1,
            "model_complexity_weight": 1.3,
            "sensitivity_weight": 1.2,
        },
        "preferred_keywords": [
            "network",
            "operations research",
            "transportation",
            "routing",
            "simulation",
            "policy",
        ],
    },
    "E": {
        "name": "ICM Problem E",
        "description": "Environmental / sustainability modeling",
        "key_indicators": {
            "formula_weight": 1.0,
            "figure_weight": 1.2,
            "data_analysis_weight": 1.3,
            "model_complexity_weight": 1.1,
            "sensitivity_weight": 1.3,
        },
        "preferred_keywords": [
            "environment",
            "ecosystem",
            "sustainability",
            "land use",
            "climate",
            "farmland",
            "forest",
        ],
    },
    "F": {
        "name": "ICM Problem F",
        "description": "Policy / social science modeling",
        "key_indicators": {
            "formula_weight": 0.9,
            "figure_weight": 1.1,
            "data_analysis_weight": 1.2,
            "model_complexity_weight": 1.0,
            "sensitivity_weight": 1.1,
        },
        "preferred_keywords": [
            "policy",
            "stakeholder",
            "governance",
            "national",
            "cybersecurity",
            "regulation",
            "decision",
        ],
    },
}


def contest_for_problem(problem: str) -> str:
    problem = (problem or "A").upper()
    return "MCM" if problem in {"A", "B", "C"} else "ICM"


def _latest_year() -> int:
    return max(AWARD_STATS_BY_YEAR_CONTEST)


def _contest_stats(problem: str, year: int = None) -> dict:
    contest = contest_for_problem(problem)
    year = year if year in AWARD_STATS_BY_YEAR_CONTEST else _latest_year()
    return AWARD_STATS_BY_YEAR_CONTEST[year][contest]


def get_award_prior(problem: str, year: int = None) -> Dict[str, float]:
    stats = _contest_stats(problem, year)
    teams = max(stats["teams"], 1)
    return {award: stats["counts"][award] / teams for award in AWARD_ORDER}


def get_contest_award_prior(contest: str, year: int = None) -> Dict[str, float]:
    contest = (contest or "MCM").upper()
    year = year if year in AWARD_STATS_BY_YEAR_CONTEST else _latest_year()
    stats = AWARD_STATS_BY_YEAR_CONTEST[year][contest]
    teams = max(stats["teams"], 1)
    return {award: stats["counts"][award] / teams for award in AWARD_ORDER}


def get_award_prior_normalized(problem: str, year: int = None) -> Dict[str, float]:
    prior = get_award_prior(problem, year)
    total = sum(prior.values())
    return {k: v / total for k, v in prior.items()}


def get_overall_average_prior() -> Dict[str, float]:
    totals = {award: 0.0 for award in AWARD_ORDER}
    count = 0
    for year_data in AWARD_STATS_BY_YEAR_CONTEST.values():
        for contest in ("MCM", "ICM"):
            teams = year_data[contest]["teams"]
            for award in AWARD_ORDER:
                totals[award] += year_data[contest]["counts"][award] / teams
            count += 1
    return {award: totals[award] / count for award in AWARD_ORDER}


def get_problem_profile(problem: str) -> Dict:
    return PROBLEM_TYPE_PROFILES.get((problem or "A").upper(), PROBLEM_TYPE_PROFILES["A"])


def get_competition_intensity(problem: str) -> float:
    problem = (problem or "A").upper()
    stats = _contest_stats(problem, _latest_year())
    problem_counts = stats.get("problem_counts", {})
    avg = sum(problem_counts.values()) / max(len(problem_counts), 1)
    return min(problem_counts.get(problem, avg) / max(avg, 1), 2.0)


def get_problem_difficulty_adjustment(problem: str) -> Dict[str, float]:
    prior = get_award_prior(problem)
    avg = get_overall_average_prior()
    return {award: prior[award] / max(avg[award], 1e-9) for award in AWARD_ORDER}


def list_available_data() -> str:
    lines = ["MCM/ICM official award priors", "=" * 60]
    for year in sorted(AWARD_STATS_BY_YEAR_CONTEST.keys(), reverse=True):
        lines.append(f"\n{year}")
        for contest in ["MCM", "ICM"]:
            stats = AWARD_STATS_BY_YEAR_CONTEST[year][contest]
            teams = stats["teams"]
            probs = {award: stats["counts"][award] / teams for award in AWARD_ORDER}
            prob_str = " | ".join(f"{k}:{v*100:.2f}%" for k, v in probs.items())
            lines.append(f"  {contest}: {prob_str} | source={stats['source']}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(list_available_data())
