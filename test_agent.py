"""
Tests for the Resume Screening Agent.

Run with:
    pip install pytest
    pytest test_agent.py -v
"""

import subprocess
import sys
from pathlib import Path

import pytest

from agent import (
    Candidate,
    compute_skill_coverage,
    compute_tfidf_similarity,
    extract_skills,
    extract_years_experience,
    guess_name,
    has_education,
    rule_based_rationale,
    REQUIRED_SKILLS,
    PREFERRED_SKILLS,
)

REPO_ROOT = Path(__file__).parent


# ---------------------------------------------------------------------------
# Skill extraction
# ---------------------------------------------------------------------------
def test_extract_skills_finds_known_skills():
    text = "Experienced with Splunk, MITRE ATT&CK mapping, and TCP/IP fundamentals."
    found = extract_skills(text.lower())
    assert "splunk" in found
    assert "mitre att&ck" in found
    assert "tcp/ip" in found


def test_extract_skills_ignores_unrelated_text():
    text = "Managed Six Sigma projects and AutoCAD drawings for manufacturing."
    found = extract_skills(text.lower())
    assert "splunk" not in found
    assert "mitre att&ck" not in found
    assert len(found) == 0


def test_extract_skills_is_case_insensitive():
    text = "SPLUNK and Microsoft Sentinel experience."
    found = extract_skills(text.lower())
    assert "splunk" in found
    assert "microsoft sentinel" in found


# ---------------------------------------------------------------------------
# Experience extraction
# ---------------------------------------------------------------------------
def test_extract_years_explicit_mention():
    text = "I have 3 years of experience in network security."
    assert extract_years_experience(text) == 3.0


def test_extract_years_picks_max_mention():
    text = "2 years as an intern, then 5 years as a full-time engineer."
    assert extract_years_experience(text) == 5.0


def test_extract_years_from_date_range():
    text = "SOC Analyst, CyberShield (2023 - Present)"
    years = extract_years_experience(text)
    assert years > 0  # should infer from the date range


def test_extract_years_defaults_to_zero():
    text = "Recent graduate with no prior work experience listed."
    assert extract_years_experience(text) == 0.0


# ---------------------------------------------------------------------------
# Education detection
# ---------------------------------------------------------------------------
def test_has_education_detects_degree():
    assert has_education("b.e. computer science, 2025") is True
    assert has_education("mba marketing") is True


def test_has_education_false_when_absent():
    assert has_education("skilled professional with no listed degree") is False


# ---------------------------------------------------------------------------
# Name guessing
# ---------------------------------------------------------------------------
def test_guess_name_picks_first_short_line():
    text = "Priya Sharma\nBengaluru, India | priya@email.com\n\nSUMMARY\n..."
    assert guess_name(text, fallback="unknown") == "Priya Sharma"


def test_guess_name_falls_back_when_no_valid_line():
    text = "12345\n67890 9999999999"
    name = guess_name(text, fallback="fallback_name")
    assert name == "fallback_name"


# ---------------------------------------------------------------------------
# Skill coverage scoring
# ---------------------------------------------------------------------------
def test_skill_coverage_zero_when_no_match():
    assert compute_skill_coverage(set()) == 0.0


def test_skill_coverage_full_required_beats_full_preferred():
    # Matching ALL required skills should score higher than matching ALL
    # preferred skills, since required is weighted 70% vs preferred's 30%.
    all_required_score = compute_skill_coverage(set(REQUIRED_SKILLS))
    all_preferred_score = compute_skill_coverage(set(PREFERRED_SKILLS))
    assert all_required_score > all_preferred_score


def test_skill_coverage_everything_matched_is_100():
    everything = REQUIRED_SKILLS | PREFERRED_SKILLS
    assert compute_skill_coverage(everything) == 100.0


# ---------------------------------------------------------------------------
# TF-IDF similarity
# ---------------------------------------------------------------------------
def test_tfidf_similarity_range_is_valid():
    jd = "We need a SOC analyst with Splunk and MITRE ATT&CK experience."
    resumes = [
        "SOC analyst skilled in Splunk and MITRE ATT&CK.",
        "Mechanical engineer with Six Sigma and AutoCAD experience.",
    ]
    sims = compute_tfidf_similarity(jd, resumes)
    assert len(sims) == 2
    for s in sims:
        assert 0.0 <= s <= 1.0


def test_tfidf_similarity_ranks_relevant_resume_higher():
    jd = "We need a SOC analyst with Splunk and MITRE ATT&CK experience."
    resumes = [
        "SOC analyst skilled in Splunk, SIEM, and MITRE ATT&CK triage.",
        "Marketing associate managing social media ad campaigns.",
    ]
    sims = compute_tfidf_similarity(jd, resumes)
    assert sims[0] > sims[1]


# ---------------------------------------------------------------------------
# Rationale generation (rule-based fallback)
# ---------------------------------------------------------------------------
def test_rule_based_rationale_mentions_key_fields():
    c = Candidate(
        filename="test.txt",
        name="Test Candidate",
        raw_text="",
        matched_skills=["splunk", "dns"],
        missing_required_skills=["firewall"],
        years_experience=2.0,
        education_found=True,
        tfidf_similarity=0.25,
    )
    rationale = rule_based_rationale(c)
    assert "splunk" in rationale
    assert "firewall" in rationale
    assert "2.0" in rationale


# ---------------------------------------------------------------------------
# End-to-end smoke test: run the actual CLI against the sample data
# ---------------------------------------------------------------------------
def test_end_to_end_cli_produces_output(tmp_path):
    output_dir = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "agent.py"),
            "--jd", str(REPO_ROOT / "data" / "job_description.txt"),
            "--resumes", str(REPO_ROOT / "data" / "resumes"),
            "--output", str(output_dir),
            "--no-llm",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr

    csv_path = output_dir / "ranked_candidates.csv"
    json_path = output_dir / "ranked_candidates.json"
    assert csv_path.exists()
    assert json_path.exists()

    # Sanity check: CSV has a header plus 10 sample resumes
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 11  # 1 header + 10 candidates


def test_end_to_end_ranking_puts_relevant_candidate_first(tmp_path):
    """The strongest SOC-fit sample resume should outrank an unrelated one."""
    output_dir = tmp_path / "output"
    subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "agent.py"),
            "--jd", str(REPO_ROOT / "data" / "job_description.txt"),
            "--resumes", str(REPO_ROOT / "data" / "resumes"),
            "--output", str(output_dir),
            "--no-llm",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    )
    lines = (output_dir / "ranked_candidates.csv").read_text(encoding="utf-8").splitlines()
    top_candidate_row = lines[1]  # first data row after header
    bottom_candidate_row = lines[-1]  # last row = weakest fit

    # Top candidate should score meaningfully higher than the bottom one
    top_score = float(top_candidate_row.split(",")[3])
    bottom_score = float(bottom_candidate_row.split(",")[3])
    assert top_score > bottom_score


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
