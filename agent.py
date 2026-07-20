#!/usr/bin/env python3
"""
Resume Screening Agent
-----------------------
Takes a Job Description (JD) and a folder of resumes (PDF/DOCX/TXT), and produces
a ranked, scored shortlist with human-readable reasoning for every candidate.

Pipeline (Input -> Think -> Act -> Output):
    1. Load JD + parse every resume in the target folder
    2. Extract structured signals: skills, years of experience, education, certifications
    3. Score each resume against the JD using a hybrid method:
         - TF-IDF cosine similarity (semantic/textual closeness to the JD)  -> 60%
         - Skill coverage ratio (required + preferred skills matched)      -> 40%
    4. Optionally ask an LLM (Anthropic Claude) to write a one-line human
       rationale per candidate. Falls back to a rule-based rationale if no
       API key is configured, so the agent is fully runnable out of the box.
    5. Rank candidates and write CSV + JSON output.

Usage:
    python agent.py --jd data/job_description.txt --resumes data/resumes --output output
"""

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional parsers - only imported if the relevant file type is encountered
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import docx  # python-docx
except ImportError:
    docx = None

# Optional: LLM-based rationale generation (Anthropic). Purely additive -
# the agent works fully offline without this.
try:
    import anthropic
except ImportError:
    anthropic = None


# ---------------------------------------------------------------------------
# Skill taxonomy: this is the "knowledge" the agent uses to recognize skills
# in free-text resumes. Grouped so we can separate "required" vs "preferred"
# matches against the JD later. Extend this list for other job families.
# ---------------------------------------------------------------------------
SKILL_TAXONOMY = [
    "splunk", "microsoft sentinel", "sentinel", "qradar", "kql",
    "mitre att&ck", "mitre attack", "siem",
    "tcp/ip", "dns", "vpn", "firewall", "firewalls",
    "windows event logs", "linux syslog", "syslog",
    "python", "powershell", "scripting",
    "phishing", "malware", "brute force", "lateral movement",
    "edr", "crowdstrike", "microsoft defender", "defender",
    "fortigate", "palo alto", "ccna", "ccnp", "ccie",
    "security+", "comptia security+", "iso 27001",
    "incident response", "incident documentation", "servicenow",
    "networking", "linux", "cloud", "aws", "azure",
]

REQUIRED_SKILLS = {
    "splunk", "microsoft sentinel", "sentinel", "qradar", "siem",
    "tcp/ip", "dns", "vpn", "firewall", "firewalls",
    "mitre att&ck", "mitre attack",
    "windows event logs", "linux syslog", "syslog",
    "python", "powershell",
    "phishing", "malware", "brute force", "lateral movement",
    "edr", "crowdstrike", "microsoft defender", "defender",
}

PREFERRED_SKILLS = {
    "security+", "comptia security+", "ccna", "fortigate", "palo alto",
    "kql", "incident response",
}

EDUCATION_KEYWORDS = [
    "b.e.", "b.tech", "bachelor", "m.sc", "m.tech", "mba", "b.sc", "b.com",
]

CERT_KEYWORDS = [
    "certified", "certification", "certificate", "(certified)",
]


@dataclass
class Candidate:
    filename: str
    name: str
    raw_text: str = field(repr=False)
    matched_skills: list = field(default_factory=list)
    missing_required_skills: list = field(default_factory=list)
    years_experience: float = 0.0
    education_found: bool = False
    tfidf_similarity: float = 0.0
    skill_coverage: float = 0.0
    final_score: float = 0.0
    rationale: str = ""


# ---------------------------------------------------------------------------
# Step 1: Parsing - read PDF / DOCX / TXT into plain text
# ---------------------------------------------------------------------------
def parse_resume(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        if pdfplumber is None:
            raise RuntimeError("pdfplumber not installed - cannot parse PDF resumes")
        text = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)
    elif suffix == ".docx":
        if docx is None:
            raise RuntimeError("python-docx not installed - cannot parse DOCX resumes")
        d = docx.Document(path)
        return "\n".join(p.text for p in d.paragraphs)
    else:  # .txt and anything else - read as plain text
        return path.read_text(encoding="utf-8", errors="ignore")


def guess_name(text: str, fallback: str) -> str:
    """Best-effort: assume the first non-empty line is the candidate's name."""
    for line in text.splitlines():
        line = line.strip()
        if line and len(line.split()) <= 5 and not any(ch.isdigit() for ch in line):
            return line
    return fallback


# ---------------------------------------------------------------------------
# Step 2: Extraction - skills, experience, education
# ---------------------------------------------------------------------------
def extract_skills(text_lower: str) -> set:
    found = set()
    for skill in SKILL_TAXONOMY:
        if skill in text_lower:
            found.add(skill)
    return found


def extract_years_experience(text: str) -> float:
    """
    Looks for explicit 'X years' mentions, and falls back to inferring from
    a date range like '(2023 - Present)' or '(2021 - 2024)' if present.
    """
    years_mentions = re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", text, re.IGNORECASE)
    if years_mentions:
        return max(float(y) for y in years_mentions)

    ranges = re.findall(r"\(?(20\d{2})\s*-\s*(Present|present|20\d{2})\)?", text)
    if ranges:
        total = 0.0
        for start, end in ranges:
            end_year = 2026 if end.lower() == "present" else int(end)
            total += max(end_year - int(start), 0)
        return total
    return 0.0


def has_education(text_lower: str) -> bool:
    return any(kw in text_lower for kw in EDUCATION_KEYWORDS)


# ---------------------------------------------------------------------------
# Step 3: Scoring
# ---------------------------------------------------------------------------
def compute_tfidf_similarity(jd_text: str, resume_texts: list) -> list:
    """Cosine similarity between the JD and each resume in shared TF-IDF space."""
    corpus = [jd_text] + resume_texts
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]
    sims = cosine_similarity(jd_vector, resume_vectors)[0]
    return sims.tolist()


def compute_skill_coverage(matched: set) -> float:
    """
    Weighted coverage: required skills count more than preferred skills.
    Required = 70% of the skill-coverage score, Preferred = 30%.
    """
    req_matched = matched & REQUIRED_SKILLS
    pref_matched = matched & PREFERRED_SKILLS

    req_ratio = len(req_matched) / max(len(REQUIRED_SKILLS), 1)
    pref_ratio = len(pref_matched) / max(len(PREFERRED_SKILLS), 1)

    return round((req_ratio * 0.7 + pref_ratio * 0.3) * 100, 2)


def rule_based_rationale(c: "Candidate") -> str:
    matched = ", ".join(sorted(c.matched_skills)) or "none"
    missing = ", ".join(sorted(c.missing_required_skills)[:5]) or "none"
    edu = "has relevant technical education" if c.education_found else "education not clearly technical"
    return (
        f"Matched skills: {matched}. Missing key required skills: {missing}. "
        f"~{c.years_experience:.1f} yrs relevant experience detected; {edu}. "
        f"TF-IDF textual similarity to JD: {c.tfidf_similarity*100:.1f}%."
    )


def llm_rationale(client, jd_text: str, c: "Candidate") -> str:
    """Ask Claude for a concise, recruiter-style rationale. Falls back on any error."""
    try:
        prompt = (
            "You are a technical recruiter. In exactly 2 sentences, explain why this "
            "candidate is or isn't a strong fit for the job description below. Be specific "
            "about matched and missing skills. Do not repeat the job title.\n\n"
            f"JOB DESCRIPTION:\n{jd_text}\n\n"
            f"CANDIDATE RESUME:\n{c.raw_text[:3000]}\n\n"
            f"Matched skills detected: {sorted(c.matched_skills)}\n"
            f"Missing required skills: {sorted(c.missing_required_skills)}\n"
        )
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        return rule_based_rationale(c) + f" [LLM rationale unavailable: {e}]"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run(jd_path: Path, resumes_dir: Path, output_dir: Path, use_llm: bool):
    jd_text = jd_path.read_text(encoding="utf-8", errors="ignore")

    resume_paths = sorted(
        p for p in resumes_dir.iterdir()
        if p.suffix.lower() in (".pdf", ".docx", ".txt") and p.is_file()
    )
    if not resume_paths:
        print(f"No resumes found in {resumes_dir}", file=sys.stderr)
        sys.exit(1)

    candidates = []
    for path in resume_paths:
        text = parse_resume(path)
        text_lower = text.lower()
        matched = extract_skills(text_lower)
        required_missing = REQUIRED_SKILLS - matched

        c = Candidate(
            filename=path.name,
            name=guess_name(text, fallback=path.stem),
            raw_text=text,
            matched_skills=sorted(matched),
            missing_required_skills=sorted(required_missing),
            years_experience=extract_years_experience(text),
            education_found=has_education(text_lower),
        )
        candidates.append(c)

    # TF-IDF similarity across the whole batch at once (shared vocabulary space)
    sims = compute_tfidf_similarity(jd_text, [c.raw_text for c in candidates])
    for c, sim in zip(candidates, sims):
        c.tfidf_similarity = round(float(sim), 4)
        c.skill_coverage = compute_skill_coverage(set(c.matched_skills))
        # Final score: 60% TF-IDF similarity + 40% skill coverage (both 0-100 scale)
        c.final_score = round((c.tfidf_similarity * 100 * 0.6) + (c.skill_coverage * 0.4), 2)

    # Rank descending by final score
    candidates.sort(key=lambda c: c.final_score, reverse=True)

    # Rationale generation
    client = None
    if use_llm and anthropic is not None and os.environ.get("ANTHROPIC_API_KEY"):
        client = anthropic.Anthropic()

    for c in candidates:
        if client:
            c.rationale = llm_rationale(client, jd_text, c)
        else:
            c.rationale = rule_based_rationale(c)

    # Output
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(candidates, output_dir / "ranked_candidates.csv")
    write_json(candidates, output_dir / "ranked_candidates.json")
    print_summary(candidates)


def write_csv(candidates, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "filename", "name", "final_score", "tfidf_similarity_pct",
            "skill_coverage_pct", "years_experience", "education_found",
            "matched_skills", "missing_required_skills", "rationale",
        ])
        for i, c in enumerate(candidates, start=1):
            writer.writerow([
                i, c.filename, c.name, c.final_score,
                round(c.tfidf_similarity * 100, 2), c.skill_coverage,
                c.years_experience, c.education_found,
                "; ".join(c.matched_skills), "; ".join(c.missing_required_skills),
                c.rationale,
            ])


def write_json(candidates, path: Path):
    data = []
    for i, c in enumerate(candidates, start=1):
        d = asdict(c)
        d.pop("raw_text")
        d["rank"] = i
        data.append(d)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def print_summary(candidates):
    print("\n=== RANKED SHORTLIST ===")
    for i, c in enumerate(candidates, start=1):
        print(f"{i:>2}. {c.name:<20} score={c.final_score:>6.2f}  "
              f"(tfidf={c.tfidf_similarity*100:5.1f}%  skills={c.skill_coverage:5.1f}%  "
              f"exp={c.years_experience:.1f}y)")
    print(f"\nWrote {len(candidates)} ranked candidates to output/ranked_candidates.csv"
          f" and output/ranked_candidates.json\n")


def main():
    parser = argparse.ArgumentParser(description="Resume Screening Agent")
    parser.add_argument("--jd", type=Path, default=Path("data/job_description.txt"))
    parser.add_argument("--resumes", type=Path, default=Path("data/resumes"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--no-llm", action="store_true",
                         help="Skip LLM rationale generation even if ANTHROPIC_API_KEY is set")
    args = parser.parse_args()

    run(args.jd, args.resumes, args.output, use_llm=not args.no_llm)


if __name__ == "__main__":
    main()
