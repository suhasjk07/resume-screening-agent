# Resume Screening Agent

An AI agent that takes a **Job Description** and a folder of **resumes** (PDF, DOCX,
or TXT) and produces a **ranked, scored shortlist with reasoning** for every candidate.

Built for the Rooman AI Challenge (Category 1 — HR & Recruitment).

---

## What it does

```
Input: Job Description + folder of resumes
  ↓
Parse each resume (PDF/DOCX/TXT → plain text)
  ↓
Extract skills, years of experience, education signals
  ↓
Score each resume against the JD:
   60% TF-IDF cosine similarity (textual/semantic closeness to JD)
   40% weighted skill coverage (required skills weigh more than preferred)
  ↓
Rank candidates highest → lowest score
  ↓
Output: ranked_candidates.csv + ranked_candidates.json (+ console summary)
```

## Setup

```bash
git clone <your-repo-url>
cd resume-screening-agent
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

No API key is required to run the agent. See **Optional: LLM rationale** below if
you want richer, LLM-written explanations instead of the built-in rule-based ones.

## Run it

```bash
python agent.py --jd data/job_description.txt --resumes data/resumes --output output
```

This runs against the included sample JD (SOC L1 Analyst) and 10 sample resumes,
and writes results to `output/ranked_candidates.csv` and `output/ranked_candidates.json`.

To use your own data:

```bash
python agent.py --jd path/to/your_jd.txt --resumes path/to/your_resumes_folder --output output
```

Resumes can be `.pdf`, `.docx`, or `.txt` — you can mix formats in the same folder.

### Optional: LLM rationale

By default the agent writes a rule-based rationale ("Matched skills: X, Y. Missing:
Z. ~N years experience..."). If you'd rather have Claude write a natural-language
recruiter-style rationale for each candidate:

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
python agent.py --jd data/job_description.txt --resumes data/resumes --output output
```

Pass `--no-llm` to force the rule-based rationale even if a key is set.

## Sample output

Running the included sample data produces this ranking (`output/ranked_candidates.csv`):

| Rank | Candidate | Score | TF-IDF | Skill Coverage | Experience |
|---|---|---|---|---|---|
| 1 | Mohammed Ali | 34.63 | 21.1% | 54.9% | 1.5y |
| 2 | Priya Sharma | 25.31 | 17.8% | 36.6% | 0.0y |
| 3 | Neha Gupta | 22.76 | 12.6% | 38.1% | 0.0y |
| 4 | Sara Khan | 20.18 | 10.1% | 35.3% | 1.0y |
| 5 | Ananya Das | 17.58 | 10.5% | 28.2% | 0.0y |
| 6 | Divya Reddy | 4.04 | 4.9% | 2.8% | 0.0y |
| 7 | Karthik Iyer | 2.67 | 2.6% | 2.8% | 3.0y |
| 8 | Arjun Mehta | 0.37 | 0.6% | 0.0% | 2.0y |
| 9 | Vikram Singh | 0.35 | 0.6% | 0.0% | 4.0y |
| 10 | Rahul Verma | 0.25 | 0.4% | 0.0% | 2.0y |

This matches intuition: candidates with real SOC/networking/security backgrounds
(Mohammed, Priya, Neha, Sara, Ananya) rank at the top; candidates from unrelated
fields (marketing, mechanical engineering, generic web dev, data analytics) correctly
sink to the bottom despite having more total years of work experience — because
raw experience alone shouldn't outrank relevant skills for this JD.

Full per-candidate detail (matched skills, missing required skills, rationale) is in
`output/ranked_candidates.json`.

## Project structure

```
resume-screening-agent/
├── agent.py                    # Main agent (parsing, scoring, ranking, output)
├── requirements.txt
├── .env.example
├── data/
│   ├── job_description.txt     # Sample JD: SOC L1 Analyst
│   └── resumes/                # 10 sample resumes (varied fit levels)
├── output/
│   ├── ranked_candidates.csv
│   └── ranked_candidates.json
├── README.md
└── TRADEOFFS.md
```

## How scoring works (to avoid a "black box")

1. **Skill extraction**: a fixed taxonomy of ~35 cybersecurity/SOC-relevant terms
   (SIEM tools, networking terms, EDR platforms, scripting languages, attack
   vectors) is matched against the lowercased resume text via substring search.
   Matched skills are split into **required** (from the JD's "Required Skills"
   section) and **preferred** (from "Preferred Qualifications").
2. **Skill coverage score** = `0.7 × (required matched / total required) + 0.3 ×
   (preferred matched / total preferred)`, scaled to 0–100.
3. **TF-IDF similarity**: the JD and every resume are vectorized together (shared
   vocabulary, unigrams + bigrams, English stopwords removed) and scored via cosine
   similarity — this captures phrasing/context overlap that a fixed keyword list
   would miss.
4. **Final score** = `0.6 × TF-IDF similarity + 0.4 × skill coverage` (both on a
   0–100 scale). Skill coverage is weighted heavily because for a technical SOC
   role, having the right named tools/skills is a stronger signal than generic
   textual overlap with the JD.
5. **Years of experience** and **education** are extracted and reported alongside
   the score for human review, but are **not** currently part of the numeric score
   (see Tradeoffs).

See `TRADEOFFS.md` for design decisions, limitations, and what I'd improve with
more time.
