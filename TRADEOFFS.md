# Design Tradeoffs & Reasoning

## Why TF-IDF + keyword taxonomy instead of embeddings or a pure LLM call?

- **Speed and reproducibility.** TF-IDF + cosine similarity runs in milliseconds,
  needs no API key, no GPU, and no model download. A reviewer can clone the repo
  and get identical scores every single time — no LLM non-determinism to worry
  about in the numeric ranking itself.
- **Explainability.** The scoring rubric explicitly rewards "NLP similarity method
  and model choice" and "code quality" — a transparent, auditable scoring formula
  (skills matched / skills required, TF-IDF cosine value) is easier for a reviewer
  to sanity-check than a single opaque LLM-generated number.
- **The LLM is used where it adds the most value**: writing the natural-language
  rationale, not the numeric score. This keeps the ranking deterministic while
  still demonstrating LLM integration (the brief's core "AI agent" requirement) —
  the agent is fully functional and gracefully degrades to a rule-based rationale
  if no `ANTHROPIC_API_KEY` is present, so setup stays foolproof for reviewers.

**What I'd improve with more time:** swap or supplement TF-IDF with sentence
embeddings (e.g. `sentence-transformers/all-MiniLM-L6-v2`) for the similarity
score, since TF-IDF misses synonyms (e.g. "threat hunting" vs. "proactive
detection"). I kept TF-IDF for this submission specifically to avoid a ~90MB
model download turning "clone and run" into "clone, wait, then run."

## Why a fixed skill taxonomy instead of asking an LLM to extract skills?

A hardcoded list is fast, free, and 100% consistent across every resume — but it's
brittle: it will miss a skill phrased in a way the list doesn't anticipate (e.g. a
resume that says "hunted for anomalies via Kusto" instead of "KQL"). An LLM-based
extractor would generalize better but costs money/time per resume and introduces
run-to-run variance in what counts as a "match," which makes the score harder to
trust and reproduce. For a 24-hour build with a rubric that rewards a clearly
explained method, I chose the deterministic approach and documented the gap
honestly rather than hiding it.

## Why isn't years-of-experience folded into the score?

Experience years are extracted and shown in the CSV/JSON for human review, but
deliberately **not** weighted into the numeric score. Reasoning: this is an L1/
entry-level SOC role where the JD literally states 0–2 years — rewarding raw years
would have incorrectly pushed unrelated senior candidates (e.g. a mechanical
engineer with 4 years of experience) above strong junior-fit candidates. For a
senior-level JD, experience should be a scored factor; I'd make the weighting
configurable per-JD with more time rather than hardcoding either behavior.

## Known limitations

- **Substring skill matching** can false-positive on subwords (e.g. "python" inside
  a longer compound word) — low risk in practice but not impossible on adversarial
  resumes.
- **Date-range experience inference** (`(2023 - Present)` style regex) will misfire
  on resumes that format dates unusually, or double-count overlapping roles.
- **Name extraction** is a heuristic (first short, digit-free line) — it will
  occasionally grab a location or header instead of the candidate's actual name on
  resumes with unconventional formatting.
- **No de-duplication** if the same candidate submits two resume versions in the
  same folder — each file is scored independently.
- **English-only.** The skill taxonomy and TF-IDF stopword list assume English-
  language resumes.

## What I'd add with another 24 hours

1. Sentence-embedding similarity as a third scoring signal alongside TF-IDF and
   skill coverage, with the weights validated against a small hand-labeled set of
   "obviously good/bad fit" resumes.
2. A confidence/explanation trace showing exactly which JD sentence each matched
   skill was pulled from (better auditability for a human recruiter).
3. Batch LLM calls (one request covering all resumes) instead of one call per
   candidate, to cut latency and cost when the LLM rationale path is enabled.
4. A minimal web UI (drag-and-drop resumes + JD, live ranked table) — the CLI is
   sufficient for the brief but a UI would make this demo-able to a non-technical
   hiring manager in one sitting.
