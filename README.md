# ClauseCheck

**AI-assisted contract review with evidence retrieval, structured analysis, deterministic classification, and human review.**

ClauseCheck is a prototype contract review application built to explore how generative AI can support structured legal document analysis while maintaining traceability and human oversight.

The application reviews uploaded PDF contracts across 16 common provision categories, retrieves supporting contract language, identifies potentially significant or uncertain terms, and classifies each provision as **FOUND, REVIEW, AMBIGUOUS, or MISSING**.

Rather than allowing the language model to make every decision, ClauseCheck separates evidence retrieval, semantic analysis, and final classification into distinct stages.

> **Disclaimer:** ClauseCheck is a portfolio prototype and does not provide legal advice. AI-generated findings may be incomplete or inaccurate and should be independently reviewed.

---

## Features

- PDF contract upload and text extraction
- Review across 16 common contract provisions
- Evidence retrieval with source locations
- Cross-clause relationship analysis
- Detection of missing referenced documents or dependencies
- Structured AI outputs using Pydantic
- Deterministic Python status classification
- Human-review queue for potentially significant or uncertain provisions
- Contract overview and review summary
- Expandable clause-level findings
- PDF review report export
- CSV structured-results export

---

## Contract Provisions Reviewed

ClauseCheck evaluates:

1. Parties
2. Effective Date
3. Term / Duration
4. Renewal
5. Termination
6. Payment
7. Confidentiality
8. Data Privacy
9. Intellectual Property
10. Indemnification
11. Limitation of Liability
12. Warranties
13. Dispute Resolution
14. Governing Law
15. Assignment
16. Force Majeure

---

## Architecture

ClauseCheck uses a two-stage AI pipeline followed by deterministic Python classification.

```text
              CONTRACT PDF
                    │
                    ▼
             Text Extraction
                    │
                    ▼
        ┌───────────────────────┐
        │ Stage 1               │
        │ Evidence Retrieval    │
        │                       │
        │ Search the contract   │
        │ for relevant passages │
        │ across 16 provisions  │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Stage 2               │
        │ Relationship Analysis │
        │                       │
        │ Compare evidence and  │
        │ generate structured   │
        │ findings + signals    │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Python Classification │
        │                       │
        │ Deterministic rules   │
        │ assign final status   │
        └───────────┬───────────┘
                    │
                    ▼
          FOUND / REVIEW /
        AMBIGUOUS / MISSING
                    │
                    ▼
        Human Review + Exports
```

### Stage 1: Evidence Retrieval

The first AI pass searches the full contract for passages materially relevant to each provision.

It is instructed to retrieve all relevant passages rather than stopping after the first apparent match, including language that may modify or qualify another provision.

This is particularly important for terms introduced through language such as:

- notwithstanding
- subject to
- except
- unless
- provided that
- as set forth in
- pursuant to
- defined in
- governed by

Stage 1 does **not** determine the final status of a provision.

### Stage 2: Relationship and Signal Analysis

The second AI pass receives the evidence retrieved during Stage 1.

It evaluates relationships among the passages and produces a finding along with structured signals such as:

- provision present
- conflicting language
- unclear meaning
- missing dependency
- human review flag

This allows the model to perform semantic reasoning while keeping the final classification decision outside the model.

### Deterministic Classification

Final statuses are assigned using Python rules:

```python
if not provision_present:
    status = "MISSING"

elif conflicting_language or unclear_meaning or missing_dependency:
    status = "AMBIGUOUS"

elif review_flag:
    status = "REVIEW"

else:
    status = "FOUND"
```

This architecture was introduced after testing showed that model-generated status labels could vary even when the underlying evidence and analysis were substantially correct.

---

## Status Definitions

| Status | Meaning |
|---|---|
| **FOUND** | The provision is present and no material uncertainty requiring escalation was identified. |
| **REVIEW** | The provision is present, but its substance or risk characteristics warrant human review. |
| **AMBIGUOUS** | The provision is present, but material uncertainty, unresolved relationships, or missing dependencies prevent confident interpretation. |
| **MISSING** | No meaningful language addressing the provision was identified. |

These statuses are intended to organize review, not determine legal conclusions.

---

## Evaluation and Iteration

ClauseCheck was developed through a series of synthetic contract tests designed to expose different failure modes rather than simply demonstrate successful outputs.

### Test 01: Baseline Contract

A relatively complete agreement was used to validate basic extraction and evidence grounding.

**Result:** All 16 provision categories were addressed with no observed evidence hallucinations or false missing/found classifications.

### Test 02: Classification Consistency

A deliberately messy consulting agreement included competing dates, deferred payment terms, asymmetric provisions, missing exhibits, and intentionally omitted clauses.

Testing showed that the model could often identify the correct evidence while still producing inconsistent final status labels.

**Design change:** Final status assignment was removed from the model and moved into deterministic Python logic.

### Test 03: Cross-Clause Retrieval

A Mutual NDA contained liability-related language across separate sections.

The single-pass architecture initially identified the general liability provision but failed to retrieve a later provision that materially changed its application.

**Design change:** ClauseCheck was redesigned as a two-stage pipeline:

1. retrieve relevant evidence across the entire contract
2. analyze the complete retrieved evidence set

The final system successfully retrieved the related passages and recognized that explicit override language established how the provisions operated together.

This test also prompted reevaluation of the expected answer itself. Rather than tuning the model to reproduce the original label, the classification target was revised when the evidence showed that the controlling relationship was explicit.

### Test 04: Cross-Clause Stress Test

The final synthetic agreement tested:

- automatic renewal
- external order forms
- missing schedules
- liability caps
- cross-clause overrides
- arbitration carve-outs
- assignment modifications
- precedence language
- missing definitions

An intermediate v2 run matched **13 of 16 intended classifications (81.25%)**.

Error analysis showed that evidence retrieval was working, but explicit overrides were sometimes being treated as unresolved conflicts and a material missing dependency was not being recognized.

After refining the general signal definitions, the final run matched:

**16 of 16 intended classifications.**

The architecture was then frozen rather than continuing to tune against the small test set.

---

## Why the Architecture Changed

ClauseCheck's development surfaced two different AI failure modes.

### 1. Classification inconsistency

The model could identify relevant language but inconsistently map that analysis to a fixed status taxonomy.

**Response:** Move the rule-governed classification step into deterministic Python.

### 2. Retrieval incompleteness

A model cannot reason about contract language it did not retrieve or consider.

**Response:** Separate full-contract evidence retrieval from downstream relationship analysis.

The resulting system deliberately uses AI where semantic reasoning is valuable and conventional code where predictable rules are preferable.

---

## Human-in-the-Loop Design

ClauseCheck is designed as a review aid rather than an autonomous legal decision-maker.

The interface specifically surfaces:

- provisions requiring review
- ambiguous provisions
- missing provisions
- supporting evidence
- evidence locations
- limitations
- referenced material that may be missing

This makes uncertainty visible rather than attempting to hide it behind a single model-generated conclusion.

---

## Technology

- **Python**
- **Streamlit**
- **OpenAI API**
- **Pydantic**
- **PyPDF**
- **Pandas**
- **ReportLab**

---

## Running Locally

### 1. Clone the repository

```bash
git clone <repository-url>
cd ClauseCheck
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file in the project directory:

```text
OPENAI_API_KEY=your_api_key_here
```

The `.env` file is excluded from version control.

### 5. Start ClauseCheck

```bash
streamlit run app.py
```

---

## Output

ClauseCheck provides both interactive and downloadable results.

### Interactive Review

The Streamlit interface displays:

- review summary
- contract overview
- items requiring human review
- missing provisions
- limitations
- clause-level evidence and findings

### PDF Report

Generates a human-readable contract review containing the overview, summary, review items, missing provisions, limitations, and complete clause-level analysis.

### CSV Results

Exports structured clause-level results including statuses, findings, evidence, locations, review notes, and underlying classification signals.

---

## Limitations

ClauseCheck is an experimental portfolio project.

Current limitations include:

- AI outputs can be incomplete or inaccurate.
- PDF extraction quality depends on the source document.
- The evaluation set is small and synthetic.
- Successful performance on the test contracts does not establish accuracy across real-world legal agreements.
- Provision significance depends on contract context and organizational requirements.
- The system does not replace attorney review or legal judgment.
- Referenced documents that are not provided may prevent complete interpretation.

Future evaluation would require a larger and more diverse contract set, expert-reviewed ground truth, systematic error analysis, and repeated-run testing.

---

## Development Approach

This project emphasizes **evaluation-driven AI development** rather than prompt engineering alone.

The development process included:

1. defining a structured review taxonomy
2. creating synthetic test contracts with known edge cases
3. comparing expected and observed behavior
4. identifying whether failures originated in retrieval, reasoning, or classification
5. modifying the architecture based on those failure modes
6. retesting against previously observed edge cases
7. documenting limitations and preserving human review
8. stopping further tuning when additional changes risked overfitting the evaluation set

The goal is not to demonstrate that an LLM can review contracts perfectly. It is to demonstrate how an AI-supported workflow can be tested, constrained, made more auditable, and integrated with deterministic software components.