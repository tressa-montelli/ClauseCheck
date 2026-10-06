import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI
import os
from pydantic import BaseModel
from typing import List
import io
import csv
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, 
    Paragraph, 
    Spacer, 
    Table, 
    TableStyle, 
    PageBreak
)
from reportlab.lib import colors

# Load environment variables from .env
load_dotenv()

# Create OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EVIDENCE_EXTRACTION_PROMPT = """
You are the evidence-extraction stage of ClauseCheck, an AI-assisted contract review system.

Your only task is to identify contract passages that may materially affect each of the 16 provision categories listed below.

Do not classify provisions.
Do not determine whether language is ambiguous, conflicting, unusual, risky, acceptable, or legally enforceable.
Do not provide legal advice.
Do not resolve relationships between passages.
Do not exclude a passage merely because another passage appears more specific or controlling.

PROVISIONS

1. Parties
2. Effective Date
3. Term / Duration
4. Renewal / Auto-Renewal
5. Termination
6. Payment Terms
7. Confidentiality
8. Data Privacy / Data Protection
9. Intellectual Property
10. Indemnification
11. Limitation of Liability
12. Warranties
13. Dispute Resolution
14. Governing Law
15. Assignment
16. Force Majeure

EVIDENCE EXTRACTION RULES

For every provision, search the entire supplied contract for all passages that could materially affect that provision.

A relevant passage may:

- directly establish the provision
- modify or qualify another provision
- create an exception or carve-out
- override another provision
- refer to another section, exhibit, schedule, order form, statement of work, or external document
- define a term that materially affects the provision
- establish a different right, obligation, date, value, limitation, or condition related to the provision

A passage may be relevant to more than one provision. Include it under every provision it materially affects.

Pay particular attention to language such as:

- notwithstanding
- subject to
- except
- unless
- provided that
- in addition to
- as set forth in
- pursuant to
- defined in
- governed by

Do not stop after finding the first relevant passage. Search the full contract for additional relevant language.

For each relevant passage return:

evidence:
Exact contract language. Keep the quotation as short as practical while preserving the relevant meaning.

location:
The section number, heading, page, or other identifiable location.

relevance:
A short factual explanation of why the passage may affect the provision. Do not decide the ultimate legal effect or classification.

If no relevant passage can be identified for a provision, return an empty passages list.

Never invent, reconstruct, or supplement contract language.

Base the extraction only on the supplied document.
"""

CLAUSECHECK_PROMPT = """
You are ClauseCheck, an AI-assisted contract review tool designed to help legal and business professionals identify and organize important contract provisions for human review.

Your role is to extract and summarize information contained in a contract and identify structured signals that application logic will use for classification. You are NOT a lawyer, you do not provide legal advice, and you must not make definitive legal conclusions.

CORE TASK

Review the contract provided by the user and evaluate it for the following provisions:

1. Parties
2. Effective Date
3. Term / Duration
4. Renewal / Auto-Renewal
5. Termination
6. Payment Terms
7. Confidentiality
8. Data Privacy / Data Protection
9. Intellectual Property
10. Indemnification
11. Limitation of Liability
12. Warranties
13. Dispute Resolution
14. Governing Law
15. Assignment
16. Force Majeure

For each provision, return:

- Clause: The provision being evaluated.
- Finding: A concise plain-language description of what the contract states.
- Evidence: The exact contract language supporting the finding. Keep quotations as short as practical while preserving the relevant meaning.
- Location: The section number, heading, page, or other identifiable location when available.
- Review Note: Briefly explain any ambiguity, conflict, missing dependency, or characteristic requiring human attention. Leave this blank when no meaningful review issue is identified.
- Extraction Signals: Return the required structured signals described below.

EVIDENCE REQUIREMENTS

Every substantive finding must be supported by evidence from the supplied contract.

Never invent, infer, or reconstruct contract language.

Do not treat absence of evidence as evidence that a particular legal conclusion is true.

Do not rely on outside knowledge to fill gaps in the contract.

If multiple sections affect the same provision, identify the relevant sections and explain the relationship between them.

If no relevant language addressing a provision can be identified, do not invent evidence or a location.

EXTRACTION SIGNALS

For each provision, evaluate and return the following structured signals. These signals will be used by application logic to determine the final status.

provision_present:
Set to true if any contract language meaningfully addresses the provision, even if the language is incomplete, ambiguous, unusual, or contained across multiple sections.
Set to false only when no relevant language addressing the provision can be identified.

conflicting_language:

Set to true only when multiple relevant passages address the same provision and they cannot be confidently reconciled from the supplied evidence, or when their relationship or controlling effect cannot be determined.
Do not set conflicting_language to true merely because multiple passages contain different rules, rights, obligations, dates, values, or procedures.
When the supplied evidence clearly establishes the relationship between passages through an express exception, carve-out, qualification, modification, priority rule, or override, conflicting_language should be false if the controlling effect can be confidently determined.
Language such as "notwithstanding," "subject to," "except," or similar language may establish which provision controls or how the provisions operate together. Evaluate the actual relationship rather than treating the existence of different terms as a conflict.
Set conflicting_language to true only when the passages remain materially inconsistent or their controlling relationship remains unresolved after considering all relevant evidence.
Otherwise set to false.

unclear_meaning:
Set to true when relevant language exists but a material part of the provision cannot be confidently determined from the supplied contract. This includes vague language, unspecified material terms, multiple plausible interpretations, or terms that will be determined later.
Otherwise set to false.

missing_dependency:

Set to true when understanding a material part of the provision depends on a referenced document, exhibit, schedule, statement of work, definition, order form, or other source that is not included in the supplied evidence.
Evaluate all passages retrieved for the provision for missing dependencies, including passages that modify, qualify, override, or create exceptions to another passage.
If a material term, definition, obligation, condition, amount, procedure, or scope cannot be fully determined without the missing referenced source, set missing_dependency to true even when other parts of the provision are clear.
Do not set missing_dependency to true merely because another document is mentioned. The missing source must materially affect understanding of the provision.
Otherwise set to false.

review_flag:
Set to true when the provision is sufficiently identifiable but contains a characteristic listed under REVIEW FLAGS that warrants human attention.
Otherwise set to false.

IMPORTANT SIGNAL RULES

A provision may have more than one signal set to true.

Do not set provision_present to false merely because a provision is incomplete or ambiguous. If relevant language exists, provision_present should be true.

For Term / Duration, evaluate whether the contract clearly establishes the current or initial contractual term. If the initial term can be confidently determined from the supplied contract, do not set unclear_meaning to true solely because renewal, extension, or post-term continuation mechanics are incomplete or unclear. Those post-term issues should be evaluated under Renewal / Auto-Renewal.

For example, if the contract clearly establishes an initial twelve-month term but states only that the parties may continue afterward by mutual agreement, Term / Duration may be sufficiently clear while Renewal / Auto-Renewal may have unclear_meaning = true.

For renewal or continuation language, language stating that an agreement may continue, extend, or renew is relevant language even when the renewal mechanics are incomplete. In that situation, provision_present should be true and unclear_meaning should be true when the controlling renewal terms cannot be determined.

For Effective Date, do not assume that a date stating when an agreement was "entered into," "signed," or "executed" is necessarily the controlling Effective Date.

If the contract contains one date for entry, signing, or execution and another date for when services, obligations, performance, or the contract term begin, and the document does not explicitly define which date is the Effective Date, then:
- provision_present = true
- unclear_meaning = true

This applies even if one of the dates appears in the opening sentence of the agreement.

For termination, if multiple termination rights or notice periods apply and their relationship is unclear, provision_present should be true and conflicting_language should be true.

For payment terms, if a material payment term is deferred to future agreement or contained in a referenced document that was not supplied, provision_present should be true and set unclear_meaning and/or missing_dependency to true as applicable.

For confidentiality, if the confidentiality obligations apply only to one party and there is no corresponding confidentiality obligation for the other party, set review_flag = true.

REVIEW FLAGS

Set review_flag to true when a sufficiently identifiable provision contains characteristics that may deserve closer human examination, including but not limited to:

- automatic renewal
- unilateral termination rights
- unusually broad obligations
- broad indemnification language
- liability limitations or exclusions
- data handling or privacy obligations
- intellectual property ownership or licensing provisions
- materially asymmetric rights or obligations between the parties
- one-way confidentiality obligations where only one party is expressly required to protect the other party's confidential information

Do not use review_flag merely because a provision is ambiguous, conflicting, incomplete, or dependent on a missing document. Those conditions should be represented by unclear_meaning, conflicting_language, or missing_dependency as applicable.

A provision may nevertheless have review_flag = true in addition to another signal when it independently contains a REVIEW FLAGS characteristic.

Do not state that a provision is legally acceptable, enforceable, standard, invalid, or risky. Instead, identify the relevant language and explain why a human reviewer may want to examine it.

OUTPUT FORMAT

CONTRACT OVERVIEW

Provide a short factual overview containing:
- Document type, if identifiable
- Parties
- Effective date, if identifiable from the supplied contract
- Contract term, if identifiable from the supplied contract

When an Effective Date cannot be confidently determined because multiple plausible dates exist, the overview may identify the competing dates rather than selecting one as controlling.

CLAUSE REVIEW

Evaluate all 16 provisions.

For each provision provide:

Clause:
The provision being evaluated.

Finding:
A concise description of what the supplied contract says about the provision.

Evidence:
Exact supporting contract language. Keep quotations as short as practical while preserving the relevant meaning. If no relevant language exists, leave Evidence blank.

Location:
The section, heading, and/or page where the supporting language appears. If no relevant language exists, leave Location blank.

Review Note:
For provisions with ambiguity, conflicting language, missing dependencies, or a review flag, briefly explain what requires human attention. Otherwise leave blank.

Extraction Signals:
Return the required structured signal fields for each provision:
- provision_present
- conflicting_language
- unclear_meaning
- missing_dependency
- review_flag

Do not determine the final FOUND, MISSING, AMBIGUOUS, or REVIEW status. Final status classification is performed by application logic after extraction.

LIMITATIONS

Briefly identify anything that limited the analysis, such as unreadable text, missing pages, referenced exhibits that were not supplied, unclear section numbering, or information that could not be confidently determined.

If there are no identifiable limitations, state that no document-specific limitations were identified.

IMPORTANT BEHAVIOR

Accuracy and traceability are more important than producing a complete-looking answer.

Never fabricate information to fill an empty field.

When uncertain, explicitly communicate uncertainty.

Base the analysis only on the document supplied by the user.

The output is intended to assist human contract review and must not be presented as a substitute for professional legal judgment.
"""

class EvidencePassage(BaseModel):
    evidence: str
    location: str
    relevance: str


class ProvisionEvidence(BaseModel):
    clause: str
    passages: List[EvidencePassage]


class EvidenceExtractionResult(BaseModel):
    provisions: List[ProvisionEvidence]

class ClauseResult(BaseModel):
    clause: str
    finding: str
    evidence: str
    location: str
    review_note: str

    # Signals used by Python to determine final status
    provision_present: bool
    conflicting_language: bool
    unclear_meaning: bool
    missing_dependency: bool
    review_flag: bool
    
    @property
    def status(self):
        return determine_status(self)


class ContractOverview(BaseModel):
    document_type: str
    parties: str
    effective_date: str
    contract_term: str


class ClauseCheckResult(BaseModel):
    overview: ContractOverview
    clauses: List[ClauseResult]
    limitations: List[str]

    @property
    def human_review_items(self):
        return [
            f"{clause.clause}: {clause.review_note or clause.finding}"
            for clause in self.clauses
            if clause.status in ["REVIEW", "AMBIGUOUS"]
        ]

    @property
    def missing_provisions(self):
        return [
            clause.clause
            for clause in self.clauses
            if clause.status == "MISSING"
        ]
def determine_status(clause):
    if not clause.provision_present:
        return "MISSING"

    if (
        clause.conflicting_language
        or clause.unclear_meaning
        or clause.missing_dependency
    ):
        return "AMBIGUOUS"

    if clause.review_flag:
        return "REVIEW"

    return "FOUND"

def build_report(result, filename):
    report = []

    report.append("# ClauseCheck Review Report")
    report.append("")
    report.append(f"**Source Document:** {filename}")
    report.append("")
    report.append(
        "**Important:** ClauseCheck is an AI-assisted review tool and does not "
        "provide legal advice. Findings should be reviewed by a qualified human reviewer."
    )
    report.append("")

    # Contract overview
    report.append("## Contract Overview")
    report.append("")
    report.append(f"**Document Type:** {result.overview.document_type}")
    report.append(f"**Parties:** {result.overview.parties}")
    report.append(f"**Effective Date:** {result.overview.effective_date}")
    report.append(f"**Contract Term:** {result.overview.contract_term}")
    report.append("")

    # Review summary
    found_count = sum(
        1 for clause in result.clauses
        if clause.status == "FOUND"
    )
    review_count = sum(
        1 for clause in result.clauses
        if clause.status == "REVIEW"
    )
    ambiguous_count = sum(
        1 for clause in result.clauses
        if clause.status == "AMBIGUOUS"
    )
    missing_count = sum(
        1 for clause in result.clauses
        if clause.status == "MISSING"
    )

    report.append("## Review Summary")
    report.append("")
    report.append(f"- Found: {found_count}")
    report.append(f"- Review: {review_count}")
    report.append(f"- Ambiguous: {ambiguous_count}")
    report.append(f"- Missing: {missing_count}")
    report.append("")

    # Items for human review
    if result.human_review_items:
        report.append("## Items for Human Review")
        report.append("")

        for item in result.human_review_items:
            report.append(f"- {item}")

        report.append("")

    # Missing provisions
    if result.missing_provisions:
        report.append("## Missing Provisions")
        report.append("")

        for item in result.missing_provisions:
            report.append(f"- {item}")

        report.append("")

    # Limitations
    if result.limitations:
        report.append("## Limitations")
        report.append("")

        for item in result.limitations:
            report.append(f"- {item}")

        report.append("")

    # Detailed clause review
    report.append("## Clause Review")
    report.append("")

    for clause in result.clauses:
        report.append(f"### {clause.clause} | {clause.status}")
        report.append("")
        report.append(f"**Finding:** {clause.finding}")
        report.append("")
        report.append(f"**Evidence:** {clause.evidence}")
        report.append("")
        report.append(f"**Location:** {clause.location}")
        report.append("")

        if clause.review_note:
            report.append(f"**Review Note:** {clause.review_note}")
            report.append("")

    return "\n".join(report)
def build_pdf_report(result):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ClauseCheckTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        spaceAfter=20,
    )

    heading_style = styles["Heading2"]
    subheading_style = styles["Heading3"]
    body_style = styles["BodyText"]

    story = []

    # Title
    story.append(Paragraph("ClauseCheck Review", title_style))

    # Review Summary
    story.append(Paragraph("Review Summary", heading_style))

    status_counts = {
        "FOUND": 0,
        "REVIEW": 0,
        "AMBIGUOUS": 0,
        "MISSING": 0,
    }

    for clause in result.clauses:
        status_counts[clause.status] += 1

    summary_data = [
        ["Found", "Review", "Ambiguous", "Missing"],
        [
            status_counts["FOUND"],
            status_counts["REVIEW"],
            status_counts["AMBIGUOUS"],
            status_counts["MISSING"],
        ],
    ]

    summary_table = Table(
        summary_data,
        colWidths=[110, 110, 110, 110]
    )

    summary_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )

    story.append(summary_table)
    story.append(Spacer(1, 18))

    # Contract Overview
    story.append(Paragraph("Contract Overview", heading_style))

    story.append(
        Paragraph(
            f"<b>Document Type:</b> {result.overview.document_type}",
            body_style,
        )
    )

    story.append(
        Paragraph(
            f"<b>Parties:</b> {result.overview.parties}",
            body_style,
        )
    )

    story.append(
        Paragraph(
            f"<b>Effective Date:</b> {result.overview.effective_date}",
            body_style,
        )
    )

    story.append(
        Paragraph(
            f"<b>Contract Term:</b> {result.overview.contract_term}",
            body_style,
        )
    )

    story.append(Spacer(1, 18))

    # Items for Human Review
    story.append(Paragraph("Items for Human Review", heading_style))

    if result.human_review_items:
        for item in result.human_review_items:
            story.append(Paragraph(f"• {item}", body_style))
            story.append(Spacer(1, 5))
    else:
        story.append(Paragraph("None identified.", body_style))

    story.append(Spacer(1, 18))

    # Missing Provisions
    story.append(Paragraph("Missing Provisions", heading_style))

    if result.missing_provisions:
        for item in result.missing_provisions:
            story.append(Paragraph(f"• {item}", body_style))
    else:
        story.append(Paragraph("None identified.", body_style))

    story.append(Spacer(1, 18))

    # Limitations
    story.append(Paragraph("Limitations", heading_style))

    if result.limitations:
        for limitation in result.limitations:
            story.append(Paragraph(f"• {limitation}", body_style))
            story.append(Spacer(1, 5))
    else:
        story.append(Paragraph("None identified.", body_style))

    story.append(PageBreak())

    # Full Clause Review
    story.append(Paragraph("Clause Review", heading_style))

    for clause in result.clauses:
        story.append(
            Paragraph(
                f"{clause.clause} — {clause.status}",
                subheading_style,
            )
        )

        story.append(
            Paragraph(
                f"<b>Finding:</b> {clause.finding}",
                body_style,
            )
        )

        if clause.evidence:
            story.append(
                Paragraph(
                    f"<b>Evidence:</b> {clause.evidence}",
                    body_style,
                )
            )

        if clause.location:
            story.append(
                Paragraph(
                    f"<b>Location:</b> {clause.location}",
                    body_style,
                )
            )

        if clause.review_note:
            story.append(
                Paragraph(
                    f"<b>Review Note:</b> {clause.review_note}",
                    body_style,
                )
            )

        story.append(Spacer(1, 14))

    doc.build(story)

    buffer.seek(0)
    return buffer.getvalue()
def build_csv_report(result):
    output = io.StringIO()

    fieldnames = [
        "Clause",
        "Status",
        "Finding",
        "Evidence",
        "Location",
        "Review Note",
        "Provision Present",
        "Conflicting Language",
        "Unclear Meaning",
        "Missing Dependency",
        "Review Flag",
    ]

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for clause in result.clauses:
        writer.writerow(
            {
                "Clause": clause.clause,
                "Status": clause.status,
                "Finding": clause.finding,
                "Evidence": clause.evidence,
                "Location": clause.location,
                "Review Note": clause.review_note,
                "Provision Present": clause.provision_present,
                "Conflicting Language": clause.conflicting_language,
                "Unclear Meaning": clause.unclear_meaning,
                "Missing Dependency": clause.missing_dependency,
                "Review Flag": clause.review_flag,
            }
        )

    return output.getvalue()
# Page setup
st.set_page_config(
    page_title="ClauseCheck",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ClauseCheck visual theme
st.markdown(
    """
    <style>
    :root {
        --cc-maroon-dark: #5B1F2A;
        --cc-maroon: #7A2E3A;
        --cc-maroon-light: #B56F78;
        --cc-blush: #E5C5C1;
        --cc-background: #F4E7E2;
        --cc-cream: #FFF8E9;
        --cc-brown: #49302C;
        --cc-muted-brown: #755B56;
        --cc-border: #D5AAA5;
    }

    /* Main page background */
    .stApp {
        background-color: var(--cc-background);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
    }

    /* Main ClauseCheck banner */
    .cc-hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(
            120deg,
            #7A2636 0%,
            #681E2C 55%,
            #541823 100%
        );
        border-radius: 12px;
        padding: 38px 42px;
        margin-bottom: 30px;
        box-shadow: 0 7px 20px rgba(73, 48, 44, 0.15);
    }

    .cc-hero-content {
        display: flex;
        align-items: center;
        position: relative;
        z-index: 2;
    }

    /* Document icon */
    .cc-document-icon {
        width: 72px;
        height: 72px;
        border: 1px solid rgba(255, 246, 229, 0.40);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--cc-cream);
        font-family: Georgia, "Times New Roman", serif;
        font-size: 30px;
        background-color: rgba(255, 246, 229, 0.05);
        margin-right: 28px;
    }

    .cc-vertical-line {
        height: 76px;
        width: 1px;
        background-color: rgba(255, 246, 229, 0.42);
        margin-right: 28px;
    }

    /* Main title */
    .cc-title {
        color: var(--cc-cream);
        font-family: Georgia, "Times New Roman", serif;
        font-size: 3.15rem;
        font-weight: 600;
        letter-spacing: -0.035em;
        line-height: 1;
        margin: 0;
    }

    .cc-kicker {
        color: #D9B8B3;
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.22em;
        margin-top: 12px;
    }

    /* Faint decorative document */
    .cc-hero::after {
        content: "";
        position: absolute;
        width: 220px;
        height: 270px;
        right: 55px;
        top: 35px;
        border: 2px solid rgba(255, 246, 229, 0.07);
        border-radius: 5px;
        transform: rotate(14deg);
    }

    /* Description area */
    .cc-intro {
        margin: 0 4px 27px 4px;
        max-width: 900px;
    }

    .cc-intro-title {
        color: var(--cc-maroon-dark);
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.9rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .cc-intro-text {
        color: var(--cc-brown);
        font-size: 0.98rem;
        line-height: 1.7;
        max-width: 850px;
    }

    /* General headings */
    h1, h2, h3, h4 {
        color: var(--cc-maroon-dark) !important;
    }

    /* Normal text */
    p, label {
        color: var(--cc-brown);
    }

    /* Disclaimer / alert boxes */
div[data-testid="stAlert"] {
    background-color: #F8E8DF;
    border: 1px solid #C98D8A;
    border-left: 4px solid var(--cc-maroon);
    border-radius: 8px;
}

div[data-testid="stAlert"] p {
    color: var(--cc-maroon-dark) !important;
    font-weight: 500;
}

div[data-testid="stAlert"] svg {
    fill: var(--cc-maroon-dark) !important;
    color: var(--cc-maroon-dark) !important;
}

    /* File uploader */
    div[data-testid="stFileUploader"] section {
        background-color: rgba(255, 246, 229, 0.55);
        border: 1.5px dashed var(--cc-maroon-light);
        border-radius: 9px;
    }

    div[data-testid="stFileUploader"] button {
    background-color: var(--cc-maroon) !important;
    color: var(--cc-cream) !important;
    border: 1px solid var(--cc-maroon) !important;
    border-radius: 7px;
    font-weight: 600;
    }

    div[data-testid="stFileUploader"] button * {
    color: var(--cc-cream) !important;
    }

    div[data-testid="stFileUploader"] button:hover {
    background-color: var(--cc-maroon-dark) !important;
    color: var(--cc-cream) !important;
    border-color: var(--cc-maroon-dark) !important;
    }

    /* Review summary metric cards */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 246, 229, 0.78);
        border: 1px solid var(--cc-border);
        border-radius: 9px;
        padding: 16px 18px;
        box-shadow: none;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--cc-muted-brown);
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: var(--cc-maroon-dark);
    }

    /* Clause expanders */
    div[data-testid="stExpander"] {
        background-color: rgba(255, 246, 229, 0.78);
        border: 1px solid var(--cc-border);
        border-radius: 8px;
        box-shadow: none;
    }

    div[data-testid="stExpander"] summary {
        color: var(--cc-brown);
        font-weight: 600;
    }

    /* Main action buttons */
    div[data-testid="stButton"] button[kind="primary"] {
    background-color: var(--cc-maroon) !important;
    border: 1px solid var(--cc-maroon) !important;
    border-radius: 7px !important;
    }

    /* Analyze Contract button text */
    div[data-testid="stButton"] button[kind="primary"] p {
    color: #FFF8E9 !important;
    font-weight: 600 !important;
    }

    div[data-testid="stButton"] button[kind="primary"] span {
    color: #FFF8E9 !important;
    }

/* Analyze Contract hover */
div[data-testid="stButton"] button[kind="primary"]:hover {
    background-color: var(--cc-maroon-dark) !important;
    border-color: var(--cc-maroon-dark) !important;
}

div[data-testid="stButton"] button[kind="primary"]:hover p,
div[data-testid="stButton"] button[kind="primary"]:hover span {
    color: #FFF8E9 !important;
}

    /* Download buttons */
    .stDownloadButton > button {
        background-color: var(--cc-cream);
        color: var(--cc-maroon-dark);
        border: 1px solid var(--cc-maroon-light);
        border-radius: 7px;
        font-weight: 600;
    }

    .stDownloadButton > button:hover {
        background-color: #F2DAD5;
        color: var(--cc-maroon-dark);
        border-color: var(--cc-maroon);
    }

    /* Dividers */
    hr {
        border: none;
        border-top: 1px solid var(--cc-border);
        margin: 1.6rem 0;
    }
    /* --------------------------------
   RESULTS TEXT
   -------------------------------- */

/* Markdown text throughout results */
div[data-testid="stMarkdownContainer"] {
    color: var(--cc-brown);
}

div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li {
    color: var(--cc-brown) !important;
}

/* Human Review and Limitations lists */
div[data-testid="stMarkdownContainer"] ul {
    color: var(--cc-brown);
}

div[data-testid="stMarkdownContainer"] li {
    line-height: 1.6;
    margin-bottom: 0.55rem;
}

/* --------------------------------
   CLAUSE REVIEW EXPANDERS
   -------------------------------- */

div[data-testid="stExpander"] {
    background-color: var(--cc-cream) !important;
    border: 1px solid var(--cc-border) !important;
    border-radius: 9px !important;
    overflow: hidden;
    margin-bottom: 10px;
}

/* Expander header */
div[data-testid="stExpander"] details summary {
    background-color: #F8E9DF !important;
    color: var(--cc-brown) !important;
    padding: 5px 8px;
}

/* Keep expanded header light instead of black */
div[data-testid="stExpander"] details[open] summary {
    background-color: #EBCFC8 !important;
    color: var(--cc-maroon-dark) !important;
}

/* Text inside expander header */
div[data-testid="stExpander"] summary * {
    color: var(--cc-brown) !important;
}

div[data-testid="stExpander"] details[open] summary * {
    color: var(--cc-maroon-dark) !important;
}

/* Expander content */
div[data-testid="stExpander"] details > div {
    background-color: var(--cc-cream);
    color: var(--cc-brown);
}

/* Strong/bold labels */
div[data-testid="stExpander"] strong {
    color: var(--cc-maroon-dark);
}
    </style>
    """,
    unsafe_allow_html=True,
)

# ClauseCheck header banner
st.markdown(
"""<div class="cc-hero">
<div class="cc-hero-content">
<div class="cc-document-icon">§</div>
<div class="cc-vertical-line"></div>
<div>
<div class="cc-title">ClauseCheck</div>
<div class="cc-kicker">CONTRACT REVIEW TOOL</div>
</div>
</div>
</div>""",
    unsafe_allow_html=True,
)

# ClauseCheck description
st.markdown(
"""<div class="cc-intro">
<div class="cc-intro-title">ClauseCheck</div>
<div class="cc-intro-text">
ClauseCheck analyzes contracts to identify key provisions, retrieve supporting language and its location, and surface ambiguous, missing, or potentially significant terms for human review. Findings are AI-generated and may be incomplete or inaccurate, so ClauseCheck is designed to support structured contract analysis rather than replace legal judgment.
</div>
</div>""",
    unsafe_allow_html=True,
)

st.info(
    "ClauseCheck is an AI-assisted review tool and does not provide legal advice."
)

uploaded_file = st.file_uploader(
    "Upload a contract",
    type=["pdf"]
)

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")

    # Read the uploaded PDF
    reader = PdfReader(uploaded_file)

    contract_text = ""

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text:
            contract_text += f"\n\n--- PAGE {page_number} ---\n\n"
            contract_text += page_text

    # Show document information
    st.write(f"Pages detected: {len(reader.pages)}")

    if contract_text.strip():
        st.subheader("Extracted Contract Text")

        with st.expander("Preview extracted text"):
            st.text(contract_text[:5000])

        if st.button("Analyze Contract", type="primary"):
            with st.spinner("ClauseCheck is reviewing the contract..."):
                try:
                    # Stage 1: Extract all potentially relevant evidence
                    evidence_response = client.responses.parse(
                        model="gpt-5.4-mini",
                        instructions=EVIDENCE_EXTRACTION_PROMPT,
                        input=(
                            "Extract relevant evidence from the following contract.\n\n"
                            f"[CONTRACT TEXT]\n\n{contract_text}"
                        ),
                        text_format=EvidenceExtractionResult,
                    )

                    evidence_result = evidence_response.output_parsed
                    evidence_context = evidence_result.model_dump_json(indent=2)
                            
                    response = client.responses.parse(
                        model="gpt-5.4-mini",
                        instructions=CLAUSECHECK_PROMPT,
                        input=(
                            "Review the contract using the ClauseCheck framework.\n\n"
                            "Stage 1 has already searched the full contract and extracted potentially "
                            "relevant evidence for each provision. Analyze the extracted evidence below "
                            "to produce the findings and structured signals.\n\n"
                            "Treat the Stage 1 evidence as the evidence set for this analysis. "
                            "Consider all passages listed for each provision and evaluate their relationship "
                            "before setting the extraction signals.\n\n"
                            f"[EXTRACTED EVIDENCE]\n\n{evidence_context}"
                        ),
                        text_format=ClauseCheckResult,
                    )

                    result = response.output_parsed

                    st.subheader("ClauseCheck Review")

                    # Review summary
                    st.markdown("### Review Summary")

                    found_count = sum(
                        1 for clause in result.clauses
                        if clause.status == "FOUND"
                    )
                    review_count = sum(
                        1 for clause in result.clauses
                        if clause.status == "REVIEW"
                    )
                    ambiguous_count = sum(
                        1 for clause in result.clauses
                        if clause.status == "AMBIGUOUS"
                    )
                    missing_count = sum(
                        1 for clause in result.clauses
                        if clause.status == "MISSING"
                    )

                    col1, col2, col3, col4 = st.columns(4)

                    col1.metric("Found", found_count)
                    col2.metric("Review", review_count)
                    col3.metric("Ambiguous", ambiguous_count)
                    col4.metric("Missing", missing_count)

                    # Contract overview
                    st.markdown("### Contract Overview")

                    st.markdown(
                        f"**Document Type:** {result.overview.document_type}"
                    )
                    st.markdown(
                        f"**Parties:** {result.overview.parties}"
                    )
                    st.markdown(
                        f"**Effective Date:** {result.overview.effective_date}"
                    )
                    st.markdown(
                        f"**Contract Term:** {result.overview.contract_term}"
                    )

                    # Items requiring human review
                    if result.human_review_items:
                        st.markdown("### Items for Human Review")

                        for item in result.human_review_items:
                            st.markdown(f"- ⚠️ {item}")

                    # Missing provisions
                    if result.missing_provisions:
                        st.markdown("### Missing Provisions")

                        for item in result.missing_provisions:
                            st.markdown(f"- ❌ {item}")

                    # Limitations
                    if result.limitations:
                        st.markdown("### Limitations")

                        for item in result.limitations:
                            st.markdown(f"- ℹ️ {item}")

                    # Clause review
                    st.markdown("### Clause Review")

                    status_icons = {
                        "FOUND": "✅",
                        "REVIEW": "⚠️",
                        "AMBIGUOUS": "❓",
                        "MISSING": "❌",
                    }

                    for clause in result.clauses:
                        icon = status_icons.get(clause.status, "•")

                        with st.expander(
                            f"{icon} {clause.clause} — {clause.status}",
                            expanded=clause.status in [
                                "REVIEW",
                                "AMBIGUOUS",
                                "MISSING"
                            ]
                        ):
                            st.markdown("**Finding**")
                            st.write(clause.finding)

                            if clause.evidence:
                                st.markdown("**Evidence**")
                                st.write(clause.evidence)

                            if clause.location:
                                st.markdown("**Location**")
                                st.write(clause.location)

                            if clause.review_note:
                                st.markdown("**Review Note**")
                                st.write(clause.review_note)
                                
                            
                    # Generate downloadable report
                    report_text = build_report(
                        result,
                        uploaded_file.name
                    )

                    # Build downloadable reports
                    pdf_report = build_pdf_report(result)
                    csv_report = build_csv_report(result)

                    st.markdown("### Download Results")

                    download_col1, download_col2 = st.columns(2)

                    with download_col1:
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_report,
                            file_name="ClauseCheck_Review.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )

                    with download_col2:
                        st.download_button(
                            label="Download CSV Results",
                            data=csv_report,
                            file_name="ClauseCheck_Results.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                except Exception as e:
                    st.error(
                        f"An error occurred while analyzing the contract: {e}"
                    )

    else:
        st.error("No readable text was detected in this PDF.")
