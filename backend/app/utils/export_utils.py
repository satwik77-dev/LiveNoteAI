"""Export utilities — PDF (reportlab) and JSON export for meeting sessions."""
from __future__ import annotations

import io
import json
from datetime import datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ..models.memory import MemoryState


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_json(session: MemoryState) -> bytes:
    """Serialise the full meeting state to indented JSON bytes."""
    data: dict[str, Any] = session.model_dump()
    # Add a human-readable export timestamp
    data["exported_at"] = datetime.utcnow().isoformat() + "Z"
    return json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")


# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------

def export_pdf(session: MemoryState) -> bytes:
    """Generate a formatted PDF meeting report and return it as bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.85 * inch,
    )

    styles = getSampleStyleSheet()
    story = []

    # ── Header ───────────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "LiveNoteTitle",
        parent=styles["Title"],
        fontSize=20,
        spaceAfter=4,
        textColor=colors.HexColor("#1a1a2e"),
    )
    meta_style = ParagraphStyle(
        "LiveNoteMeta",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#6b7280"),
        spaceAfter=2,
    )
    section_style = ParagraphStyle(
        "LiveNoteSection",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=14,
        spaceAfter=6,
        textColor=colors.HexColor("#1e3a5f"),
        borderPad=2,
    )
    body_style = ParagraphStyle(
        "LiveNoteBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        spaceAfter=4,
    )
    item_style = ParagraphStyle(
        "LiveNoteItem",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        leftIndent=12,
        spaceAfter=3,
    )

    story.append(Paragraph("LiveNote Meeting Report", title_style))
    story.append(Paragraph(f"Meeting ID: {session.meeting_id}", meta_style))
    story.append(Paragraph(f"Date: {session.meeting_start_date}", meta_style))
    story.append(Paragraph(f"Started: {session.started_at}", meta_style))
    story.append(Paragraph(f"Status: {'Active' if session.is_active else 'Ended'}", meta_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e5e7eb"), spaceAfter=10))

    # ── Summary ──────────────────────────────────────────────────────────────
    story.append(Paragraph("Summary", section_style))
    summary_text = session.running_summary or "<i>No summary generated yet.</i>"
    if session.summary_human_locked:
        summary_text += " <font color='#059669'>[human-verified]</font>"
    story.append(Paragraph(summary_text, body_style))

    # ── Action Items ─────────────────────────────────────────────────────────
    story.append(Paragraph("Action Items", section_style))
    active_actions = [a for a in session.action_items if a.status != "deleted"]
    if active_actions:
        table_data = [["Task", "Owner", "Deadline", "Priority", "Status"]]
        for item in active_actions:
            lock_mark = " ✓" if item.human_locked else ""
            review_mark = " ⚠" if item.needs_review else ""
            table_data.append([
                Paragraph(f"{item.task}{lock_mark}{review_mark}", item_style),
                item.owner,
                item.deadline_normalized if item.deadline_normalized != "unspecified" else item.deadline,
                item.priority,
                item.status,
            ])
        table = Table(table_data, colWidths=[3.0 * inch, 1.2 * inch, 1.1 * inch, 0.8 * inch, 0.7 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(table)
    else:
        story.append(Paragraph("<i>No action items recorded.</i>", body_style))

    # ── Decisions ────────────────────────────────────────────────────────────
    story.append(Paragraph("Decisions", section_style))
    active_decisions = [d for d in session.decisions if not getattr(d, "deleted", False)]
    if active_decisions:
        for decision in active_decisions:
            lock_mark = " <font color='#059669'>[human-verified]</font>" if decision.human_locked else ""
            story.append(Paragraph(f"• {decision.decision}{lock_mark}", item_style))
    else:
        story.append(Paragraph("<i>No decisions recorded.</i>", body_style))

    # ── Risks ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("Risks", section_style))
    active_risks = [r for r in session.risks if not getattr(r, "deleted", False)]
    if active_risks:
        for risk in active_risks:
            lock_mark = " <font color='#059669'>[human-verified]</font>" if risk.human_locked else ""
            story.append(Paragraph(f"• {risk.risk}{lock_mark}", item_style))
    else:
        story.append(Paragraph("<i>No risks recorded.</i>", body_style))

    # ── Speakers ─────────────────────────────────────────────────────────────
    if session.known_speakers:
        story.append(Paragraph("Participants", section_style))
        story.append(Paragraph(", ".join(session.known_speakers), body_style))

    # ── Processing Stats ─────────────────────────────────────────────────────
    if session.chunk_history:
        story.append(Paragraph("Processing Summary", section_style))
        total_chunks = len(session.chunk_history)
        total_utterances = sum(c.utterance_count for c in session.chunk_history)
        total_violations = sum(c.trust_violations for c in session.chunk_history)
        story.append(Paragraph(
            f"Chunks processed: {total_chunks} &nbsp;|&nbsp; "
            f"Utterances: {total_utterances} &nbsp;|&nbsp; "
            f"Trust violations filtered: {total_violations}",
            body_style,
        ))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.3 * inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb")))
    story.append(Paragraph(
        f"Generated by LiveNote &nbsp;|&nbsp; {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        meta_style,
    ))

    doc.build(story)
    return buffer.getvalue()
