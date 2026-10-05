"""Report generation service for AURA Vision.

Generates inspection reports in multiple formats:
- Plain Text (.txt)
- Professional PDF (.pdf) using fpdf2
- Formatted JSON (.json)
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from fpdf import FPDF


def sanitize_pdf_text(text: Any) -> str:
    """Replaces non-latin1 unicode characters with ASCII equivalents for core PDF fonts."""
    if text is None:
        return ""
    s = str(text)
    replacements = {
        "\u2014": "--",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "*",
        "\u00b0": " deg",
        "™": "TM",
        "©": "(c)",
    }
    for orig, rep in replacements.items():
        s = s.replace(orig, rep)
    return s.encode("latin-1", errors="replace").decode("latin-1")


def generate_text_report(analysis: Dict[str, Any]) -> str:
    """Generates the standardized plain-text inspection report including asset metadata."""
    meta = analysis.get("inspection_metadata") or {}
    asset_name = meta.get("asset_name") or "Unnamed Asset"
    asset_type = (meta.get("asset_type") or analysis.get("asset_type", "Unknown")).capitalize()
    location = meta.get("location") or "Unspecified Location"
    inspector = meta.get("inspector_name") or "Field Inspector"
    timestamp = meta.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    model_name = analysis.get("model", {}).get("name", "N/A")
    damage_items = analysis.get("damage", [])

    if damage_items:
        defects_str = ", ".join(
            [f"{d.get('type')} ({int(round(d.get('confidence', 0) * 100))}%)" for d in damage_items]
        )
    else:
        defects_str = "None detected (Clean inspection)"

    condition_score = analysis.get("condition", {}).get("score", "N/A")
    condition_status = str(analysis.get("condition", {}).get("status", "N/A")).upper()
    severity_level = str(analysis.get("severity", {}).get("level", "N/A")).upper()
    recommendation_action = analysis.get("recommendation", {}).get("action", "Maintain routine monitoring.")

    report = f"""=================================================
            AURA VISION INSPECTION REPORT
=================================================
INSPECTION METADATA:
Asset Name:     {asset_name}
Asset Type:     {asset_type}
Location:       {location}
Inspector:      {inspector}
Timestamp:      {timestamp}
-------------------------------------------------
ASSESSMENT RESULTS:
Model Used:     {model_name}
Detected Defects: {defects_str}
Condition Score: {condition_score} / 100 ({condition_status})
Severity Level: {severity_level}

Recommended Action:
- {recommendation_action}

DISCLAIMER:
This report is produced by AI-assisted screening tools.
A certified engineering evaluation is required for safety verification.
================================================="""
    return report


class InspectionPDF(FPDF):
    """Custom PDF generator for AURA Vision inspection reports."""

    def header(self):
        # Header banner
        self.set_fill_color(30, 41, 59)  # Dark slate #1E293B
        self.rect(0, 0, 210, 24, "F")
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 6)
        self.cell(0, 6, "AURA VISION -- INFRASTRUCTURE DAMAGE INSPECTION REPORT", align="L")
        self.set_font("Helvetica", "I", 9)
        self.set_xy(10, 14)
        self.cell(0, 4, "AI-Powered Computer Vision Damage Screening & Maintenance Analysis", align="L")
        self.ln(16)

    def footer(self):
        self.set_y(-18)
        self.set_draw_color(203, 213, 225)
        self.line(10, self.get_y(), 200, self.get_y())
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 116, 139)
        self.set_xy(10, -14)
        self.cell(0, 5, "CONFIDENTIAL & PROPRIETARY -- AURA VISION INSPECTION SYSTEM", align="L")
        self.set_xy(10, -14)
        self.cell(0, 5, f"Page {self.page_no()}", align="R")


def generate_pdf_report(analysis: Dict[str, Any]) -> bytes:
    """Generates a professional PDF inspection report as raw bytes."""
    meta = analysis.get("inspection_metadata") or {}
    asset_name = sanitize_pdf_text(meta.get("asset_name") or "Unnamed Asset")
    asset_type = sanitize_pdf_text((meta.get("asset_type") or analysis.get("asset_type", "Unknown")).capitalize())
    location = sanitize_pdf_text(meta.get("location") or "Unspecified Location")
    inspector = sanitize_pdf_text(meta.get("inspector_name") or "Field Inspector")
    timestamp = sanitize_pdf_text(meta.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    model_info = analysis.get("model", {})
    model_name = sanitize_pdf_text(model_info.get("name", "N/A"))

    cond = analysis.get("condition", {})
    cond_score = sanitize_pdf_text(cond.get("score", "N/A"))
    cond_status = sanitize_pdf_text(str(cond.get("status", "N/A")).upper())

    sev = analysis.get("severity", {})
    sev_level = sanitize_pdf_text(str(sev.get("level", "N/A")).upper())
    sev_score = sanitize_pdf_text(sev.get("score", "N/A"))

    rec = analysis.get("recommendation", {})
    rec_action = sanitize_pdf_text(rec.get("action", "Maintain routine monitoring."))
    rec_prio = sanitize_pdf_text(str(rec.get("priority", "N/A")).upper())

    damage_items = analysis.get("damage", [])

    pdf = InspectionPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Section 1: Inspection Metadata Table
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 8, "1. Inspection & Asset Metadata", ln=True)
    pdf.ln(1)

    meta_rows = [
        ("Asset Name:", asset_name, "Asset Category:", asset_type),
        ("Location:", location, "Inspector Name:", inspector),
        ("Inspection Date / Time:", timestamp, "Model Engine:", model_name),
    ]

    for label1, val1, label2, val2 in meta_rows:
        pdf.set_fill_color(248, 250, 252)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(42, 7, f" {label1}", border=1, fill=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(53, 7, f" {val1}", border=1)

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(42, 7, f" {label2}", border=1, fill=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(53, 7, f" {val2}", border=1)
        pdf.ln()

    pdf.ln(5)

    # Section 2: Key Assessment Metrics
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 8, "2. Structural Condition & Severity Rating", ln=True)
    pdf.ln(1)

    col_w = 95
    # Box 1: Condition Score
    pdf.set_fill_color(241, 245, 249)
    pdf.rect(10, pdf.get_y(), col_w, 20, "F")
    pdf.set_xy(14, pdf.get_y() + 2)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(col_w - 8, 4, "OVERALL CONDITION SCORE", ln=True)
    pdf.set_x(14)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 64, 175)
    pdf.cell(col_w - 8, 7, f"{cond_score} / 100 -- {cond_status}", ln=True)

    # Box 2: Severity Level
    pdf.set_fill_color(241, 245, 249)
    pdf.rect(105, pdf.get_y() - 13, col_w, 20, "F")
    pdf.set_xy(109, pdf.get_y() - 11)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(col_w - 8, 4, "DAMAGE SEVERITY ASSESSMENT", ln=True)
    pdf.set_x(109)
    pdf.set_font("Helvetica", "B", 13)
    if sev_level in ["CRITICAL", "HIGH"]:
        pdf.set_text_color(185, 28, 28)
    elif sev_level == "MEDIUM":
        pdf.set_text_color(180, 83, 9)
    else:
        pdf.set_text_color(21, 128, 61)
    pdf.cell(col_w - 8, 7, f"{sev_level} ({sev_score} / 100)", ln=True)

    pdf.set_y(pdf.get_y() + 9)

    # Section 3: Defect Findings Table
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 8, f"3. Detected Structural Findings ({len(damage_items)} item(s))", ln=True)
    pdf.ln(1)

    pdf.set_fill_color(30, 41, 59)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(15, 7, " #", border=1, fill=True)
    pdf.cell(50, 7, " Defect Type", border=1, fill=True)
    pdf.cell(35, 7, " Model Confidence", border=1, fill=True)
    pdf.cell(90, 7, " Bounding Box Coordinates / Scope", border=1, fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)

    if damage_items:
        for idx, item in enumerate(damage_items, 1):
            pdf.set_fill_color(248, 250, 252) if idx % 2 == 0 else pdf.set_fill_color(255, 255, 255)
            d_type = sanitize_pdf_text(item.get("type", "Unknown"))
            conf_val = f"{int(round(item.get('confidence', 0) * 100))}%"
            bbox = item.get("bounding_box")
            if bbox:
                bbox_str = f"x1:{bbox.get('x1')}, y1:{bbox.get('y1')} to x2:{bbox.get('x2')}, y2:{bbox.get('y2')}"
            else:
                bbox_str = "Whole-surface classification (No bounding box)"
            bbox_str = sanitize_pdf_text(bbox_str)

            pdf.cell(15, 7, f" {idx}", border=1, fill=True)
            pdf.cell(50, 7, f" {d_type}", border=1, fill=True)
            pdf.cell(35, 7, f" {conf_val}", border=1, fill=True)
            pdf.cell(90, 7, f" {bbox_str}", border=1, fill=True)
            pdf.ln()
    else:
        pdf.cell(190, 7, " No defects identified during screening.", border=1, ln=True)

    pdf.ln(5)

    # Section 4: Maintenance Guidance
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 8, "4. Recommended Actionable Maintenance Guidance", ln=True)
    pdf.ln(1)

    pdf.set_fill_color(238, 242, 255)
    pdf.rect(10, pdf.get_y(), 190, 20, "F")
    pdf.set_xy(14, pdf.get_y() + 2)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(67, 56, 202)
    pdf.cell(182, 4, f"ACTION PRIORITY: {rec_prio}", ln=True)
    pdf.set_x(14)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 41, 59)
    pdf.multi_cell(182, 4.5, f"- {rec_action}")

    pdf.set_y(pdf.get_y() + 8)

    # Section 5: Engineering Disclaimer
    pdf.set_fill_color(254, 242, 242)
    pdf.set_draw_color(239, 68, 68)
    pdf.rect(10, pdf.get_y(), 190, 18, "DF")
    pdf.set_xy(14, pdf.get_y() + 2)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(153, 27, 27)
    pdf.cell(182, 4, "DISCLAIMER & ENGINEERING NOTICE:", ln=True)
    pdf.set_x(14)
    pdf.set_font("Helvetica", "", 8)
    pdf.multi_cell(
        182,
        4,
        "This report is generated by automated AI screening models for decision-support and prioritization only. "
        "It does not replace certified structural engineering evaluations, physical sampling, or on-site inspections.",
    )

    return bytes(pdf.output())


def generate_json_report(analysis: Dict[str, Any]) -> str:
    """Returns the formatted JSON representation of the inspection analysis."""
    return json.dumps(analysis, indent=2)
