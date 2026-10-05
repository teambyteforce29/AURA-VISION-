"""AURA Vision - Interactive Streamlit Frontend Dashboard.

AI-powered infrastructure damage inspection and maintenance recommendations.
Provides split-screen visual analysis, defect breakdown, condition metrics,
and multi-format downloadable inspection reports (PDF, TXT, JSON).
"""

import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import streamlit as st
from PIL import Image

# Ensure project root is in python path for local fallback execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.reports import (
    generate_text_report,
    generate_pdf_report,
    generate_json_report,
)

# Configuration
DEFAULT_API_URL = os.getenv("AURA_API_URL", "http://localhost:8000")
LOGO_PATH = PROJECT_ROOT / "frontend" / "assets" / "logo.png"


# Page Setup
st.set_page_config(
    page_title="AURA Vision - AI Infrastructure Damage Inspection",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    .badge-low {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-medium {
        background-color: #FEF08A;
        color: #713F12;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-high {
        background-color: #FFEDD5;
        color: #9A3412;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-critical {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .disclaimer-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #1E40AF;
        margin-top: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_severity_badge_html(level: str) -> str:
    """Returns styled HTML badge according to severity category."""
    lvl = (level or "").lower()
    if lvl == "critical":
        return f'<span class="badge-critical">CRITICAL</span>'
    elif lvl == "high":
        return f'<span class="badge-high">HIGH</span>'
    elif lvl == "medium":
        return f'<span class="badge-medium">MEDIUM</span>'
    else:
        return f'<span class="badge-low">LOW</span>'


def call_backend_api(
    api_url: str,
    asset_type: str,
    image_bytes: bytes,
    filename: str,
    asset_name: str = "Unnamed Asset",
    location: str = "Unspecified Location",
    inspector_name: str = "Field Inspector",
) -> Optional[Dict[str, Any]]:
    """Calls the FastAPI /api/analyze endpoint with multipart/form-data."""
    endpoint = f"{api_url.rstrip('/')}/api/analyze"
    files = {"file": (filename, image_bytes, "image/jpeg")}
    data = {
        "asset_type": asset_type,
        "asset_name": asset_name,
        "location": location,
        "inspector_name": inspector_name,
    }

    try:
        response = requests.post(endpoint, files=files, data=data, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error ({response.status_code}): {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not reach backend API at {endpoint}: {e}")
        return None


def run_local_fallback_pipeline(
    asset_type: str,
    pil_image: Image.Image,
    asset_name: str = "Unnamed Asset",
    location: str = "Unspecified Location",
    inspector_name: str = "Field Inspector",
) -> Dict[str, Any]:
    """Fallback runner if backend API is not running locally."""
    from backend.inference.router import process_inspection
    from backend.main import resolve_models_dir, MODEL_FILENAMES
    from ultralytics import YOLO

    models_dir = resolve_models_dir()
    model_path = models_dir / MODEL_FILENAMES[asset_type]
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Load on-demand for fallback
    yolo_model = YOLO(str(model_path))
    registry = {asset_type: yolo_model}

    resp = process_inspection(
        asset_type=asset_type,
        pil_image=pil_image,
        model_registry=registry,
        asset_name=asset_name,
        location=location,
        inspector_name=inspector_name,
    )
    return resp.model_dump()


# Sidebar Navigation
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=110)
    st.title("AURA Vision")
    st.caption("AI Infrastructure Damage Inspection")
    st.divider()

    asset_tab = st.selectbox(
        "Select Inspection Category:",
        options=[
            ("road", "🚧 Road Inspection"),
            ("bridge", "🌉 Bridge Inspection"),
            ("building", "🏢 Building Inspection"),
            ("concrete", "🧱 Concrete Inspection"),
        ],
        format_func=lambda x: x[1],
    )
    selected_asset = asset_tab[0]

# Silent backend health check
api_url = DEFAULT_API_URL
api_online = False
try:
    h_resp = requests.get(f"{api_url.rstrip('/')}/api/health", timeout=1.5)
    api_online = (h_resp.status_code == 200)
except Exception:
    api_online = False


# Header with Logo above application name
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=120)

st.markdown('<div class="main-title">AURA Vision - AI Infrastructure Damage Inspection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Real-time computer vision analysis, structural severity estimation, and actionable maintenance planning.</div>',
    unsafe_allow_html=True,
)

# Active Inspection Header
asset_labels = {
    "road": ("🚧 Road Damage Inspection", "Detection of Cracks, Potholes, and Surface Erosion"),
    "bridge": ("🌉 Bridge Structural Screening", "Detection of Rust, Spalling, Cavities, Weathering, Efflorescence, and Cracks"),
    "building": ("🏢 Building Crack Classification", "Whole-surface crack classification on concrete/masonry walls"),
    "concrete": ("🧱 Concrete & Corrosion Inspection", "Specialized corrosion and concrete degradation analysis"),
}
title_txt, sub_txt = asset_labels[selected_asset]
st.subheader(title_txt)
st.caption(sub_txt)

# 1. Asset & Field Inspection Details Form
with st.expander("📋 Inspection & Asset Metadata", expanded=True):
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        asset_name = st.text_input(
            "Asset Name / Structure ID:",
            value=f"{selected_asset.capitalize()} Structure #104",
            help="Designate the specific facility, highway section, or structure ID.",
        )
    with col_m2:
        location = st.text_input(
            "Inspection Location / Station:",
            value="Sector 4B, Station 12+50",
            help="Physical, structural, or GPS location coordinates.",
        )
    with col_m3:
        inspector_name = st.text_input(
            "Inspector Name & Credentials:",
            value="Eng. Alex Miller, PE",
            help="Name and credentials of the inspecting civil/structural engineer.",
        )

# 2. Image Upload
uploaded_file = st.file_uploader(
    f"Upload inspection image for {selected_asset.capitalize()} analysis (JPG, PNG, WEBP):",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload an infrastructure surface photo for AI analysis.",
)

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        st.error(f"Error opening image file: {e}")
        st.stop()

    analyze_btn = st.button("🚀 Run AI Inspection", type="primary", use_container_width=True)

    if analyze_btn:
        with st.spinner(f"Analyzing {selected_asset} image using AURA Vision deep learning models..."):
            result_data = None
            if api_online:
                result_data = call_backend_api(
                    api_url=api_url,
                    asset_type=selected_asset,
                    image_bytes=image_bytes,
                    filename=uploaded_file.name,
                    asset_name=asset_name,
                    location=location,
                    inspector_name=inspector_name,
                )

            # Fallback to local execution if backend is not reachable
            if not result_data:
                try:
                    result_data = run_local_fallback_pipeline(
                        asset_type=selected_asset,
                        pil_image=pil_image,
                        asset_name=asset_name,
                        location=location,
                        inspector_name=inspector_name,
                    )
                except Exception as ex:
                    st.error(f"Analysis failed: {ex}")
                    st.stop()

            st.session_state["last_result"] = result_data
            st.session_state["original_img"] = pil_image

    # Display results if available
    if "last_result" in st.session_state and st.session_state.get("last_result") is not None:
        result = st.session_state["last_result"]
        orig_img = st.session_state.get("original_img", pil_image)

        st.divider()

        # Split Layout: Left Panel = Visuals, Right Panel = Insights
        left_col, right_col = st.columns([1.1, 1.0], gap="large")

        # LEFT PANEL: Images
        with left_col:
            st.markdown("### 📷 Visual Inspection")

            # Check if annotated image is returned
            annotated_b64 = result.get("annotated_image")
            if annotated_b64 and result.get("model", {}).get("task") == "object_detection":
                from backend.utils.image import base64_to_pil

                try:
                    annot_img = base64_to_pil(annotated_b64)
                    tab1, tab2 = st.tabs(["AI Detections (Annotated)", "Original Image"])
                    with tab1:
                        st.image(annot_img, use_container_width=True, caption="Model Annotations & Bounding Boxes")
                    with tab2:
                        st.image(orig_img, use_container_width=True, caption="Original Input Image")
                except Exception:
                    st.image(orig_img, use_container_width=True, caption="Original Image")
            else:
                # Classification (Building): original image untouched
                st.image(
                    orig_img,
                    use_container_width=True,
                    caption="Original Building Wall Image (Classification Mode - No Bounding Boxes)",
                )

        # RIGHT PANEL: Results, Metrics & Recommendations
        with right_col:
            st.markdown("### 📊 Assessment Findings")

            # Inspection Metadata Banner
            meta = result.get("inspection_metadata") or {}
            if meta:
                st.markdown(
                    f"""
                    <div style="background-color: #F1F5F9; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; font-size: 0.88rem; border-left: 3px solid #64748B;">
                        <strong>Asset:</strong> {meta.get('asset_name', 'N/A')}<br>
                        <strong>Location:</strong> {meta.get('location', 'N/A')} &nbsp;|&nbsp; <strong>Inspector:</strong> {meta.get('inspector_name', 'N/A')}<br>
                        <span style="color: #64748B; font-size: 0.8rem;">Timestamp: {meta.get('timestamp', 'N/A')}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Top Key Metrics
            cond_score = result.get("condition", {}).get("score", 0)
            cond_status = result.get("condition", {}).get("status", "N/A")
            sev_level = result.get("severity", {}).get("level", "N/A")
            sev_score = result.get("severity", {}).get("score", 0)

            m1, m2 = st.columns(2)
            with m1:
                st.metric(
                    label="Condition Score",
                    value=f"{cond_score} / 100",
                    delta=f"{str(cond_status).upper()}",
                    delta_color="normal" if cond_score >= 60 else "inverse",
                )
            with m2:
                st.markdown(
                    f"""
                    <div style="padding-top: 5px;">
                        <span style="font-size:0.875rem; color:#475569; font-weight:600;">Damage Severity</span><br>
                        <div style="margin-top:6px;">{get_severity_badge_html(sev_level)} &nbsp; <strong>{sev_score}/100</strong></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write("")

            # Model Details
            model_info = result.get("model", {})
            st.caption(f"**Model:** `{model_info.get('name')}` | **Task:** `{model_info.get('task')}`")

            # Detected Damage List
            st.markdown("#### Detected Defects")
            damages = result.get("damage", [])
            if damages:
                for idx, d in enumerate(damages, 1):
                    d_type = d.get("type", "Unknown")
                    d_conf = int(round(d.get("confidence", 0) * 100))
                    bbox = d.get("bounding_box")

                    with st.expander(f"Defect #{idx}: **{d_type}** — Confidence: **{d_conf}%**", expanded=True):
                        st.progress(d_conf / 100.0)
                        if bbox:
                            st.write(
                                f"📍 Bounding Box: `[x1: {bbox.get('x1')}, y1: {bbox.get('y1')}, x2: {bbox.get('x2')}, y2: {bbox.get('y2')}]`"
                            )
            else:
                st.success("✅ No structural damage detected in this image.")

            # Actionable Recommendation Card
            st.markdown("#### Recommended Maintenance Action")
            rec = result.get("recommendation", {})
            rec_action = rec.get("action", "Maintain routine monitoring.")
            rec_prio = rec.get("priority", "Low")

            st.info(f"**Priority:** `{rec_prio}`\n\n📌 **Action:** {rec_action}")

            # Multi-Format Report Generator with Selector
            st.markdown("#### 📄 Export Inspection Report")
            format_choice = st.radio(
                "Select Export Format:",
                options=["📄 PDF Report (.pdf)", "📝 Text Summary (.txt)", "💾 Full JSON (.json)"],
                horizontal=True,
            )

            safe_asset = selected_asset.lower()
            if "PDF" in format_choice:
                try:
                    pdf_bytes = generate_pdf_report(result)
                    st.download_button(
                        label="📥 Download Inspection Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"AURA_Vision_{safe_asset}_Inspection_Report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                    st.caption("Includes executive summary, metadata table, condition metrics, findings table, and engineering disclaimer.")
                except Exception as pdf_err:
                    st.error(f"Error generating PDF: {pdf_err}")
            elif "Text" in format_choice:
                text_report = generate_text_report(result)
                st.download_button(
                    label="📥 Download Plaintext Report (.txt)",
                    data=text_report,
                    file_name=f"AURA_Vision_{safe_asset}_Inspection_Report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
                with st.expander("Preview Text Report"):
                    st.code(text_report, language="text")
            else:  # JSON
                json_str = generate_json_report(result)
                st.download_button(
                    label="💾 Download Raw Assessment Data (.json)",
                    data=json_str,
                    file_name=f"AURA_Vision_{safe_asset}_Analysis_Data.json",
                    mime="application/json",
                    use_container_width=True,
                )
                with st.expander("Preview JSON Payload"):
                    st.json(result)

        # Bottom Disclaimer
        st.markdown(
            """
            <div class="disclaimer-box">
                <strong>DISCLAIMER:</strong> This report is produced by AI-assisted screening tools.
                A certified engineering evaluation is required for safety verification and formal sign-off.
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info("👆 Please review the metadata, select an inspection category, and upload an infrastructure image to begin.")
