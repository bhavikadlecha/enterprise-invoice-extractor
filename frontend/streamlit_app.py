"""
frontend/streamlit_app.py
--------------------------
Drag-and-drop Streamlit UI for the Vision-RAG Invoice Extractor.
Features:
  - Mode Selection: Single Invoice vs Batch Upload
  - Single Mode: HITL (Human-in-the-loop) Editing & Saving
  - Batch Mode: Async ZIP processing & JSON downloads
  - Persona-Based Tabs (Accountant vs Developer)
"""

import base64
import json
import time
import requests
import pandas as pd
import streamlit as st

import os
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
EXTRACT_ENDPOINT = f"{API_BASE}/extract-invoice"
APPROVE_ENDPOINT = f"{API_BASE}/approve-invoice"
BATCH_UPLOAD_ENDPOINT = f"{API_BASE}/batch/upload"
METRICS_ENDPOINT = f"{API_BASE}/metrics/summary"
PO_ENDPOINT = f"{API_BASE}/purchase-orders"

st.set_page_config(page_title="Invoice Data Extractor", layout="wide")
st.title("Enterprise Invoice Extractor")
st.caption("AI-powered OCR, Math Validation, RAG PO Matching, and Batch Processing.")

# ---------------------------------------------------------------------------
# Sidebar: System Dashboard & PO Management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("System Dashboard")
    st.caption("Cumulative metrics across all processed invoices.")
    if st.button("Refresh Dashboard"):
        st.rerun()

    try:
        metrics_resp = requests.get(METRICS_ENDPOINT, timeout=5)
        if metrics_resp.status_code == 200:
            m = metrics_resp.json()
            total = m.get("total_extractions", 0)
            successful = m.get("successful", 0)
            st.metric("Total Extractions", total)
            st.metric("Success Rate", f"{100 * successful / max(total, 1):.1f}%")
            st.divider()
            st.metric("Avg Latency", f"{m.get('avg_latency_seconds', 0):.2f}s")
            st.metric("Self-Healing Trigger Rate", f"{m.get('self_healing_trigger_pct', 0):.1f}%")
            st.divider()
            st.metric("PO Matches", m.get("po_matches", 0))
        else:
            st.warning("Could not load metrics.")
    except Exception:
        st.info("Start the API server to see live metrics.")

    st.divider()
    
    with st.expander("Add New Purchase Order (RAG)"):
        st.caption("Embed a new authorized vendor into the Vector Database.")
        with st.form("add_po_form", clear_on_submit=True):
            new_po_num = st.text_input("PO Number (e.g., PO-2026-999)")
            new_vendor = st.text_input("Vendor Name (e.g., Apple Inc)")
            new_total = st.number_input("Approved Total Amount", min_value=0.0, format="%.2f")
            new_items = st.text_input("Approved Items (e.g., MacBooks)")
            
            if st.form_submit_button("Embed & Store PO"):
                if new_po_num and new_vendor:
                    resp = requests.post(PO_ENDPOINT, json={
                        "po_number": new_po_num,
                        "vendor_name": new_vendor,
                        "total_amount": new_total,
                        "items": new_items
                    })
                    if resp.status_code == 201:
                        st.success("Successfully injected into Vector DB!")
                    else:
                        st.error("Failed to store PO.")
                else:
                    st.warning("PO Number and Vendor Name are required.")

# ---------------------------------------------------------------------------
# Main Mode Selector
# ---------------------------------------------------------------------------
app_mode = st.radio("Select Processing Mode", ["Single Invoice (HITL Edit)", "Batch Upload (Multi-file)"], horizontal=True)
st.divider()

if app_mode == "Batch Upload (Multi-file)":
    # ====================================================================
    # BATCH PROCESSING MODE
    # ====================================================================
    st.subheader("Batch Invoice Processing")
    st.markdown("Drag and drop multiple PDFs or Images (max 50) directly below. The system will process them all asynchronously.")
    
    batch_files = st.file_uploader(
        "Upload Multiple Invoices", 
        type=["pdf", "jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if batch_files:
        if st.button(f"Start Batch Job ({len(batch_files)} files)", type="primary"):
            import zipfile
            import io
            
            with st.spinner("Preparing and queueing batch job..."):
                # Zip the files in-memory so we don't have to change our backend API
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for f in batch_files:
                        zip_file.writestr(f.name, f.read())
                
                zip_buffer.seek(0)
                
                # Send the in-memory zip to the backend
                resp = requests.post(
                    BATCH_UPLOAD_ENDPOINT,
                    files={"file": ("batch.zip", zip_buffer, "application/zip")}
                )
                if resp.status_code == 202:
                    job_data = resp.json()
                    st.session_state["batch_job_id"] = job_data["job_id"]
                    st.success(f"Job queued successfully! {job_data['total_files']} files to process.")
                else:
                    st.error(f"Upload failed: {resp.text}")

    # Polling logic if job exists
    if "batch_job_id" in st.session_state:
        job_id = st.session_state["batch_job_id"]
        st.markdown("### Job Progress")
        
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        results_placeholder = st.empty()
        
        is_running = True
        while is_running:
            try:
                poll_resp = requests.get(f"{API_BASE}/batch/{job_id}", timeout=5)
                if poll_resp.status_code == 200:
                    job = poll_resp.json()
                    status = job["status"]
                    total = job["total_files"]
                    processed = job["processed_files"]
                    
                    progress = processed / max(total, 1)
                    progress_bar.progress(progress)
                    status_placeholder.info(f"Status: **{status.upper()}** | Processed {processed} / {total} files")
                    
                    if status in ["completed", "failed"]:
                        is_running = False
                        st.success("Batch Processing Finished!")
                        
                        # Render results download
                        results = job["results"]
                        if results:
                            results_json = json.dumps(results, indent=2)
                            st.download_button(
                                label="Download Consolidated JSON Results",
                                data=results_json,
                                file_name=f"batch_results_{job_id}.json",
                                mime="application/json",
                                type="primary"
                            )
                            with st.expander("View Batch Logs"):
                                for r in results:
                                    if r["status"] == "success":
                                        st.write(f"Success: {r['filename']}")
                                    else:
                                        st.write(f"Failed: {r['filename']} - Error: {r.get('error')}")
                else:
                    status_placeholder.error("Failed to poll job status.")
                    is_running = False
            except Exception as e:
                status_placeholder.error(f"Polling error: {e}")
                is_running = False
                
            if is_running:
                time.sleep(2)

else:
    # ====================================================================
    # SINGLE INVOICE MODE (HITL)
    # ====================================================================
    uploaded_file = st.file_uploader(
        "Upload a PDF or Image Invoice",
        type=["pdf", "jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name
        filename_lower = filename.lower()
        content_type = "application/pdf" if filename_lower.endswith(".pdf") else "image/jpeg"

        col_left, col_right = st.columns([1, 1])

        # ── Left Column: Preview ──
        with col_left:
            st.subheader("Document Preview")
            if filename_lower.endswith(".pdf"):
                b64_pdf = base64.b64encode(file_bytes).decode("utf-8")
                st.markdown(
                    f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="100%" height="600px" style="border: 1px solid #ddd; border-radius: 8px;"></iframe>',
                    unsafe_allow_html=True
                )
            else:
                st.image(file_bytes, width="stretch", caption=filename)

        # ── Right Column: Extraction ──
        with col_right:
            st.subheader("Extraction & Verification")
            
            # Use session state to hold the AI response so it doesn't re-extract on every keystroke
            if "extract_response" not in st.session_state or st.session_state.get("last_file") != filename:
                with st.spinner("Processing invoice data..."):
                    try:
                        resp = requests.post(
                            EXTRACT_ENDPOINT,
                            files={"file": (filename, file_bytes, content_type)},
                            timeout=120,
                        )
                        if resp.status_code == 200:
                            st.session_state["extract_response"] = resp.json()
                            st.session_state["last_file"] = filename
                        else:
                            st.error(f"Extraction failed: {resp.text}")
                            st.stop()
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.stop()
            
            if "extract_response" in st.session_state:
                data = st.session_state["extract_response"]
                invoice = data["invoice"]
                po_match = data["po_match"]
                metrics = data["metrics"]
                curr = invoice.get("Currency", "")
                c_str = f"{curr} " if len(curr) > 1 else curr

                st.success("Extraction Successful. Review and edit fields below.")

                tab_accountant, tab_developer = st.tabs(["Accountant View (HITL)", "Developer Console"])

                # ── ACCOUNTANT TAB (HITL) ──
                with tab_accountant:
                    with st.form("hitl_approval_form"):
                        st.markdown("#### Invoice Details")
                        col1, col2, col3 = st.columns(3)
                        edited_vendor = col1.text_input("Vendor", value=invoice.get("VendorName", ""))
                        edited_date = col2.text_input("Invoice Date", value=invoice.get("InvoiceDate", ""))
                        edited_taxid = col3.text_input("Tax ID", value=invoice.get("TaxID", ""))

                        st.markdown("#### Financials")
                        col1, col2, col3, col4 = st.columns(4)
                        edited_subtotal = col1.number_input("Subtotal", value=float(invoice.get("Subtotal") or 0.0), format="%.2f")
                        edited_tax = col2.number_input("Tax", value=float(invoice.get("Tax") or 0.0), format="%.2f")
                        edited_shipping = col3.number_input("Shipping", value=float(invoice.get("ShippingFee") or 0.0), format="%.2f")
                        edited_discount = col4.number_input("Discount", value=float(invoice.get("Discount") or 0.0), format="%.2f")
                        edited_total = st.number_input("Total Amount", value=float(invoice.get("TotalAmount") or 0.0), format="%.2f")

                        # Math Check on edited values
                        expected = edited_subtotal + edited_tax + edited_shipping - edited_discount
                        if abs(expected - edited_total) <= 0.02:
                            st.success(f"Math Verification Passed ({c_str}{edited_total:.2f})")
                        else:
                            st.error(f"Math Error: Expected {c_str}{expected:.2f} but Total is {c_str}{edited_total:.2f}")

                        st.markdown("#### Line Items")
                        df_items = pd.DataFrame(invoice.get("LineItems", []))
                        # Make table editable
                        edited_items_df = st.data_editor(df_items, num_rows="dynamic", use_container_width=True)

                        # PO Match Display (Read Only)
                        st.markdown("#### Purchase Order Verification")
                        if po_match.get("matched"):
                            st.info(f"Approved: Matched to **{po_match['po_number']}** ({po_match.get('confidence_score', 0):.0%})")
                        else:
                            st.warning(f"No PO Match Found: {po_match.get('reason', '—')}")

                        # Approve Button
                        submitted = st.form_submit_button("Approve & Save to Database", type="primary")
                        if submitted:
                            final_invoice = {
                                "VendorName": edited_vendor,
                                "InvoiceDate": edited_date,
                                "TaxID": edited_taxid,
                                "Currency": curr,
                                "Subtotal": edited_subtotal,
                                "Tax": edited_tax,
                                "ShippingFee": edited_shipping,
                                "Discount": edited_discount,
                                "TotalAmount": edited_total,
                                "LineItems": edited_items_df.to_dict("records")
                            }
                            with st.spinner("Saving to database..."):
                                save_resp = requests.post(APPROVE_ENDPOINT, json=final_invoice)
                                if save_resp.status_code == 201:
                                    st.success(f"Invoice approved and saved! ({save_resp.json().get('file_saved')})")
                                else:
                                    st.error("Failed to save.")

                # ── DEVELOPER TAB ──
                with tab_developer:
                    st.markdown("#### Request Metrics")
                    c1, c2 = st.columns(2)
                    c1.metric("Latency", f"{metrics.get('latency_seconds', 0):.2f}s")
                    c2.metric("Self-healing retries", metrics.get("retries", 0))
                    st.caption(f"Model: `{metrics.get('model', '—')}` | Provider: `{metrics.get('provider', '—')}`")
                    st.divider()
                    st.markdown("#### Raw JSON Payload")
                    st.json(data)
