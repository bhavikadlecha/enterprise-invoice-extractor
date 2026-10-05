"""
eval/eval.py
------------
Automated Evaluation Benchmark for the Invoice Extractor.

How it works:
  1. Looks for PDF files in eval/test_invoices/ paired with JSON ground-truth
     files in eval/ground_truth/ (matched by filename stem).
  2. Calls the running FastAPI server's /extract-invoice endpoint for each PDF.
  3. Compares extracted fields against ground truth and calculates:
       - Per-field Exact Match Accuracy
       - Math Check (Subtotal + Tax == Total) pass rate
       - PO Match rate
       - Average latency and cost per invoice
  4. Prints a formatted report table to the terminal.

Usage:
  # Start the API first:
  uvicorn app.main:app --port 8000

  # Then run:
  python eval/eval.py
"""

import json
import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import requests
from tabulate import tabulate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_URL = os.getenv("EVAL_API_URL", "http://localhost:8000/extract-invoice")
GROUND_TRUTH_DIR = Path(__file__).parent / "ground_truth"
TEST_INVOICES_DIR = Path(__file__).parent / "test_invoices"

# Fields to evaluate (using API alias names)
SCALAR_FIELDS = ["VendorName", "InvoiceDate", "TaxID", "Subtotal", "Tax", "TotalAmount"]


# ---------------------------------------------------------------------------
# Field comparison helpers
# ---------------------------------------------------------------------------
def _normalise(value: Any) -> str:
    """Normalise a field value to a lowercase stripped string for comparison."""
    if value is None:
        return ""
    return str(value).strip().lower()


def _fields_match(extracted: Any, ground_truth: Any, field: str) -> bool:
    """Compare a single field with type-appropriate logic."""
    if field in ("Subtotal", "Tax", "TotalAmount"):
        try:
            return abs(Decimal(str(extracted)) - Decimal(str(ground_truth))) <= Decimal("0.02")
        except Exception:
            return False
    return _normalise(extracted) == _normalise(ground_truth)


def _line_items_match(extracted_items: list, gt_items: list) -> float:
    """
    Returns a 0.0–1.0 score for line item accuracy.
    Uses best-effort matching on description substring.
    """
    if not gt_items:
        return 1.0
    matched = 0
    for gt_item in gt_items:
        for ex_item in extracted_items:
            if _normalise(gt_item["description"]) in _normalise(ex_item.get("description", "")):
                matched += 1
                break
    return matched / len(gt_items)


# ---------------------------------------------------------------------------
# Single invoice evaluation
# ---------------------------------------------------------------------------
def evaluate_invoice(pdf_path: Path, ground_truth: dict) -> dict:
    """
    Call the API, compare results against ground truth, return a result dict.
    """
    result = {
        "file": pdf_path.name,
        "api_status": "N/A",
        "math_check": False,
        "po_matched": False,
        "latency_s": 0.0,
        "cost_usd": 0.0,
        "retries": 0,
        "field_scores": {},
        "line_item_score": 0.0,
        "overall_accuracy": 0.0,
        "error": None,
    }

    try:
        with open(pdf_path, "rb") as f:
            response = requests.post(
                API_URL, files={"file": (pdf_path.name, f, "application/pdf")}, timeout=120
            )

        result["api_status"] = response.status_code

        if response.status_code != 200:
            result["error"] = response.text[:300]
            return result

        data = response.json()
        invoice = data.get("invoice", {})
        metrics = data.get("metrics", {})
        po_match = data.get("po_match", {})

        # Observability metrics
        result["latency_s"] = metrics.get("latency_seconds", 0)
        result["cost_usd"] = metrics.get("cost_usd", 0)
        result["retries"] = metrics.get("retries", 0)
        result["po_matched"] = po_match.get("matched", False)

        # Math check
        try:
            sub = Decimal(str(invoice.get("Subtotal", 0)))
            tax = Decimal(str(invoice.get("Tax", 0)))
            ship = Decimal(str(invoice.get("ShippingFee", 0)))
            disc = Decimal(str(invoice.get("Discount", 0)))
            total = Decimal(str(invoice.get("TotalAmount", 0)))
            result["math_check"] = abs(sub + tax + ship - disc - total) <= Decimal("0.02")
        except Exception:
            result["math_check"] = False

        # Scalar field accuracy
        field_hits = 0
        for field in SCALAR_FIELDS:
            match = _fields_match(invoice.get(field), ground_truth.get(field), field)
            result["field_scores"][field] = "✅" if match else "❌"
            if match:
                field_hits += 1

        # Line item accuracy
        result["line_item_score"] = _line_items_match(
            invoice.get("LineItems", []), ground_truth.get("LineItems", [])
        )

        # Overall accuracy: weighted average of field hits + line item score
        total_fields = len(SCALAR_FIELDS) + 1  # +1 for line items
        result["overall_accuracy"] = round(
            (field_hits + result["line_item_score"]) / total_fields * 100, 1
        )

    except requests.exceptions.ConnectionError:
        result["error"] = "❌ Cannot connect to API. Is the server running?"
    except Exception as exc:
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n" + "=" * 70)
    print("  Vision-RAG Invoice Extractor — Evaluation Benchmark")
    print("=" * 70)

    # Discover PDF / ground-truth pairs
    gt_files = {p.stem: p for p in GROUND_TRUTH_DIR.glob("*.json")}
    pdf_files = {p.stem: p for p in TEST_INVOICES_DIR.glob("*.pdf")} if TEST_INVOICES_DIR.exists() else {}

    pairs = [(stem, pdf_files[stem], gt_files[stem]) for stem in gt_files if stem in pdf_files]

    if not pairs:
        print(
            "\n⚠️  No paired PDF + ground-truth files found.\n"
            f"  → Put PDF invoices in: {TEST_INVOICES_DIR}\n"
            f"  → Matching JSON ground truths in: {GROUND_TRUTH_DIR}\n"
            "  → Files must share the same stem (e.g. invoice_001.pdf ↔ invoice_001.json)\n"
        )
        print(
            "  Ground truth files available:",
            ", ".join(gt_files.keys()) or "none",
        )
        sys.exit(0)

    all_results = []
    for stem, pdf_path, gt_path in pairs:
        with open(gt_path) as f:
            ground_truth = json.load(f)
        print(f"\n  🔍 Evaluating: {pdf_path.name} …")
        res = evaluate_invoice(pdf_path, ground_truth)
        all_results.append(res)

    # --- Summary table ---
    table_rows = []
    for r in all_results:
        row = [
            r["file"],
            r["api_status"],
            "✅" if r["math_check"] else "❌",
            "✅" if r["po_matched"] else "❌",
            f"{r['overall_accuracy']}%",
            f"{r['latency_s']:.2f}s",
            f"${r['cost_usd']:.5f}",
            r["retries"],
            r.get("error") or "—",
        ]
        table_rows.append(row)

    print("\n\n" + tabulate(
        table_rows,
        headers=["File", "HTTP", "Math ✓", "PO Match", "Accuracy", "Latency", "Cost", "Retries", "Error"],
        tablefmt="rounded_outline",
    ))

    # --- Per-field breakdown ---
    print("\n\n  📊 Per-Field Accuracy Breakdown\n")
    field_table = []
    for r in all_results:
        if r["field_scores"]:
            row = [r["file"]] + [r["field_scores"].get(f, "—") for f in SCALAR_FIELDS]
            row.append(f"{r['line_item_score']*100:.0f}%")
            field_table.append(row)

    print(tabulate(
        field_table,
        headers=["File"] + SCALAR_FIELDS + ["LineItems"],
        tablefmt="rounded_outline",
    ))

    # --- Aggregate stats ---
    if all_results:
        successful = [r for r in all_results if r["api_status"] == 200]
        avg_acc = sum(r["overall_accuracy"] for r in successful) / max(len(successful), 1)
        avg_lat = sum(r["latency_s"] for r in successful) / max(len(successful), 1)
        total_cost = sum(r["cost_usd"] for r in successful)
        math_pass_rate = sum(1 for r in successful if r["math_check"]) / max(len(successful), 1) * 100

        print(f"""
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📈 AGGREGATE RESULTS ({len(successful)}/{len(all_results)} succeeded)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Average Accuracy   : {avg_acc:.1f}%
  Math Check Pass    : {math_pass_rate:.1f}%
  Avg Latency        : {avg_lat:.2f}s
  Total API Cost     : ${total_cost:.5f}
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


if __name__ == "__main__":
    main()
