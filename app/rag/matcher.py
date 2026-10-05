"""
app/rag/matcher.py
------------------
Semantic PO matching: given an extracted invoice, find the best matching
Purchase Order from the ChromaDB vector store.

Matching Strategy:
  1. Build a query text from the extracted invoice (vendor name + line item
     descriptions). This mirrors how POs were embedded in po_store.py.
  2. Run a cosine-similarity vector search (top-k candidates).
  3. Re-rank candidates by also comparing the invoice's TotalAmount to the
     PO's expected total. Score = 0.7 * semantic_score + 0.3 * amount_score.
  4. Return the best match if its combined score exceeds the threshold (0.5).
"""

import logging
from typing import Optional

from pydantic import BaseModel

from app.config import settings
from app.rag.po_store import get_collection
from app.schemas import InvoiceSchema

logger = logging.getLogger(__name__)

# Minimum combined score for a PO match to be considered valid
_MATCH_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
class POMatchResult(BaseModel):
    """Result of a PO matching operation."""

    matched: bool
    po_number: Optional[str] = None
    vendor_name: Optional[str] = None
    po_items: Optional[str] = None
    po_total_amount: Optional[float] = None
    # 0.0 → no match, 1.0 → perfect match
    confidence_score: float = 0.0
    reason: str = "No matching PO found."


# ---------------------------------------------------------------------------
# Matching function
# ---------------------------------------------------------------------------
async def match_invoice_to_po(invoice: InvoiceSchema) -> POMatchResult:
    """
    Performs semantic + amount-based PO matching for a given invoice.

    Args:
        invoice: A fully validated InvoiceSchema instance.

    Returns:
        A POMatchResult with match details and a confidence score.
    """
    collection = get_collection()

    if collection.count() == 0:
        return POMatchResult(matched=False, reason="PO store is empty.")

    # --- Build query text ---
    item_descriptions = " ".join(item.description for item in invoice.line_items)
    query_text = f"Vendor: {invoice.vendor_name}. Items: {item_descriptions}."

    # --- Vector search ---
    results = collection.query(
        query_texts=[query_text],
        n_results=min(settings.rag_top_k, collection.count()),
        include=["metadatas", "distances"],
    )

    if not results["metadatas"] or not results["metadatas"][0]:
        return POMatchResult(matched=False, reason="Vector search returned no results.")

    # ChromaDB cosine distance: 0.0 = identical, 2.0 = opposite
    # Convert to similarity score (0.0 → 1.0)
    best_metadata = results["metadatas"][0][0]
    best_distance = results["distances"][0][0]
    semantic_score = max(0.0, 1.0 - (best_distance / 2.0))

    # --- Amount similarity score ---
    po_total = float(best_metadata.get("total_amount", 0))
    invoice_total = float(invoice.total_amount)
    amount_score = _amount_similarity(invoice_total, po_total)

    # --- Combined re-ranking score ---
    combined_score = round(0.7 * semantic_score + 0.3 * amount_score, 4)

    logger.info(
        "PO match: %s | semantic=%.3f | amount=%.3f | combined=%.3f",
        best_metadata["po_number"],
        semantic_score,
        amount_score,
        combined_score,
    )

    if combined_score < _MATCH_THRESHOLD:
        return POMatchResult(
            matched=False,
            confidence_score=combined_score,
            reason=f"Best candidate '{best_metadata['po_number']}' scored {combined_score:.2f}, "
                   f"below threshold {_MATCH_THRESHOLD}.",
        )

    return POMatchResult(
        matched=True,
        po_number=best_metadata["po_number"],
        vendor_name=best_metadata["vendor_name"],
        po_items=best_metadata["items"],
        po_total_amount=po_total,
        confidence_score=combined_score,
        reason=f"Matched with {combined_score:.0%} confidence.",
    )


def _amount_similarity(invoice_total: float, po_total: float) -> float:
    """
    Returns a 0.0–1.0 score for how close the invoice total is to the PO total.
    Uses a tolerance band: within 10% = full score, beyond 50% = zero score.
    """
    if po_total == 0:
        return 0.0
    ratio = abs(invoice_total - po_total) / po_total
    if ratio <= 0.10:
        return 1.0
    if ratio >= 0.50:
        return 0.0
    # Linear decay between 10% and 50%
    return round(1.0 - (ratio - 0.10) / 0.40, 4)
