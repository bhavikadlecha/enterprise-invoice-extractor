"""
app/rag/po_store.py
-------------------
ChromaDB-backed Purchase Order (PO) vector store.

Architecture:
  - Each PO is embedded as a text string combining vendor name + line item
    descriptions + PO number. This gives the semantic search context to
    match a new invoice against the right PO.
  - We use `sentence-transformers` (all-MiniLM-L6-v2) as the local embedding
    model — no external embedding API calls needed, zero cost.
  - ChromaDB persists its data to disk (CHROMA_PERSIST_DIR) so the PO
    catalogue survives server restarts.

Seeding:
  - `seed_mock_pos()` is called at FastAPI startup. It checks if the
    collection already has data before inserting, making it idempotent.
"""

import logging
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from app.config import settings

logger = logging.getLogger(__name__)

# Collection name inside ChromaDB
_COLLECTION_NAME = "purchase_orders"

# ---------------------------------------------------------------------------
# Mock PO dataset — in production this would come from your ERP / database
# ---------------------------------------------------------------------------
MOCK_PURCHASE_ORDERS: list[dict[str, Any]] = [
    {
        "po_number": "PO-2024-001",
        "vendor_name": "Acme Supplies Ltd",
        "items": ["Office Chairs", "Standing Desks"],
        "total_amount": 4500.00,
        "currency": "USD",
    },
    {
        "po_number": "PO-2024-002",
        "vendor_name": "TechWorld Electronics",
        "items": ["Laptop x5", "USB Hubs x10", "HDMI Cables x20"],
        "total_amount": 12350.00,
        "currency": "USD",
    },
    {
        "po_number": "PO-2024-003",
        "vendor_name": "CloudHost Services",
        "items": ["Cloud Server Subscription", "Support Plan"],
        "total_amount": 890.00,
        "currency": "USD",
    },
    {
        "po_number": "PO-2024-004",
        "vendor_name": "PrintPro Solutions",
        "items": ["A4 Paper Reams x100", "Printer Ink Cartridges x20", "Staples x50"],
        "total_amount": 650.00,
        "currency": "USD",
    },
    {
        "po_number": "PO-2024-005",
        "vendor_name": "Catering Plus",
        "items": ["Team Lunch Catering", "Coffee Machine Rental"],
        "total_amount": 1200.00,
        "currency": "USD",
    },
]


def _build_po_text(po: dict[str, Any]) -> str:
    """Convert a PO dict into a single searchable text string for embedding."""
    items_str = ", ".join(po["items"])
    return (
        f"Vendor: {po['vendor_name']}. "
        f"Items: {items_str}. "
        f"PO Number: {po['po_number']}. "
        f"Total: {po['total_amount']} {po['currency']}."
    )


def get_collection() -> chromadb.Collection:
    """
    Returns the ChromaDB PO collection (creates it if not present).
    Uses a local sentence-transformer embedding function.
    """
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )
    collection = client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def seed_mock_pos() -> None:
    """
    Seeds the ChromaDB collection with mock Purchase Orders.
    Safe to call multiple times — skips seeding if data already exists.
    """
    collection = get_collection()

    if collection.count() > 0:
        logger.info("PO store already seeded (%d records). Skipping.", collection.count())
        return

    logger.info("Seeding PO store with %d mock purchase orders…", len(MOCK_PURCHASE_ORDERS))

    collection.add(
        ids=[po["po_number"] for po in MOCK_PURCHASE_ORDERS],
        documents=[_build_po_text(po) for po in MOCK_PURCHASE_ORDERS],
        metadatas=[
            {
                "po_number": po["po_number"],
                "vendor_name": po["vendor_name"],
                "items": ", ".join(po["items"]),
                "total_amount": po["total_amount"],
                "currency": po["currency"],
            }
            for po in MOCK_PURCHASE_ORDERS
        ],
    )
    logger.info("PO store seeded successfully.")


def add_purchase_order(po_number: str, vendor_name: str, total_amount: float, items: str) -> None:
    """Dynamically embed and add a new Purchase Order to the Vector Database."""
    collection = get_collection()
    
    metadata = {
        "po_number": po_number,
        "vendor_name": vendor_name,
        "items": items,
        "total_amount": total_amount,
        "currency": "USD" # default or pass in
    }
    
    document_text = f"Vendor: {vendor_name}. Items: {items}. PO Number: {po_number}. Total: {total_amount}."
    
    collection.add(
        documents=[document_text],
        metadatas=[metadata],
        ids=[po_number]
    )
    logger.info("Successfully embedded and stored new PO: %s", po_number)
