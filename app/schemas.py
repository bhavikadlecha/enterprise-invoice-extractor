"""
schemas.py
----------
Strict Pydantic v2 schema for invoice data extraction.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class LineItem(BaseModel):
    description: str = Field(..., description="Description of the product or service.")
    quantity: Decimal = Field(..., gt=0, description="Number of units (must be > 0).")
    unit_price: Decimal = Field(..., ge=0, description="Price per unit.")
    amount: Decimal = Field(..., ge=0, description="Total for this line: quantity × unit_price.")


class InvoiceSchema(BaseModel):
    vendor_name: str = Field(..., alias="VendorName")
    invoice_date: date = Field(..., alias="InvoiceDate")
    tax_id: Optional[str] = Field(None, alias="TaxID")
    line_items: list[LineItem] = Field(..., alias="LineItems", min_length=1)
    
    subtotal: Decimal = Field(..., alias="Subtotal", ge=0)
    tax: Decimal = Field(..., alias="Tax", ge=0)
    
    # 🆕 Flexible real-world fields (Optional, default to 0.00)
    shipping_fee: Decimal = Field(Decimal("0.00"), alias="ShippingFee", description="Any shipping, handling, or freight charges.")
    discount: Decimal = Field(Decimal("0.00"), alias="Discount", description="Any total discounts applied before final total.")
    
    total_amount: Decimal = Field(..., alias="TotalAmount", ge=0)

    model_config = {
        "populate_by_name": True,
    }

    @model_validator(mode="after")
    def check_totals(self) -> "InvoiceSchema":
        """
        Enforce the accounting invariant: 
        Subtotal + Tax + ShippingFee - Discount == TotalAmount.
        """
        tolerance = Decimal("0.02")
        expected = self.subtotal + self.tax + self.shipping_fee - self.discount
        
        if abs(expected - self.total_amount) > tolerance:
            raise ValueError(
                f"Accounting check failed: Subtotal ({self.subtotal}) + Tax ({self.tax}) "
                f"+ Shipping ({self.shipping_fee}) - Discount ({self.discount}) = {expected}, "
                f"but TotalAmount is {self.total_amount}. "
                "Please re-read the invoice and correct the values."
            )
        return self
