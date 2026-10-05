import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

def create_invoice(filename, vendor, date, tax_id, items, subtotal, tax, total):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Header
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height - 50, "INVOICE")
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 90, vendor)
    
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 110, f"Tax ID: {tax_id}")
    c.drawString(450, height - 90, f"Date: {date}")
    
    # Table Header
    y = height - 160
    c.setFillColor(colors.lightgrey)
    c.rect(50, y-5, 500, 20, fill=1)
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(60, y, "Description")
    c.drawString(300, y, "Qty")
    c.drawString(380, y, "Unit Price")
    c.drawString(480, y, "Amount")
    
    # Table Items
    y -= 25
    c.setFont("Helvetica", 10)
    for item in items:
        c.drawString(60, y, item["desc"])
        c.drawString(300, y, str(item["qty"]))
        c.drawString(380, y, f"${item['price']:.2f}")
        c.drawString(480, y, f"${item['amount']:.2f}")
        y -= 20
        
    # Totals
    y -= 30
    c.line(300, y+15, 550, y+15)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(380, y, "Subtotal:")
    c.drawString(480, y, f"${subtotal:.2f}")
    
    y -= 20
    c.drawString(380, y, "Tax (10%):")
    c.drawString(480, y, f"${tax:.2f}")
    
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(380, y, "Total Amount:")
    c.drawString(480, y, f"${total:.2f}")
    
    # Footer
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, 50, "Thank you for your business!")
    
    c.save()

def main():
    out_dir = "eval/test_invoices"
    os.makedirs(out_dir, exist_ok=True)
    
    # Invoice 1 (Matches PO-2024-001)
    create_invoice(
        f"{out_dir}/invoice_001.pdf",
        vendor="Acme Supplies Ltd",
        date="2024-11-15",
        tax_id="US12-3456789",
        items=[
            {"desc": "Office Chairs", "qty": 2, "price": 150.00, "amount": 300.00},
            {"desc": "Standing Desks", "qty": 3, "price": 400.00, "amount": 1200.00}
        ],
        subtotal=1500.00,
        tax=150.00,
        total=1650.00
    )
    
    # Invoice 2 (Matches PO-2024-002)
    create_invoice(
        f"{out_dir}/invoice_002.pdf",
        vendor="TechWorld Electronics",
        date="2024-10-03",
        tax_id="US98-7654321",
        items=[
            {"desc": "Laptop", "qty": 5, "price": 1200.00, "amount": 6000.00},
            {"desc": "USB Hubs", "qty": 10, "price": 25.00, "amount": 250.00},
            {"desc": "HDMI Cables", "qty": 20, "price": 10.00, "amount": 200.00}
        ],
        subtotal=6450.00,
        tax=645.00,
        total=7095.00
    )
    print("Generated 2 test PDFs successfully!")

if __name__ == "__main__":
    main()
