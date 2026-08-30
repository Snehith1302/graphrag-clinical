import os
from fpdf import FPDF

def generate_sample_pdf(output_path: str):
    """
    Generates a standard PDF with metadata to use as a test fixture.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    pdf = FPDF()
    pdf.add_page()
    
    # Set document metadata properties
    pdf.set_title("Metformin Clinical Guideline 2024")
    pdf.set_author("Alice Smith, Robert Jones")
    pdf.set_creator("WHO Publisher")
    pdf.set_subject("https://www.who.int/publications/metformin-guideline")
    
    # Write page content
    pdf.set_font("Helvetica", "B", size=16)
    pdf.cell(200, 10, txt="Metformin Clinical Guideline 2024", ln=1, align="C")
    pdf.ln(10)
    
    pdf.set_font("Helvetica", "B", size=12)
    pdf.cell(200, 10, txt="1. Indications and Clinical Usage", ln=1, align="L")
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 8, txt=(
        "Metformin is indicated as the first-line pharmacotherapy for type 2 diabetes mellitus. "
        "It helps lower blood glucose levels primarily by decreasing hepatic glucose production "
        "and increasing insulin sensitivity in peripheral tissues."
    ))
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", size=12)
    pdf.cell(200, 10, txt="2. Contraindications", ln=1, align="L")
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 8, txt=(
        "Metformin is contraindicated in patients with severe renal impairment, defined as an estimated "
        "glomerular filtration rate (eGFR) below 30 mL/min/1.73 m2. It should also be avoided in cases "
        "of acute metabolic acidosis, including diabetic ketoacidosis."
    ))
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", size=12)
    pdf.cell(200, 10, txt="3. Adverse Reactions and Side Effects", ln=1, align="L")
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 8, txt=(
        "Common adverse reactions include gastrointestinal side effects such as diarrhea, nausea, "
        "vomiting, flatulence, and abdominal discomfort. Rare but serious risks include lactic acidosis, "
        "especially in patients with renal dysfunction."
    ))
    
    pdf.output(output_path)
    print(f"Generated test PDF: {output_path}")

if __name__ == "__main__":
    generate_sample_pdf("tests/fixtures/sample.pdf")
