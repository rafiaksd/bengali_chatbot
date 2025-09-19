import pdfplumber

def extract_text_from_pdf(pdf_path):
    all_text = []  # List to store the text from all pages
    
    # Open the PDF file
    with pdfplumber.open(pdf_path) as pdf:
        # Loop through all the pages
        for page_num, page in enumerate(pdf.pages):
            print(f"Processing page {page_num + 1}...")
            
            # Extract text from the page
            text = page.extract_text()
            
            if text:
                all_text.append(text)  # Append text of this page to the list
    
    # Join all pages' text into a single string
    full_text = "\n".join(all_text)
    
    return full_text

# Path to the PDF file
pdf_path = "bengali_pdf/bengali_small.pdf"

# Extract the text from the PDF
extracted_text = extract_text_from_pdf(pdf_path)

# Save the extracted text
with open("extracted_bengali_text.pdfplumber.txt", "w", encoding="utf-8") as file:
    file.write(extracted_text)

print("PDF text extraction completed and saved to 'extracted_bengali_text.pdfplumber.txt'")
