import fitz # PyMuPDF
import io

def extract_text_from_pdf_bytes(pdf_bytes):
    """
    Reads PDF bytes and extracts text content from all pages using PyMuPDF.

    Args:
        pdf_bytes: The PDF content as bytes.

    Returns:
        A string containing the concatenated text from the PDF,
        or None if the file cannot be opened or processed.
    """
    full_text = ""
    try:
        # Open PDF from bytes
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text += page.get_text("text") # Extract text content
        doc.close()
        if not full_text:
             print("Warning: No text extracted from PDF.")
             return None
        return full_text
    except Exception as e:
        print(f"Error processing PDF bytes with PyMuPDF: {e}")
        return None

def read_text_file_bytes(text_bytes):
     """Reads text file bytes and returns decoded string."""
     try:
          return text_bytes.decode('utf-8')
     except Exception as e:
          print(f"Error decoding text file bytes: {e}")
          return None

