"""
Debug script that writes output to file.
"""

import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path

# Write all output to file
output_file = Path("debug_output.txt")

with open(output_file, "w") as f:
    f.write("PDF PROCESSING PIPELINE DEBUG\n")
    f.write("="*80 + "\n\n")
    
    # Step 1: Find PDFs
    pdf_dir = Path("data/pdfs")
    pdfs = list(pdf_dir.glob("*.pdf"))
    f.write(f"Found {len(pdfs)} PDFs:\n")
    for pdf in pdfs:
        f.write(f"  - {pdf.name} ({pdf.stat().st_size / 1024:.1f} KB)\n")
    
    # Step 2: Load first PDF
    f.write("\n" + "-"*80 + "\n")
    f.write("STEP 2: Loading first PDF...\n")
    
    try:
        from core.ingestion.pdf_loader import load_pdf_as_document
        doc = load_pdf_as_document(str(pdfs[0]))
        f.write(f"Document loaded: type={type(doc).__name__}\n")
        
        if isinstance(doc, dict):
            sections = doc.get('sections', [])
            doc_id = doc.get('document_id', 'unknown')
        else:
            sections = getattr(doc, 'sections', [])
            doc_id = getattr(doc, 'document_id', 'unknown')
        
        f.write(f"Document ID: {doc_id}\n")
        f.write(f"Number of sections: {len(sections)}\n")
        
        total_content = 0
        for i, sec in enumerate(sections):
            if isinstance(sec, dict):
                sec_type = sec.get('section_type', 'unknown')
                content = sec.get('content', '')
            else:
                sec_type = getattr(sec, 'section_type', 'unknown')
                content = getattr(sec, 'content', '')
            content_len = len(content or '')
            total_content += content_len
            f.write(f"  Section {i}: {sec_type} - {content_len} chars\n")
        
        f.write(f"\nTotal content chars: {total_content}\n")
        
    except Exception as e:
        f.write(f"ERROR loading PDF: {e}\n")
        import traceback
        f.write(traceback.format_exc())
    
    # Step 3: Chunk
    f.write("\n" + "-"*80 + "\n")
    f.write("STEP 3: Chunking...\n")
    
    try:
        from core.chunking.pdf_chunker import make_chunks_from_doc
        chunks = make_chunks_from_doc(doc)
        f.write(f"Generated {len(chunks)} chunks\n")
        
        if len(chunks) == 0:
            f.write("WARNING: Zero chunks! This causes BM25 error.\n")
        else:
            f.write(f"First chunk: {str(chunks[0])[:200]}...\n")
            
    except Exception as e:
        f.write(f"ERROR chunking: {e}\n")
        import traceback
        f.write(traceback.format_exc())
    
    f.write("\n" + "="*80 + "\n")
    f.write("DONE\n")

print(f"Debug output written to {output_file}")
