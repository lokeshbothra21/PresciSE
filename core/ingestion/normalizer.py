from core.ingestion.document_schema import Document, Section, Figure


def normalize_docling_output(docling_result, document_id: str) -> Document:
    """
    Normalizes Docling conversion output to PresciSE's canonical Document format.
    """
    sections = []
    figures = []

    # docling_result is the DoclingDocument object
    doc = docling_result

    # Iterate over document items using the proper docling API
    for item, level in doc.iterate_items():
        # Get the item label/type
        item_label = item.label if hasattr(item, 'label') else None
        
        # Text items (paragraphs, headings, etc.)
        if hasattr(item, 'text') and item.text:
            text = item.text.strip()
            if text:
                sections.append(
                    Section(
                        section_type=str(item_label) if item_label else "text",
                        content=text,
                        page_numbers=[item.prov[0].page_no if item.prov and len(item.prov) > 0 else -1],
                    )
                )

        # Figure/Image items
        if item_label and 'figure' in str(item_label).lower():
            caption = getattr(item, 'caption', {})
            caption_text = caption.get('text', '') if isinstance(caption, dict) else str(caption) if caption else ''
            figures.append(
                Figure(
                    caption=caption_text,
                    page_number=item.prov[0].page_no if hasattr(item, 'prov') and item.prov and len(item.prov) > 0 else -1,
                )
            )

    return Document(
        document_id=document_id,
        title=getattr(doc, "name", None),
        sections=sections,
        figures=figures,
        metadata={"source": "docling"},
    )
