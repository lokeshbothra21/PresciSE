import logging
import sys
import os

# Configure logging to suppress output from docling and its dependencies
# Use a NullHandler to prevent "No handlers found" warnings
logging.basicConfig(
    level=logging.CRITICAL,
    handlers=[logging.NullHandler()],
    force=True
)

# Set all relevant loggers to CRITICAL to suppress their output
for logger_name in ["docling", "pdfminer", "pypdf", "pypdfium2", "rapidocr", "RapidOCR"]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)
    logger.addHandler(logging.NullHandler())
    logger.propagate = False

from docling.document_converter import DocumentConverter
from docling.exceptions import ConversionError


def load_with_docling(pdf_path: str):
    """
    Extracts structured content from a PDF using Docling.
    Returns Docling conversion result object.
    """
    # Suppress output at OS level to prevent pypdfium2 from printing raw PDF debug output
    # This is necessary because pypdfium2 writes directly to file descriptors
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    
    with open(os.devnull, 'w') as devnull:
        old_stdout_fd = os.dup(stdout_fd)
        old_stderr_fd = os.dup(stderr_fd)
        os.dup2(devnull.fileno(), stdout_fd)
        os.dup2(devnull.fileno(), stderr_fd)
        conversion_error = None
        result = None
        try:
            converter = DocumentConverter()
            result = converter.convert(pdf_path)
        except ConversionError as e:
            # Capture the error but don't let it print with raw PDF data
            conversion_error = e
        finally:
            # Restore stdout and stderr
            os.dup2(old_stdout_fd, stdout_fd)
            os.dup2(old_stderr_fd, stderr_fd)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
    
    # Re-raise with clean message if there was an error
    if conversion_error:
        import os as os_module
        filename = os_module.path.basename(pdf_path)
        raise ConversionError(f"Failed to convert PDF: {filename}. The PDF may be malformed or missing required metadata.") from None
    
    return result.document
