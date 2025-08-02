"""
Document Processor for extracting and parsing content from various file formats.
"""

import logging
import json
import re
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processor for extracting and parsing document content."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def extract_content(self, file_path: Path) -> Optional[str]:
        """
        Extract content from a file based on its format.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Extracted content as string, or None if extraction failed
        """
        try:
            file_extension = file_path.suffix.lower()
            
            if file_extension == ".pdf":
                return self._extract_pdf_content(file_path)
            elif file_extension == ".json":
                return self._extract_json_content(file_path)
            elif file_extension == ".md":
                return self._extract_markdown_content(file_path)
            elif file_extension == ".txt":
                return self._extract_text_content(file_path)
            elif file_extension == ".docx":
                return self._extract_docx_content(file_path)
            else:
                logger.warning(f"Unsupported file format: {file_extension}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to extract content from {file_path}: {e}")
            return None
    
    def _extract_pdf_content(self, file_path: Path) -> str:
        """Extract text content from PDF file."""
        try:
            import pypdf
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except ImportError:
            logger.error("pypdf not installed, cannot extract PDF content")
            return ""
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return ""
    
    def _extract_json_content(self, file_path: Path) -> str:
        """Extract content from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                # Convert JSON to readable text
                return json.dumps(data, indent=2)
        except Exception as e:
            logger.error(f"JSON extraction error: {e}")
            return ""
    
    def _extract_markdown_content(self, file_path: Path) -> str:
        """Extract content from Markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Markdown extraction error: {e}")
            return ""
    
    def _extract_text_content(self, file_path: Path) -> str:
        """Extract content from text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Text extraction error: {e}")
            return ""
    
    def _extract_docx_content(self, file_path: Path) -> str:
        """Extract content from DOCX file."""
        try:
            from docx import Document
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except ImportError:
            logger.error("python-docx not installed, cannot extract DOCX content")
            return ""
        except Exception as e:
            logger.error(f"DOCX extraction error: {e}")
            return "" 