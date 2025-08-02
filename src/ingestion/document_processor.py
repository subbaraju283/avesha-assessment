"""
Document Processor for extracting and parsing content from various file formats.
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional
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
    
    def parse_content(self, content: str, file_extension: str) -> Dict[str, Any]:
        """
        Parse content into structured data for different subsystems.
        
        Args:
            content: Raw content from file
            file_extension: File extension to determine parsing strategy
            
        Returns:
            Dictionary with parsed data for KB, KG, and RAG
        """
        parsed_data = {
            "content": content,
            "facts": [],
            "entities": [],
            "relationships": []
        }
        
        try:
            if file_extension == ".json":
                # JSON files might contain structured data
                parsed_data.update(self._parse_json_content(content))
            elif file_extension == ".md":
                # Markdown files might contain structured sections
                parsed_data.update(self._parse_markdown_content(content))
            else:
                # Generic text parsing
                parsed_data.update(self._parse_generic_content(content))
            
        except Exception as e:
            logger.error(f"Content parsing error: {e}")
        
        return parsed_data
    
    def _parse_json_content(self, content: str) -> Dict[str, Any]:
        """Parse JSON content for structured data."""
        try:
            data = json.loads(content)
            facts = []
            entities = []
            relationships = []
            
            # Extract facts from JSON
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (str, int, float)):
                        facts.append({
                            "subject": key,
                            "predicate": "has_value",
                            "object": str(value),
                            "confidence": 0.9
                        })
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            facts.append({
                                "subject": key,
                                "predicate": sub_key,
                                "object": str(sub_value),
                                "confidence": 0.8
                            })
            
            return {
                "facts": facts,
                "entities": entities,
                "relationships": relationships
            }
            
        except Exception as e:
            logger.error(f"JSON parsing error: {e}")
            return {}
    
    def _parse_markdown_content(self, content: str) -> Dict[str, Any]:
        """Parse Markdown content for structured data."""
        facts = []
        entities = []
        relationships = []
        
        # Extract headers as entities
        header_pattern = r'^(#{1,6})\s+(.+)$'
        for line in content.split('\n'):
            match = re.match(header_pattern, line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                entities.append({
                    "name": title,
                    "type": f"header_level_{level}",
                    "properties": {"level": level}
                })
        
        # Extract links as relationships
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        for match in re.finditer(link_pattern, content):
            text = match.group(1)
            url = match.group(2)
            relationships.append({
                "source_id": "document",
                "target_id": url,
                "type": "LINKS_TO",
                "properties": {"link_text": text}
            })
        
        return {
            "facts": facts,
            "entities": entities,
            "relationships": relationships
        }
    
    def _parse_generic_content(self, content: str) -> Dict[str, Any]:
        """Parse generic text content for structured data."""
        facts = []
        entities = []
        relationships = []
        
        # Extract potential entities (capitalized phrases)
        entity_pattern = r'\b[A-Z][a-zA-Z\s]{2,}\b'
        potential_entities = re.findall(entity_pattern, content)
        
        # Filter and create entities
        for entity in set(potential_entities):
            if len(entity.strip()) > 3:  # Filter out short matches
                entities.append({
                    "name": entity.strip(),
                    "type": "mentioned_entity",
                    "properties": {"source": "text_extraction"}
                })
        
        # Extract potential facts (subject-verb-object patterns)
        # This is a simplified approach
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Only process substantial sentences
                # Look for patterns like "X is Y" or "X has Y"
                fact_patterns = [
                    r'(\w+)\s+is\s+(\w+)',
                    r'(\w+)\s+has\s+(\w+)',
                    r'(\w+)\s+was\s+(\w+)'
                ]
                
                for pattern in fact_patterns:
                    match = re.search(pattern, sentence, re.IGNORECASE)
                    if match:
                        facts.append({
                            "subject": match.group(1),
                            "predicate": "is" if "is" in pattern else "has",
                            "object": match.group(2),
                            "confidence": 0.6
                        })
        
        return {
            "facts": facts,
            "entities": entities,
            "relationships": relationships
        }
    
    def extract_nasa_specific_data(self, content: str) -> Dict[str, Any]:
        """Extract NASA-specific data from content."""
        nasa_data = {
            "missions": [],
            "spacecraft": [],
            "planets": [],
            "technologies": [],
            "dates": [],
            "locations": []
        }
        
        # Extract mission names
        mission_patterns = [
            r'\b[A-Z][a-zA-Z\s]+(?:Mission|Probe|Rover|Satellite)\b',
            r'\b(?:Apollo|Voyager|Curiosity|Perseverance|Hubble|Webb)\b'
        ]
        
        for pattern in mission_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            nasa_data["missions"].extend(matches)
        
        # Extract planet names
        planets = ["Mars", "Jupiter", "Saturn", "Venus", "Mercury", "Neptune", "Uranus", "Pluto"]
        for planet in planets:
            if planet.lower() in content.lower():
                nasa_data["planets"].append(planet)
        
        # Extract technologies
        tech_patterns = [
            r'\b(?:ion propulsion|solar panels|nuclear power|telescope|spectrometer)\b',
            r'\b(?:propulsion|power|communication|navigation)\s+(?:system|technology)\b'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            nasa_data["technologies"].extend(matches)
        
        # Extract dates
        date_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\b\d{4}\b'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, content)
            nasa_data["dates"].extend(matches)
        
        # Extract locations
        location_patterns = [
            r'\b(?:Kennedy Space Center|Jet Propulsion Laboratory|Johnson Space Center|Cape Canaveral)\b',
            r'\b(?:NASA|JPL|KSC|JSC)\b'
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            nasa_data["locations"].extend(matches)
        
        return nasa_data
    
    def validate_content(self, content: str) -> bool:
        """Validate that content is suitable for processing."""
        if not content or len(content.strip()) < 10:
            return False
        
        # Check for minimum meaningful content
        words = content.split()
        if len(words) < 5:
            return False
        
        # Check for reasonable character distribution
        if len(content) > 1000000:  # 1MB limit
            return False
        
        return True 