from datasets import load_dataset
from typing import List, Iterator, Any, Optional
from dataclasses import dataclass
import re
from hipo_rank import Document, Section


@dataclass
class BillsumDoc:
    # dataclass wrapper for original billsum dataset format
    text: str
    summary: List[str]
    title: str
    id: Optional[str] = None


class BillsumDataset(object):
    def __init__(self,
                 split: str = "train",
                 dataset_name: str = "FiscalNote/billsum",
                 no_sections: bool = False,
                 min_words: Optional[int] = None,
                 max_words: Optional[int] = None,
                 min_sent_len: int = 1  # min num of alphabetical words
                 ):
        self.no_sections = no_sections
        self.min_sent_len = min_sent_len
        
        # Common abbreviations that shouldn't split sentences
        self.abbreviations = [
            "Stat.", "U.S.", "U.S.C.", "USC.", "et al.", "i.e.", "e.g.", 
            "etc.", "cf.", "v.", "vs.", "fig.", "Fig.", "No.", "no.", 
            "St.", "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Inc.", "Corp.",
            "Ltd.", "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.",
            "Sep.", "Sept.", "Oct.", "Nov.", "Dec."
        ]
        
        # Load dataset using datasets library
        dataset = load_dataset(dataset_name)
        raw_docs = dataset[split]
        
        # Convert to our document format
        docs = []
        for i, item in enumerate(raw_docs):
            # Clean and normalize text
            cleaned_text = self._clean_text(item['text'])
            
            # Split summary into sentences
            summary_sentences = self._text_to_sentences(item['summary'])
            
            docs.append(BillsumDoc(
                text=cleaned_text,
                summary=summary_sentences,
                title=item['title'],
                id=str(i)
            ))
        
        if min_words or max_words:
            docs = self._filter_doc_len(docs, min_words, max_words)
            
        self.docs = docs

    def _clean_text(self, text: str) -> str:
        """Clean text by removing excessive whitespace and normalizing newlines"""
        # Replace multiple whitespace characters (including newlines) with a single space
        text = re.sub(r'\s+', ' ', text)
        # Remove any trailing/leading whitespace
        return text.strip()

    def _filter_doc_len(self, docs: List[BillsumDoc], min_words: int, max_words: int):
        def f(doc: BillsumDoc):
            l = len(doc.text.split())
            if min_words and l < min_words:
                return False
            if max_words and l >= max_words:
                return False
            return True
        return list(filter(f, docs))
    
    def _get_sections(self, doc: BillsumDoc) -> List[Section]:
        if self.no_sections:
            # Treat the entire document as a single section
            sentences = self._text_to_sentences(doc.text)
            return [Section(id="no_sections", sentences=sentences)]
        
        # Find all section markers and their positions
        section_pattern = r'(?:SECTION|SEC\.)\s+(\d+)\.'
        matches = list(re.finditer(section_pattern, doc.text, re.IGNORECASE))
        
        # If no sections found, return whole text as one section
        if not matches:
            sentences = self._text_to_sentences(doc.text)
            return [Section(id="full_text", sentences=sentences)]
        
        # Process each section
        sections = []
        for i, match in enumerate(matches):
            section_num = match.group(1)
            start_idx = match.end()
            
            # Find the end of this section (start of next section or end of text)
            if i < len(matches) - 1:
                end_idx = matches[i+1].start()
            else:
                end_idx = len(doc.text)
            
            # Extract section text
            section_text = doc.text[start_idx:end_idx].strip()
            
            # Skip the heading by finding the first period that's not part of the heading
            # Look for a pattern like "SHORT TITLE." or "DEFINITIONS." at the beginning
            heading_match = re.match(r'^([^.]+)\.', section_text)
            sentences_text = section_text
            heading = None
            
            if heading_match:
                heading = heading_match.group(1).strip()
                # Skip the heading when extracting sentences
                sentences_text = section_text[len(heading_match.group(0)):].strip()
            
            sentences = self._text_to_sentences(sentences_text)
            
            if sentences:  # Only add non-empty sections
                section_id = f"Section {section_num}"
                if heading:
                    section_id += f": {heading}"
                sections.append(Section(id=section_id, sentences=sentences))
        
        return sections
    
    def _text_to_sentences(self, text: str) -> List[str]:
        """Split text into clean sentences with improved handling for abbreviations"""
        if not text:
            return []
        
        # Temporarily replace known abbreviations to prevent incorrect splitting
        placeholder_map = {}
        for i, abbr in enumerate(self.abbreviations):
            placeholder = f"___ABBR{i}___"
            placeholder_map[placeholder] = abbr
            text = text.replace(abbr, placeholder)
        
        # Split into sentences using period-space pattern
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Restore abbreviations
        cleaned_sentences = []
        for s in raw_sentences:
            if s.strip():
                for placeholder, abbr in placeholder_map.items():
                    s = s.replace(placeholder, abbr)
                cleaned_sentences.append(self._clean_text(s))
        
        # Filter by minimum length requirement
        return [s for s in cleaned_sentences if len([w for w in s.split() if w.isalpha()]) >= self.min_sent_len]
    
    def _get_reference(self, doc: BillsumDoc) -> List[str]:
        return doc.summary
    
    def __iter__(self) -> Iterator[Document]:
        for doc in self.docs:
            sections = self._get_sections(doc)
            reference = self._get_reference(doc)
            yield Document(sections=sections, reference=reference)
            
    def __getitem__(self, i):
        if isinstance(i, int):
            doc = self.docs[i]
            sections = self._get_sections(doc)
            reference = self._get_reference(doc)
            return Document(sections=sections, reference=reference)
        elif isinstance(i, slice):
            docs = self.docs[i]
            sections = [self._get_sections(doc) for doc in docs]
            references = [self._get_reference(doc) for doc in docs]
            return [Document(sections=s, reference=r) for s, r in zip(sections, references)]