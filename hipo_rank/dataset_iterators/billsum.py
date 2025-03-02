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
        
        # Load dataset using datasets library
        dataset = load_dataset(dataset_name)
        raw_docs = dataset[split]
        
        # Convert to our document format
        docs = []
        for i, item in enumerate(raw_docs):
            # Split summary into sentences
            summary_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', item['summary']) if s.strip()]
            
            docs.append(BillsumDoc(
                text=item['text'],
                summary=summary_sentences,
                title=item['title'],
                id=str(i)
            ))
        
        if min_words or max_words:
            docs = self._filter_doc_len(docs, min_words, max_words)
            
        self.docs = docs

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
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', doc.text) if s.strip()]
            sentences = [s for s in sentences if len([w for w in s.split() if w.isalpha()]) >= self.min_sent_len]
            return [Section(id="no_sections", sentences=sentences)]
        
        # Find all section markers and their positions
        section_pattern = r'(?:SECTION|SEC\.)\s+(\d+)\.'
        matches = list(re.finditer(section_pattern, doc.text, re.IGNORECASE))
        
        # If no sections found, return whole text as one section
        if not matches:
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', doc.text) if s.strip()]
            sentences = [s for s in sentences if len([w for w in s.split() if w.isalpha()]) >= self.min_sent_len]
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
            
            # Extract section text and split into sentences
            section_text = doc.text[start_idx:end_idx].strip()
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', section_text) if s.strip()]
            sentences = [s for s in sentences if len([w for w in s.split() if w.isalpha()]) >= self.min_sent_len]
            
            if sentences:  # Only add non-empty sections
                sections.append(Section(id=f"Section {section_num}", sentences=sentences))
        
        return sections
    
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