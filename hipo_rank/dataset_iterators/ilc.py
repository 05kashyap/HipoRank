from datasets import load_dataset
from typing import List, Iterator, Any, Optional
from dataclasses import dataclass
import re
import spacy
from tqdm import tqdm
from hipo_rank import Document, Section


@dataclass
class ILCDoc:
    text: str  # The case text
    summary: List[str]  # The reference summary sentences
    title: str
    id: Optional[str] = None


class ILCDataset(object):
    def __init__(self,
                 split: str = "train",
                 dataset_name: str = "d0r1h/ILC",
                 no_sections: bool = False,
                 min_words: Optional[int] = None,
                 max_words: Optional[int] = None,
                 min_sent_len: int = 1,  # min num of alphabetical words
                 use_gpu: bool = True  # Control GPU usage
                 ):
        self.no_sections = no_sections
        self.min_sent_len = min_sent_len
        
        if use_gpu:
            gpu_available = spacy.prefer_gpu()
            print(f"GPU acceleration for spaCy: {'Enabled' if gpu_available else 'Not available'}")
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        self.nlp.disable_pipes(["tagger", "parser", "ner", "lemmatizer", "attribute_ruler"])
        # Enable sentence segmentation 
        self.nlp.enable_pipe("senter")
            
        # Load dataset using datasets library
        print(f"Loading {split} split from {dataset_name}...")
        dataset = load_dataset(dataset_name)
        raw_docs = dataset[split]
        
        # Convert to our document format
        print(f"Processing {len(raw_docs)} documents...")
        docs = []
        for i, item in enumerate(tqdm(raw_docs, desc="Preparing documents")):
            # Clean and normalize text
            cleaned_text = self._clean_text(item['Case'])
            
            # Split summary into sentences using spaCy
            summary_sentences = self._text_to_sentences(item['Summary'])
            
            docs.append(ILCDoc(
                text=cleaned_text,
                summary=summary_sentences,
                title=item['Title'],
                id=str(i)
            ))
        
        if min_words or max_words:
            docs = self._filter_doc_len(docs, min_words, max_words)
            
        self.docs = docs
        print(f"Loaded {len(self.docs)} documents")

    def _clean_text(self, text: str) -> str:
        """Clean text by removing excessive whitespace and normalizing newlines"""
        # Replace multiple whitespace characters (including newlines) with a single space
        text = re.sub(r'\s+', ' ', text)
        # Remove any trailing/leading whitespace
        return text.strip()

    def _filter_doc_len(self, docs: List[ILCDoc], min_words: int, max_words: int):
        def f(doc: ILCDoc):
            l = len(doc.text.split())
            if min_words and l < min_words:
                return False
            if max_words and l >= max_words:
                return False
            return True
        return list(filter(f, docs))
    
    def _get_sections(self, doc: ILCDoc) -> List[Section]:
        if self.no_sections:
            # Treat the entire document as a single section
            sentences = self._text_to_sentences(doc.text)
            return [Section(id="no_sections", sentences=sentences)]
        
        # Find all section markers and their positions for legal cases
        # Legal cases typically have sections like "FACTS", "JUDGMENT", "REASONING", etc.
        section_patterns = [
            r'([A-Z][A-Z\s]+):',  # ALL CAPS heading followed by colon
            r'(\d+\.\s+[A-Z][A-Za-z\s]+)',  # Numbered section with title
            r'([A-Z][A-Z\s]+)(?=\s)',  # ALL CAPS heading
        ]
        
        all_matches = []
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, doc.text))
            all_matches.extend(matches)
        
        # Sort matches by their position in text
        all_matches.sort(key=lambda m: m.start())
        
        # If no sections found, return whole text as one section
        if not all_matches:
            sentences = self._text_to_sentences(doc.text)
            return [Section(id="full_text", sentences=sentences)]
        
        # Process each section
        sections = []
        for i, match in enumerate(all_matches):
            section_title = match.group(1).strip()
            start_idx = match.end()
            
            # Find the end of this section (start of next section or end of text)
            if i < len(all_matches) - 1:
                end_idx = all_matches[i+1].start()
            else:
                end_idx = len(doc.text)
            
            # Extract section text
            section_text = doc.text[start_idx:end_idx].strip()
            
            sentences = self._text_to_sentences(section_text)
            
            if sentences:  # Only add non-empty sections
                section_id = f"Section: {section_title}"
                sections.append(Section(id=section_id, sentences=sentences))
        
        return sections
    
    def _text_to_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy's sentence segmentation"""
        if not text:
            return []
        
        # Process the text with spaCy
        doc = self.nlp(text)
        
        # Extract sentences as strings and clean them
        sentences = [self._clean_text(sent.text) for sent in doc.sents]
        
        # Filter by minimum length requirement and remove any empty sentences
        return [s for s in sentences if s and len([w for w in s.split() if w.isalpha()]) >= self.min_sent_len]

    def _get_reference(self, doc: ILCDoc) -> List[str]:
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