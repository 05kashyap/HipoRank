from datasets import load_dataset
from typing import List, Iterator, Dict
from hipo_rank import Section, Document

class ILCDataset(object):
    def __init__(self, split="train"):
        """Initialize ILC dataset iterator.
        
        Args:
            split: Dataset split to use ("train", "validation", "test")
        """
        self.ds = load_dataset("d0r1h/ILC")[split]
        
    def _get_sections(self, item) -> List[Section]:
        case_text = item["Case"]
        
        # Split the case text into sentences
        sentences = [s.strip() for s in case_text.replace("\n", " ").split(". ") if s.strip()]
        sentences = [s + "." if not s.endswith(".") else s for s in sentences]
        
        # Create a single section with ID "Case"
        return [Section(id="Case", sentences=sentences)]
    
    def _get_reference(self, item) -> List[str]:
        # Return the gold summary as a list (similar to pubmed)
        return [item["Summary"]]
        
    def __iter__(self) -> Iterator[Document]:
        for item in self.ds:
            sections = self._get_sections(item)
            reference = self._get_reference(item)
            doc_id = item["Title"]
            # Store ID in meta dictionary instead of as a direct parameter
            yield Document(sections=sections, reference=reference, meta={"id": doc_id})
    
    def __getitem__(self, i):
        if isinstance(i, int):
            item = self.ds[i]
            sections = self._get_sections(item)
            reference = self._get_reference(item)
            doc_id = item["Title"]
            # Store ID in meta dictionary here too
            return Document(sections=sections, reference=reference, meta={"id": doc_id})
        elif isinstance(i, slice):
            items = self.ds[i]
            documents = []
            for item in items:
                sections = self._get_sections(item)
                reference = self._get_reference(item)
                doc_id = item["Title"]
                # Store ID in meta dictionary for each document
                documents.append(Document(sections=sections, reference=reference, meta={"id": doc_id}))
            return documents
    
    def __len__(self):
        return len(self.ds)