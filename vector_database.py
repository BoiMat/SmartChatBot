"""
Vector database interface for storing and retrieving academic papers.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Tuple
import json
import os
import re
import torch
import unicodedata
from collections import defaultdict
from unidecode import unidecode
from typing import Dict, Any, List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from database_schema import Paper, BehaviorCategories, CommonDogBreeds
import logging
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
import nltk
nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


def split_into_sections(text):
    """
    Splits an academic paper's extracted text into sections.
    Recognizes the following headings:
      - ABSTRACT: text goes into the 'abstract' section.
      - INTRODUCTION, BACKGROUND, METHODS, RESULTS: text goes into 'body'.
      - DISCUSSION, CONCLUSION(S): text goes into 'conclusion'.
      - REFERENCES, BIBLIOGRAPHY: text goes into 'references'.
      
    Additionally, everything before the first recognized heading is stored in 'title'.
    If no ABSTRACT section is found, then the 'title' content is merged into 'body'.
    
    This version handles headings that might have extra text on the same line.
    
    Returns a dict with sections: 'title', 'abstract', 'body', 'conclusion', 'references'.
    """
    section_mapping = {
        "SIMPLE SUMMARY": "abstract",
        "ABSTRACT": "abstract",
        "INTRODUCTION": "body",
        "BACKGROUND": "body",
        "METHODS": "body",
        "RESULTS": "body",
        "DISCUSSION": "conclusion",
        "CONCLUSION": "conclusion",
        "CONCLUSIONS": "conclusion",
        "REFERENCES": "references",
        "BIBLIOGRAPHY": "references"
    }
    
    # Updated regex that captures extra text on the heading line.
    # Group 1 is the heading; Group 2 captures any extra text following an optional colon.
    pattern = re.compile(
        r'^\s*(' + "|".join(section_mapping.keys()) + r')\b\s*:?\s*(.*)$',
        re.IGNORECASE
    )
    
    sections = {
        'title': [],
        'abstract': [],
        'body': [],
        'conclusion': [],
        'references': []
    }
    
    # Start with "title" by default (everything before the first recognized heading).
    current_section = 'title'
    
    lines = text.splitlines()
    
    for line in lines:
        match = pattern.match(line)
        if match:
            heading = match.group(1).upper()
            extra_text = match.group(2).strip()
            current_section = section_mapping.get(heading, current_section)
            if extra_text:
                sections[current_section].append(extra_text)
            continue
        
        sections[current_section].append(line)
    
    for key in sections:
        sections[key] = "\n".join(sections[key]).strip()
    
    # If no abstract was found, merge the title into body.
    if not sections['abstract']:
        sections['body'] = (sections['title'] + "\n" + sections['body']).strip()
        sections['title'] = ""

    if not sections['body']:
        sections['body'] = sections['abstract']
        sections['abstract'] = ""
    
    return sections

def clean_article_text(text, num_pages, header_threshold=0.7):
    """
    Cleans the text of an academic article given as a list of page texts.
    
    Steps:
      1. Splits the text into lines.
      2. Uses frequency analysis to identify lines that appear in more than
         header_threshold * number_of_pages (likely headers/footers).
      3. Removes lines matching additional regex patterns (e.g., emails, page numbers).
      4. Returns the cleaned text as a single string.
      
    Args:
      pages: List of strings, each representing the text extracted from one PDF page.
      header_threshold: Fraction of pages a line must appear in to be considered a header/footer.
      
    Returns:
      Cleaned text as a string.
    """
    pages_lines = text.splitlines()

    line_freq = defaultdict(int)
    for line in pages_lines:
      seen_lines = set()
      norm_line = line.strip()
      if norm_line:
          seen_lines.add(norm_line)
      for norm_line in seen_lines:
          line_freq[norm_line] += 1
    
    header_footer_lines = {line for line, count in line_freq.items() if count >= num_pages * header_threshold}
    
    cleaned_pages = []
    for line in pages_lines:
      stripped = line.strip()
      if stripped in header_footer_lines:
        continue
      cleaned_pages.append(line)
    
    cleaned_text = "\n".join(cleaned_pages)
    return cleaned_text, header_footer_lines


def clean_extracted_text(text):
    # Normalize unicode characters to NFKC form to handle ligatures, etc.
    text = unicodedata.normalize('NFKC', text)
    
    # Convert a wide range of Unicode characters to plain ASCII using unidecode.
    text = unidecode(text)

    # Remove page numbers / common footer artifacts (example patterns)
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+\s+/\s+\d+', ' ', text)

    # Remove URLs/DOIs
    text = re.sub(r'(https?://\S+)', ' ', text)
    text = re.sub(r'(doi:\s*\S+)', ' ', text)

    # Remove email addresses, single numbers, and 'www.' links
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'^\s*\d+\s*$', ' ', text)
    text = re.sub(r'www.\S+', ' ', text)

    # Fix hyphenated words across line breaks e.g. "signifi-\ncant" -> "significant"
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    # Convert multiple newlines or tabs to single space
    text = re.sub(r'\s+', ' ', text)

    # Remove leftover, repeated spaces
    text = re.sub(r'\s{2,}', ' ', text)

    return text.strip()

def extract_authors(documents: List[Any]) -> List[str]:
    """Extract authors from PDF metadata and content."""
    authors = []

    if documents and 'author' in documents[0].metadata:
        author_text = documents[0].metadata['author']
        if author_text and author_text != 'Unknown':
            author_names = [name.strip() for name in author_text.split(',')]
            authors.extend([str(name) for name in author_names if name])
    
    # If no authors from metadata, try to extract from content
    if not authors and documents:
        content = documents[0].page_content
        lines = content.split('\n')[:20]
        for line in lines:
            # Pattern for academic papers: "FirstName LastName1 · FirstName LastName2"
            if '·' in line and not line.startswith('Vol') and not line.startswith('Animal'):
                potential_authors = line.split('·')
                for author in potential_authors:
                    clean_author = re.sub(r'\d+', '', author).strip()
                    if len(clean_author) > 3 and ' ' in clean_author:
                        authors.append(str(clean_author))
                break
    
    return authors if authors else ["Unknown Author"]

def extract_breeds(content_lower: str) -> List[str]:
    """Extract dog breeds from PDF content."""
    detected_breeds = []
    
    # Check for common dog breeds
    for breed in CommonDogBreeds.get_all_breeds():
        breed_readable = breed.replace('_', ' ')
        if breed_readable in content_lower:
            detected_breeds.append(breed)
    
    return detected_breeds if detected_breeds else []

def extract_behaviors(content_lower: str) -> List[str]:
    behavior_synonyms = {
        'aggression': ['aggressive', 'attack', 'bite', 'hostile', 'fight'],
        'socialization': ['social', 'interaction', 'social behavior', 'social behaviour'],
        'cognition': ['cognitive', 'decision-making', 'decision making', 'learning'],
        'communication': ['signal', 'expression', 'vocal', 'body language'],
        'emotions': ['emotional', 'emotion', 'affect', 'affective']
    }
    detected_behaviors = []

    for behavior, synonyms in behavior_synonyms.items():
        if any(syn in content_lower for syn in synonyms):
            if behavior not in detected_behaviors:
                detected_behaviors.append(behavior)
    return detected_behaviors if detected_behaviors else []

def determine_study_type(content_lower: str, keywords: List[str]) -> Optional[str]:
    """Enhanced study type detection."""
   
    # Check keywords first
    keyword_indicators = {
        'experimental': ['experiment', 'trial', 'treatment', 'intervention', 'manipulation'],
        'observational': ['observ', 'field study', 'natural', 'ethogram', 'behaviour recording'],
        'review': ['review', 'meta-analysis', 'systematic review', 'literature review'],
        'case_study': ['case study', 'case report', 'single case']
    }
    
    for study_type, indicators in keyword_indicators.items():
        if any(indicator in content_lower for indicator in indicators):
            return study_type
    
    # Check by methodology description
    if any(word in content_lower for word in ['we tested', 'subjects were', 'experiment consisted']):
        return 'experimental'
    
    return None

def parse_creation_date(date_string: str) -> Optional[str]:
    """Parse creation date from PDF metadata and extract full date."""
    try:
        # Handle format like '2021-08-14T14:33:08+05:30'
        if 'T' in date_string:
            date_part = date_string.split('T')[0]  # Gets '2021-08-14'
            return date_part
        # Handle other common formats
        elif '-' in date_string and len(date_string.split('-')) >= 3:
            parts = date_string.split('-')[:3]  # Take first 3 parts: year-month-day
            return '-'.join(parts)
    except (ValueError, IndexError):
        return None
    return None

def process_content(full_content: str, num_pages: int, header_threshold: float = 0.7) -> Dict[str, str]:
    """ Processes the full content of an academic paper to extract structured sections."""
    cleaned_text, removed_line = clean_article_text(full_content, num_pages, 0.7)

    sections = split_into_sections(cleaned_text)

    for key, text in sections.items():
        sections[key] = clean_extracted_text(text)

    return sections

def has_malformed_title(title: str) -> bool:
    """Check if title looks like a malformed PDF metadata title."""
    if not title:
        return False
    
    # Pattern for titles like "pone.0003349 1..7", "pbio.1000451 1..13"
    malformed_patterns = [
        r'^[a-z]+\.\d+\s+\d+\.\.\d+$',  # pone.0003349 1..7
        r'^[a-z]+\.\d+$',               # just pone.0003349
        r'^\d+\.\.\d+$',                # just 1..7
        r'^Paper from .+\.pdf$'         # fallback title
    ]
    
    return any(re.match(pattern, title.strip()) for pattern in malformed_patterns)

def parse_academic_paper(pdf_path: str, store: bool = False) -> Paper:

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    parent_dir = pdf_path.parent
    pdf_name = pdf_path.stem

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    if not documents:
        raise ValueError(f"No content extracted from {pdf_path}")

    metadata = documents[0].metadata
    num_pages = metadata.get('total_pages', len(documents))
    full_content = "\n".join([doc.page_content for doc in documents])

    paper_id = Path(pdf_path).stem
    title = metadata.get('title', '')
    authors = extract_authors(documents)

    sections = process_content(full_content, num_pages)
    if title is None or title == "" or has_malformed_title(title):
        title = sections['title'].strip()

    if store:
        # Store as json
        sections['title'] = title
        sections['metadata'] = metadata

        parent_dir_clean = parent_dir.with_name(parent_dir.name + "_clean")
        parent_dir_clean.mkdir(exist_ok=True)
        json_path = parent_dir_clean / pdf_name
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(sections, f, ensure_ascii=False, indent=4)

    keywords = []
    if 'keywords' in metadata:
        keywords = [kw.strip() for kw in metadata['keywords'].split(',')]

    content_lower = full_content.lower()
    detected_breeds = extract_breeds(content_lower)
    detected_behaviors = extract_behaviors(content_lower)
    study_type = determine_study_type(content_lower, keywords)

    paper = Paper(
        id=paper_id,
        title=title,
        authors=authors,
        subject=metadata.get('subject', '').split(',')[0] if metadata.get('subject') else None,
        keywords=keywords,
        abstract=sections['abstract'],
        body=sections['body'],
        conclusion=sections['conclusion'],
        references=sections['references'],
        dog_breeds=detected_breeds,
        behavior_categories=detected_behaviors,
        study_type=study_type,
        publication_date=metadata.get('creationdate', ''),
        doi=metadata.get('doi'),
        metadata=metadata,
    )
    return paper


class VectorDatabase:
    """Vector database interface using ChromaDB for paper storage and retrieval."""
    
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "dog_behavior", embedding_model: str = 'all-MiniLM-L6-v2', chunk_size: int = None, skip_duplicates: bool = True):
        """Initialize the vector database."""
        self.db_path = db_path
        self.collection_name = collection_name
        
        self.client = chromadb.PersistentClient(path=db_path)

        self.paper_dicts = {}
        # self.configs = {}
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            with open(os.path.join(db_path, f"{collection_name}_papers.json"), 'r', encoding='utf-8') as f:
                self.paper_dicts = json.load(f)
            with open(os.path.join(db_path, f"{collection_name}_configs.json"), 'r', encoding='utf-8') as f:
                configs = json.load(f)
            self.embedding_model_name = configs.get('embedding_model', None)
            self.chunk_size = configs.get('chunk_size', None)
            self.skip_duplicates = configs.get('skip_duplicates', True)
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name, trust_remote_code=True, device=device)
            except Exception as e:
                logger.error(f"Error loading embedding model '{self.embedding_model_name}': {str(e)}")
                logger.warning("Falling back to default 'all-MiniLM-L6-v2' model")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', trust_remote_code=True, device=device)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Academic papers on dog behavior"}
            )
            self.embedding_model_name = embedding_model
            self.embedding_model = SentenceTransformer(embedding_model, trust_remote_code=True, device=device)
            self.chunk_size = chunk_size
            self.skip_duplicates = skip_duplicates
            logger.info(f"Created new collection: {collection_name}")

    def paper_exists(self, paper_id: str) -> bool:
        """Check if a paper already exists in the database."""
        try:
            # results = self.collection.get(ids=[paper_id])
            return paper_id in self.paper_dicts
        except Exception as e:
            logger.error(f"Error checking if paper exists {paper_id}: {str(e)}")
            return False
        
    def update_paper(self, paper: Paper) -> bool:
        """Update an existing paper or add if it doesn't exist."""
        try:
            if self.paper_exists(paper.id):
                self.delete_paper(paper.id)
                logger.info(f"Updating existing paper: {paper.id}")
            
            self.save_configs()
            
            return self.add_paper_sentences(paper)
        
        except Exception as e:
            logger.error(f"Error updating paper {paper.id}: {str(e)}")
            return False

        
    def add_paper_sentences(self, paper: Paper) -> bool:
        """Add a paper's content to the database by splitting it into sentences."""
        
        if self.skip_duplicates and self.paper_exists(paper.id):
            logger.info(f"Paper {paper.id} already exists, skipping")
            return True

        if self.chunk_size is None:
            logger.error("Chunk size is not set. Cannot split paper into sentences.")
            return False
        
        if self.chunk_size <= 0:
            logger.error(f"Invalid chunk size: {self.chunk_size}. Must be positive.")
            return False

        try:
            
            chunk_ids = []
            chunks_added = 0
            sections_processed = ['title', 'abstract', 'body', 'conclusion']
            
            for section in sections_processed:
                content = getattr(paper, section, "")
                if not content or not content.strip():
                    logger.debug(f"Skipping empty section '{section}' for paper {paper.id}")
                    continue
                    
                sentences = nltk.sent_tokenize(content.strip(), language='english')
                
                if not sentences:
                    logger.debug(f"No sentences found in section '{section}' for paper {paper.id}")
                    continue
                
                chunks = []
                if section == 'title':
                    chunks.append(content.strip())
                else:
                    for i in range(0, len(sentences), self.chunk_size):
                        chunk = " ".join(sentences[i:i + self.chunk_size]).strip()
                        if chunk:
                            chunks.append(chunk)

                for chunk_idx, chunk_text in enumerate(chunks):
                    try:
                        embedding = self.embedding_model.encode(
                            chunk_text,
                            truncation=True,
                            max_length=getattr(self.embedding_model, 'max_seq_length', 512)
                        ).tolist()
                        
                        chunk_id = f"{paper.id}_{section}_{chunk_idx}"
                        
                        chunk_metadata = {}
                        chunk_metadata['paper_id'] = paper.id
                        chunk_metadata['section'] = section
                        chunk_metadata['chunk_index'] = chunk_idx
                        chunk_metadata['total_chunks'] = len(chunks)
                        
                        self.collection.add(
                            embeddings=[embedding],
                            documents=[chunk_text],
                            metadatas=[chunk_metadata],
                            ids=[chunk_id]
                        )
                        chunks_added += 1
                        chunk_ids.append(chunk_id)
                        
                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk_idx} in section '{section}' for paper {paper.id}: {str(e)}")
                        continue

            if chunks_added > 0:
                paper_metadata = paper.to_dict()
                paper_metadata.pop('abstract', None)
                paper_metadata.pop('body', None)
                paper_metadata.pop('conclusion', None)
                
                self.paper_dicts[paper.id] = paper_metadata
                self.paper_dicts[paper.id]['chunk_ids'] = chunk_ids
                
                self.save_configs()

                logger.info(f"Added {chunks_added} chunks from paper: {paper.title}")
                return True
            else:
                logger.warning(f"No chunks were added for paper: {paper.title}")
                return False
            
        except Exception as e:
            logger.error(f"Error adding paper {paper.id} sentences: {str(e)}")
            return False
    
    
    def add_papers_sentences_batch(self, papers: List[Paper]) -> int:
        """Add multiple papers' content to the database by splitting into sentences."""
        if not papers:
            logger.warning("No papers provided for batch sentence processing")
            return 0

        # Check for duplicates if requested
        papers_to_add = papers
        if self.skip_duplicates:
            existing_paper_ids = set()
            for paper in papers:
                if paper.id in self.paper_dicts:
                    existing_paper_ids.add(paper.id)
            
            papers_to_add = [p for p in papers if p.id not in existing_paper_ids]
            
            if existing_paper_ids:
                logger.info(f"Skipping {len(existing_paper_ids)} papers that already have chunks: {list(existing_paper_ids)[:3]}...")
        
        if not papers_to_add:
            logger.info("All papers already processed, nothing to add")
            return 0
        
        logger.info(f"Processing {len(papers_to_add)} new papers out of {len(papers)} total")
    

        if self.chunk_size is None:
            logger.error("Chunk size is not set. Cannot split papers into sentences.")
            return 0
        
        if self.chunk_size <= 0:
            logger.error(f"Invalid chunk size: {self.chunk_size}. Must be positive.")
            return 0
        
        logger.info(f"Starting batch sentence processing of {len(papers_to_add)} papers with chunk size {self.chunk_size}...")
        
        # Prepare batch data
        embeddings = []
        documents = []
        metadatas = []
        ids = []
        
        papers_processed = 0
        total_chunks = 0
        
        for paper_idx, paper in enumerate(papers_to_add):
            try:

                paper_chunk_list = []
                paper_chunks = 0
                sections_processed = ['title', 'abstract', 'body', 'conclusion']
                
                logger.debug(f"Processing paper {paper_idx + 1}/{len(papers)}: {paper.title}")
                
                for section in sections_processed:
                    content = getattr(paper, section, "")
                    if not content or not content.strip():
                        logger.debug(f"Skipping empty section '{section}' for paper {paper.id}")
                        continue
                        
                    # Tokenize into sentences
                    sentences = nltk.sent_tokenize(content.strip(), language='english')
                    
                    if not sentences:
                        logger.debug(f"No sentences found in section '{section}' for paper {paper.id}")
                        continue
                    
                    # Create chunks
                    chunks = []
                    if section == 'title':
                        # Keep title as single chunk
                        chunks.append(content.strip())
                    else:
                        # Split other sections into sentence chunks
                        for i in range(0, len(sentences), self.chunk_size):
                            chunk = " ".join(sentences[i:i + self.chunk_size]).strip()
                            if chunk:  # Only add non-empty chunks
                                chunks.append(chunk)

                    # Process each chunk
                    for chunk_idx, chunk_text in enumerate(chunks):
                        try:
                            # Generate embedding
                            embedding = self.embedding_model.encode(
                                chunk_text,
                                truncation=True,
                                max_length=getattr(self.embedding_model, 'max_seq_length', 512)
                            ).tolist()
                            
                            # Create unique chunk ID
                            chunk_id = f"{paper.id}_{section}_{paper_chunks}"
                            
                            # Create chunk metadata with paper information
                            chunk_metadata = {}
                            chunk_metadata['paper_id'] = paper.id
                            chunk_metadata['section'] = section
                            chunk_metadata['chunk_index'] = paper_chunks
                            chunk_metadata['total_chunks'] = len(chunks)
                        
                            # Add to batch arrays
                            embeddings.append(embedding)
                            documents.append(chunk_text)
                            metadatas.append(chunk_metadata)
                            paper_chunk_list.append(chunk_id)
                            
                            paper_chunks += 1
                            total_chunks += 1
                            
                        except Exception as e:
                            logger.error(f"Error processing chunk {paper_chunks} in section '{section}' for paper {paper.id}: {str(e)}")
                            continue

                if paper_chunks > 0:
                    papers_processed += 1
                    ids.extend(paper_chunk_list)

                    paper_metadata = paper.to_dict()
                    paper_metadata.pop('abstract', None)
                    paper_metadata.pop('body', None)
                    paper_metadata.pop('conclusion', None)
                    
                    self.paper_dicts[paper.id] = paper_metadata
                    self.paper_dicts[paper.id]['chunk_ids'] = paper_chunk_list
                    logger.debug(f"Prepared {paper_chunks} chunks from paper: {paper.title}")
                else:
                    logger.warning(f"No chunks were prepared for paper: {paper.title}")
                        
            except Exception as e:
                logger.error(f"Error preparing paper {paper.id} ({paper_idx + 1}/{len(papers)}): {str(e)}")
                continue
        
        # Batch add to collection
        successful_adds = 0
        if embeddings:
            try:
                logger.info(f"Adding {len(embeddings)} chunks from {papers_processed} papers to vector database...")
                
                self.collection.add(
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                successful_adds = len(embeddings)
                logger.info(f"Successfully added {successful_adds} chunks from {papers_processed} papers")

                self.save_configs()
                
            except Exception as e:
                logger.error(f"Error in batch add to collection: {str(e)}")
                successful_adds = 0
        else:
            logger.warning("No valid chunks to add to the database")
        
        if papers_processed != len(papers):
            failed_papers = len(papers) - papers_processed
            logger.warning(f"Batch processing completed: {papers_processed} papers successful, {failed_papers} papers failed")
        
        return successful_adds
    
    def add_papers_from_directory(self, directory: str) -> int:
        """Add all papers from a directory to the database."""
        if not directory:
            logger.error("Directory path cannot be empty")
            return 0
            
        directory_path = Path(directory)
        if not directory_path.exists():
            logger.error(f"Directory does not exist: {directory}")
            return 0
        
        if not directory_path.is_dir():
            logger.error(f"Path is not a directory: {directory}")
            return 0
        
        # For sentence embedding, check if chunk_size is set
        if self.chunk_size is None or self.chunk_size <= 0:
            logger.error("Chunk size must be set and positive for sentence embedding method")
            return 0
        
        logger.info(f"Starting to process papers from directory: {directory}")
        
        # Find PDF files
        pdf_files = list(directory_path.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in directory: {directory}")
            return 0
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Parse papers from PDFs
        successful_papers = []
        failed_papers = []
        skipped_papers = []
        
        for i, pdf_path in enumerate(pdf_files):
            try:
                paper_id = pdf_path.stem
                if self.skip_duplicates and self.paper_exists(paper_id):
                    logger.debug(f"Paper {paper_id} already exists, skipping")
                    skipped_papers.append(pdf_path.name)
                    continue

                logger.debug(f"Processing file {i+1}/{len(pdf_files)}: {pdf_path.name}")
                paper = parse_academic_paper(str(pdf_path), store=False)
                successful_papers.append(paper)
                logger.debug(f"Successfully parsed: {paper.title}")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {str(e)}")
                failed_papers.append(pdf_path.name)
                continue
        
        # Log parsing results
        if skipped_papers:
            logger.info(f"Skipped {len(skipped_papers)} existing papers")
        if failed_papers:
            logger.warning(f"Failed to parse {len(failed_papers)} files: {', '.join(failed_papers)}")
        
        logger.info(f"Parsed {len(successful_papers)} new papers, skipped {len(skipped_papers)}, failed {len(failed_papers)}")
        
        if not successful_papers:
            logger.warning("No papers were successfully parsed")
            return 0
        
        # Add papers to database using selected method
        successful_adds = 0
        try:
            logger.info("Adding papers using sentence-based chunking...")
            successful_adds = self.add_papers_sentences_batch(successful_papers)
                
            logger.info(f"Successfully added {successful_adds} chunks to the database")
            
        except Exception as e:
            logger.error(f"Error adding papers to database: {str(e)}")
            return 0
        
        # Final summary
        logger.info(f"Directory processing completed:")
        logger.info(f"  - Files found: {len(pdf_files)}")
        logger.info(f"  - Papers parsed: {len(successful_papers)}")
        logger.info(f"  - Papers failed: {len(failed_papers)}")
        logger.info(f"  - Database entries added: {successful_adds}")
        
        return successful_adds
        
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for papers based on query text and optional filters."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Prepare where clause for filtering
            where_clause = None
            
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    paper_id = results["metadatas"][0][i].get('paper_id', 'Unknown')
                    result = {
                        "id": chunk_id,
                        "document": results["documents"][0][i],
                        "title": self.paper_dicts.get(paper_id, {}).get('title', 'Unknown Title'),
                        "authors": self.paper_dicts.get(paper_id, {}).get('authors', []),
                        "metadata": results["metadatas"][0][i],
                        "similarity_score": 1 - results["distances"][0][i],  # Convert distance to similarity
                    }
                    formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} results for query: {query}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            return []
    
    # def search_with_context(self, query: str, behavior_categories: List[str] = None,
    #                       dog_breeds: List[str] = None, study_type: str = None,
    #                       age_range: str = None, n_results: int = 10) -> List[Dict[str, Any]]:
    #     """Search with contextual filters."""
    #     filters = {}
        
    #     if behavior_categories:
    #         filters["behavior_categories"] = behavior_categories
    #     if dog_breeds:
    #         filters["dog_breeds"] = dog_breeds
    #     if study_type:
    #         filters["study_type"] = study_type
    #     if age_range:
    #         filters["age_range"] = age_range
        
    #     return self.search(query, n_results, filters)
    
    def get_paper_by_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific paper by ID."""
        try:
            results = self.collection.get(
                ids=[paper_id],
                include=["documents", "metadatas"]
            )
            
            if results["ids"] and results["ids"][0]:
                return {
                    "id": results["ids"][0],
                    "document": results["documents"][0],
                    "metadata": results["metadatas"][0]
                }
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving paper {paper_id}: {str(e)}")
            return None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            
            # Get sample of metadata to analyze
            sample_results = self.collection.get(
                limit=min(100, count),
                include=["metadatas"]
            )
            
            # Analyze behavior categories
            behavior_counts = {}
            study_type_counts = {}
            
            if sample_results["metadatas"]:
                for metadata in sample_results["metadatas"]:
                    # Count behavior categories
                    if metadata.get("behavior_categories"):
                        behaviors = json.loads(metadata["behavior_categories"])
                        for behavior in behaviors:
                            behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
                    
                    # Count study types
                    if metadata.get("study_type"):
                        study_type = metadata["study_type"]
                        study_type_counts[study_type] = study_type_counts.get(study_type, 0) + 1
            
            return {
                "total_papers": count,
                "behavior_categories": behavior_counts,
                "study_types": study_type_counts
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"total_papers": 0, "behavior_categories": {}, "study_types": {}}
    
    def delete_paper(self, paper_id: str) -> bool:
        """Delete a paper from the database."""
        try:
            chunks = self.paper_dicts.get(paper_id, {}).get('chunk_ids', [])
            if not chunks:
                logger.warning(f"No chunks found for paper {paper_id}, skipping deletion")
                return False
            self.collection.delete(ids=chunks)
            self.paper_dicts.pop(paper_id, None)
            logger.info(f"Deleted paper: {paper_id}")
            self.save_configs()
            return True
        except Exception as e:
            logger.error(f"Error deleting paper {paper_id}: {str(e)}")
            return False
    
    def clear_collection(self) -> bool:
        """Clear all papers from the collection."""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Academic papers on dog behavior"}
            )
            self.paper_dicts.clear()
            self.save_configs()
            logger.info("Cleared collection")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False
        
    def save_configs(self) -> bool:
        """Save current configurations to a JSON file."""
        try:
            papers_path = os.path.join(self.db_path, f"{self.collection_name}_papers.json")
            config_path = os.path.join(self.db_path, f"{self.collection_name}_configs.json")
            
            configs = {
                'db_path': self.db_path,
                'collection_name': self.collection_name,
                'embedding_model': self.embedding_model_name,
                'chunk_size': self.chunk_size,
                'skip_duplicates': self.skip_duplicates,
                'num_papers': len(self.paper_dicts)
            }
            # Ensure directory exists
            os.makedirs(self.db_path, exist_ok=True)
            with open(papers_path, 'w', encoding='utf-8') as f:
                json.dump(self.paper_dicts, f, ensure_ascii=False, indent=4)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(configs, f, ensure_ascii=False, indent=4)
            logger.info(f"Saved configurations to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configurations: {str(e)}")
            return False
