#!/usr/bin/env python3
"""Enhanced Markdown Chunker for RAG Applications

Processes markdown files into optimized chunks with:
- Header-aware splitting
- Code block preservation
- Context-aware chunk merging
- Enhanced metadata
"""

import re
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from config.settings import settings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class EnhancedRecursiveSplitter:
    """Improved text splitter with smart chunk merging"""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150, min_chunk_words: int = 30):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n## ", "\n\n", "\n", " ", ""],
            keep_separator=True,
            length_function=len,
        )
        self.min_chunk_words = min_chunk_words

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents with context preservation"""
        chunks = []
        for doc in documents:
            # First pass: standard splitting
            standard_chunks = self.splitter.split_documents([doc])
            
            # Second pass: merge small chunks
            merged_chunks = self._merge_small_chunks(standard_chunks)
            chunks.extend(merged_chunks)
        return chunks
    
    def _merge_small_chunks(self, chunks: List[Document]) -> List[Document]:
        """Merge chunks smaller than threshold with neighbors"""
        if not chunks:
            return []
            
        merged = []
        buffer = ""
        buffer_metadata = chunks[0].metadata.copy()
        
        for chunk in chunks:
            content = chunk.page_content
            word_count = len(content.split())
            
            if word_count < self.min_chunk_words:
                buffer += "\n\n" + content
            else:
                if buffer:
                    # Create new document with merged content
                    new_doc = Document(
                        page_content=(buffer + "\n\n" + content).strip(),
                        metadata=buffer_metadata
                    )
                    merged.append(new_doc)
                    buffer = ""
                else:
                    merged.append(chunk)
        
        # Handle remaining buffer
        if buffer:
            if merged:
                merged[-1].page_content += "\n\n" + buffer
            else:
                merged.append(Document(
                    page_content=buffer.strip(),
                    metadata=buffer_metadata
                ))
                
        return merged or chunks  # Return original if no merges

class MarkdownProcessor:
    def __init__(self):
        self.min_chunk_words = getattr(settings, "MIN_CHUNK_WORDS", 30)

    def clean_header(self, header: str) -> str:
        """Normalize markdown headers"""
        header = re.sub(r"<.*?>|\[.*?\]\(.*?\)", "", header)  # Remove links/HTML
        header = re.sub(r"[^a-zA-Z0-9\s-]", "", header).strip()  # Remove special chars
        return header or "Untitled"

    def extract_code_blocks(self, markdown: str) -> List[Dict[str, str]]:
        """Identify and extract all code blocks with metadata"""
        code_blocks = []
        pattern = r"```(?P<language>\w*)\n(?P<code>[\s\S]*?)\n```"
        for match in re.finditer(pattern, markdown):
            code_blocks.append({
                "language": match.group("language") or "text",
                "code": match.group("code"),
                "start_pos": match.start(),
                "end_pos": match.end()
            })
        return code_blocks

    def split_markdown(self, markdown: str) -> List[str]:
        """Split markdown into logical sections"""
        parts = []
        last_end = 0
        for block in self.extract_code_blocks(markdown):
            if last_end < block["start_pos"]:
                parts.append(markdown[last_end:block["start_pos"]])
            parts.append(f"```{block['language']}\n{block['code']}\n```")
            last_end = block["end_pos"]
        parts.append(markdown[last_end:])
        return [p.strip() for p in parts if p.strip()]

    def chunk_content(
        self, markdown: str, source_url: str, input_path: Path
    ) -> List[Document]:
        """Three-stage chunking pipeline"""
        try:
            # Stage 1: Header-based splitting
            header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                ]
            )
            header_chunks = header_splitter.split_text(markdown)

            # Stage 2: Recursive splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=getattr(settings, "CHUNK_SIZE", 800),
                chunk_overlap=getattr(settings, "CHUNK_OVERLAP", 150),
                separators=["\n\n## ", "\n\n", "\n", " "],
                keep_separator=True,
                length_function=len,
            )
            intermediate_chunks = text_splitter.split_documents(header_chunks)

            # Stage 3: Context-aware merging
            enhanced_splitter = EnhancedRecursiveSplitter(
                chunk_size=getattr(settings, "CHUNK_SIZE", 800),
                chunk_overlap=getattr(settings, "CHUNK_OVERLAP", 150),
                min_chunk_words=self.min_chunk_words
            )
            final_chunks = enhanced_splitter.split_documents(intermediate_chunks)

            # Add metadata
            for i, chunk in enumerate(final_chunks):
                self._enrich_metadata(chunk, source_url, input_path, i)

            return self._postprocess_chunks(final_chunks)

        except Exception as e:
            logger.error(f"Chunking failed: {str(e)}", exc_info=True)
            raise

    def _enrich_metadata(
        self, chunk: Document, source_url: str, input_path: Path, chunk_id: int
    ) -> None:
        """Add enhanced metadata to chunks"""
        content = chunk.page_content
        headers = {
            k: self.clean_header(v)
            for k, v in chunk.metadata.items()
            if k.startswith("Header")
        }

        chunk.metadata.update({
            "source": source_url,
            "chunk_id": f"{input_path.stem}_{chunk_id}",
            "content_type": self._detect_content_type(content),
            "word_count": len(content.split()),
            "has_code": "```" in content,
            **headers,
        })

    def _detect_content_type(self, text: str) -> str:
        """Classify chunk content type"""
        if "```" in text:
            return "code_example"
        if re.search(r"^\s*-\s|\*\s", text, flags=re.MULTILINE):
            return "list"
        if len(text.split()) < 30:
            return "summary"
        return "paragraph"

    def _postprocess_chunks(self, chunks: List[Document]) -> List[Document]:
        """Final validation and cleanup"""
        valid_chunks = []
        for chunk in chunks:
            content = chunk.page_content.strip()
            if not content:
                continue

            # Ensure code blocks are preserved
            if "```" in content and not content.endswith("```"):
                content += "\n```"

            chunk.page_content = content
            valid_chunks.append(chunk)

        return valid_chunks

def process_file(input_path: Path, output_dir: Path) -> Optional[List[Document]]:
    """Process a markdown file through the enhanced pipeline"""
    try:
        processor = MarkdownProcessor()
        
        logger.info(f"Processing {input_path.name}")
        markdown_content = input_path.read_text(encoding="utf-8")
        
        chunks = processor.chunk_content(
            markdown_content,
            source_url=settings.BASE_URL,
            input_path=input_path
        )
        
        # Save results
        output_path = output_dir / f"{input_path.stem}_chunks.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([chunk.dict() for chunk in chunks], f, indent=2)
            
        logger.info(f"Generated {len(chunks)} optimized chunks")
        return chunks
        
    except Exception as e:
        logger.error(f"Failed to process {input_path}: {str(e)}")
        return None

def main() -> bool:
    """Main execution pipeline"""
    try:
        input_dir = Path(settings.RAW_DATA_DIR)
        output_dir = Path(settings.PROCESSED_DATA_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_chunks = []
        for md_file in input_dir.glob("*.md"):
            if chunks := process_file(md_file, output_dir):
                all_chunks.extend(chunks)

        logger.info(f"Total chunks generated: {len(all_chunks)}")
        return True

    except Exception as e:
        logger.critical(f"Pipeline failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    if main():
        sys.exit(0)
    sys.exit(1)