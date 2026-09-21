import hashlib
import re
from collections.abc import Callable, Sequence
from typing import NamedTuple
from uuid import UUID, uuid5

from ragforge.domain.exceptions import ChunkingError
from ragforge.domain.models import Chunk, ChunkMetadata, Document
from ragforge.ports.chunkers import BaseChunker

# Deterministic namespace for RAGForge chunk identifier generation
RAGFORGE_CHUNK_NAMESPACE = UUID("6ba7b812-9dad-11d1-80b4-00c04fd430c8")

_MARKDOWN_HEADING_REGEX = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class _HeadingInfo(NamedTuple):
    start: int
    end: int
    level: int
    title: str
    raw: str


def approximate_token_count(text: str) -> int:
    """Deterministic heuristic token count estimation (~4 characters per token).

    Standard heuristic ((len(text) + 3) // 4) that closely aligns with BPE/subword
    tokenizers for English text. Configured as the default token estimation function,
    allowing downstream integration of exact tokenizers (e.g. tiktoken) without
    modifying chunking domain models.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


class DeterministicChunker(BaseChunker):
    """Deterministic, boundary-aware chunker for plain text and Markdown documents.

    Splits text hierarchically respecting Markdown section headings, paragraph breaks,
    line breaks, sentence boundaries, and word boundaries while guaranteeing:
    - Deterministic chunk ordering, indexes, hashes, and UUIDv5 identifiers.
    - Zero empty chunks.
    - Accurate character offset spans (start_char_idx, end_char_idx).
    - Preserved document-level and Markdown section-level provenance metadata.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ChunkingError(f"chunk_size must be greater than 0, got {chunk_size}")
        if chunk_overlap < 0:
            raise ChunkingError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ChunkingError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less "
                f"than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._token_counter = token_counter or approximate_token_count

    def _extract_headings(self, text: str) -> list[_HeadingInfo]:
        """Find all Markdown headings with their line spans and level information."""
        headings: list[_HeadingInfo] = []
        for match in _MARKDOWN_HEADING_REGEX.finditer(text):
            level = len(match.group(1))
            title = match.group(2).strip()
            raw = match.group(0).strip()
            headings.append(
                _HeadingInfo(
                    start=match.start(),
                    end=match.end(),
                    level=level,
                    title=title,
                    raw=raw,
                )
            )
        return headings

    def _find_active_heading(
        self,
        headings: list[_HeadingInfo],
        chunk_start: int,
        chunk_end: int,
    ) -> _HeadingInfo | None:
        """Find the active section heading applicable to a chunk slice."""
        active: _HeadingInfo | None = None
        for heading in headings:
            if heading.start <= chunk_start:
                active = heading
            elif heading.start < chunk_end:
                # Heading occurs within this chunk; adopt it if no prior heading or closer
                if active is None or (heading.start - chunk_start) < (chunk_end - heading.start):
                    active = heading
            else:
                break
        return active

    def _find_chunk_end(
        self,
        text: str,
        start: int,
        target_end: int,
        text_len: int,
        separators: list[str],
    ) -> int:
        """Find the optimal boundary index to end a chunk."""
        if target_end >= text_len:
            return text_len

        search_start = max(
            start + 1,
            target_end - max(self.chunk_overlap, int(self.chunk_size * 0.4)),
        )

        for sep in separators:
            idx = text.rfind(sep, search_start, target_end)
            if idx != -1:
                if sep.startswith("\n#") or sep == "\n\n":
                    return idx
                return idx + len(sep)

        # Fallback to earlier newline or space if available
        for sep in ["\n", " "]:
            idx = text.rfind(sep, start + 1, search_start)
            if idx != -1:
                return idx + len(sep)

        # Hard break if no delimiter found in continuous string
        return target_end

    def chunk(self, document: Document) -> Sequence[Chunk]:
        """Split a domain Document into discrete, deterministic Chunks.

        Args:
            document: Domain Document entity with raw_content.

        Returns:
            A sequence of populated domain Chunk instances.
        """
        text = document.raw_content or ""
        if not text or not text.strip():
            return []

        headings = self._extract_headings(text)
        stripped_text = text.strip()

        # Handle short documents that fit within a single chunk
        if len(stripped_text) <= self.chunk_size:
            start_idx = text.find(stripped_text)
            end_idx = start_idx + len(stripped_text)
            heading = self._find_active_heading(headings, start_idx, end_idx)
            chunk_hash = hashlib.sha256(stripped_text.encode("utf-8")).hexdigest()
            chunk_id = uuid5(
                RAGFORGE_CHUNK_NAMESPACE,
                f"urn:ragforge:chunk:{document.id}:0:{chunk_hash}",
            )
            extra_meta: dict[str, str | int] = {
                "document_title": document.title,
                "source_type": str(document.source_type),
            }
            if document.file_path is not None:
                extra_meta["file_path"] = document.file_path
            if heading is not None:
                extra_meta["heading_level"] = heading.level
                extra_meta["heading_raw"] = heading.raw

            single_chunk = Chunk(
                id=chunk_id,
                document_id=document.id,
                chunk_index=0,
                content=stripped_text,
                token_count=self._token_counter(stripped_text),
                content_hash=chunk_hash,
                metadata=ChunkMetadata(
                    section_header=heading.title if heading else None,
                    start_char_idx=start_idx,
                    end_char_idx=end_idx,
                    extra=extra_meta,
                ),
            )
            return [single_chunk]

        # Multi-chunk boundary-aware splitting
        chunks: list[Chunk] = []
        start = 0
        text_len = len(text)
        chunk_index = 0
        separators = [
            "\n# ",
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
        ]

        while start < text_len:
            # Advance past leading whitespace
            while start < text_len and text[start].isspace():
                start += 1
            if start >= text_len:
                break

            target_end = min(start + self.chunk_size, text_len)
            end = self._find_chunk_end(text, start, target_end, text_len, separators)

            chunk_slice = text[start:end]
            content = chunk_slice.strip()

            if content:
                s_idx = text.find(content, start)
                e_idx = s_idx + len(content)

                heading = self._find_active_heading(headings, s_idx, e_idx)
                chunk_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                chunk_id = uuid5(
                    RAGFORGE_CHUNK_NAMESPACE,
                    f"urn:ragforge:chunk:{document.id}:{chunk_index}:{chunk_hash}",
                )

                extra_meta = {
                    "document_title": document.title,
                    "source_type": str(document.source_type),
                }
                if document.file_path is not None:
                    extra_meta["file_path"] = document.file_path
                if heading is not None:
                    extra_meta["heading_level"] = heading.level
                    extra_meta["heading_raw"] = heading.raw

                chunk_obj = Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    chunk_index=chunk_index,
                    content=content,
                    token_count=self._token_counter(content),
                    content_hash=chunk_hash,
                    metadata=ChunkMetadata(
                        section_header=heading.title if heading else None,
                        start_char_idx=s_idx,
                        end_char_idx=e_idx,
                        extra=extra_meta,
                    ),
                )
                chunks.append(chunk_obj)
                chunk_index += 1

            if end >= text_len:
                break

            # Advance window by subtracting overlap
            next_start = max(start + 1, end - self.chunk_overlap)
            # Align to word boundary to avoid slicing words mid-token
            if (
                next_start < end
                and not text[next_start - 1].isspace()
                and not text[next_start].isspace()
            ):
                space_idx = text.find(" ", next_start, end)
                if space_idx != -1:
                    next_start = space_idx + 1

            while next_start < end and text[next_start].isspace():
                next_start += 1

            start = next_start

        return chunks
