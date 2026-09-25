from pathlib import Path
from uuid import uuid4

import pytest

from ragforge.adapters.state.in_memory import InMemoryIndexStateStore
from ragforge.adapters.state.json_file import JsonFileIndexStateStore
from ragforge.domain.models import DocumentIndexRecord


def _make_record(path: str = "/docs/test.txt", hash_val: str = "hash123") -> DocumentIndexRecord:
    return DocumentIndexRecord(
        file_path=path,
        document_id=uuid4(),
        content_hash=hash_val,
        chunk_count=3,
        metadata={"title": "Test Title"},
    )


@pytest.mark.asyncio
async def test_in_memory_state_store_lifecycle() -> None:
    """Verify CRUD and count operations on InMemoryIndexStateStore."""
    store = InMemoryIndexStateStore()
    assert store.count() == 0
    assert await store.get("/path/doc1.txt") is None

    rec1 = _make_record("/path/doc1.txt", "hash_aaa")
    rec2 = _make_record("/path/doc2.txt", "hash_bbb")

    await store.set(rec1)
    await store.set(rec2)

    assert store.count() == 2
    fetched = await store.get("/path/doc1.txt")
    assert fetched is not None
    assert fetched.content_hash == "hash_aaa"
    assert fetched.document_id == rec1.document_id

    all_records = await store.get_all()
    assert len(all_records) == 2
    assert "/path/doc1.txt" in all_records
    assert "/path/doc2.txt" in all_records

    # Delete
    deleted = await store.delete("/path/doc1.txt")
    assert deleted is True
    assert store.count() == 1
    assert await store.get("/path/doc1.txt") is None

    # Delete non-existent
    assert await store.delete("/non/existent.txt") is False

    # Clear
    await store.clear()
    assert store.count() == 0
    assert await store.get_all() == {}


@pytest.mark.asyncio
async def test_json_file_state_store_persistence(tmp_path: Path) -> None:
    """Verify JsonFileIndexStateStore persists data to disk and reloads on new instance."""
    state_file = tmp_path / "subdir" / "index_state.json"
    store1 = JsonFileIndexStateStore(file_path=state_file)
    assert store1.count() == 0

    rec1 = _make_record(str(tmp_path / "doc1.txt"), "hash_111")
    rec2 = _make_record(str(tmp_path / "doc2.txt"), "hash_222")

    await store1.set(rec1)
    await store1.set(rec2)
    assert store1.count() == 2
    assert state_file.exists()

    # Re-instantiate pointing to the same file on disk
    store2 = JsonFileIndexStateStore(file_path=state_file)
    assert store2.count() == 2
    loaded_rec1 = await store2.get(str(tmp_path / "doc1.txt"))
    assert loaded_rec1 is not None
    assert loaded_rec1.document_id == rec1.document_id
    assert loaded_rec1.content_hash == "hash_111"
    assert loaded_rec1.chunk_count == 3
    assert loaded_rec1.metadata.get("title") == "Test Title"

    # Delete in store2
    deleted = await store2.delete(str(tmp_path / "doc1.txt"))
    assert deleted is True
    assert store2.count() == 1

    # Re-instantiate third instance to verify deletion persisted
    store3 = JsonFileIndexStateStore(file_path=state_file)
    assert store3.count() == 1
    assert await store3.get(str(tmp_path / "doc1.txt")) is None
    assert await store3.get(str(tmp_path / "doc2.txt")) is not None

    # Clear
    await store3.clear()
    assert store3.count() == 0
    store4 = JsonFileIndexStateStore(file_path=state_file)
    assert store4.count() == 0


@pytest.mark.asyncio
async def test_json_file_state_store_corrupt_file_handling(tmp_path: Path) -> None:
    """Verify corrupt or empty JSON file initializes empty state gracefully."""
    state_file = tmp_path / "corrupt_state.json"
    state_file.write_text("{ not valid json !!! }", encoding="utf-8")

    store = JsonFileIndexStateStore(file_path=state_file)
    assert store.count() == 0
    assert await store.get("/any/path.txt") is None

    # Setting a record recovers and writes valid JSON
    rec = _make_record(str(tmp_path / "recovered.txt"), "hash_rec")
    await store.set(rec)
    assert store.count() == 1

    reloaded = JsonFileIndexStateStore(file_path=state_file)
    assert reloaded.count() == 1
    assert await reloaded.get(str(tmp_path / "recovered.txt")) is not None
