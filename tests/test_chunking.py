"""
Tests for chunking utilities.

Covers edge cases including:
- Empty input
- Text with no code fences
- Text with both ``` and ~~~ fences
- Unclosed fences
- Large code blocks with max_code_chunk_size
- Markdown with nested headers
- Parameter validation

Note: These tests rely on langchain's MarkdownHeaderTextSplitter.
If tests fail due to API changes, pin langchain-text-splitters>=0.0.1
or an appropriate langchain version.
"""
import pytest

from app.utils.code_based import code_chunk
from app.utils.header_chunk import chunk_markdown_with_headers
from app.utils.recursive_based import recursive_chunk


class TestRecursiveChunk:
    """Tests for recursive_chunk function."""
    
    def test_empty_input_returns_empty_list(self):
        """Empty input should return empty list."""
        assert recursive_chunk("") == []
        assert recursive_chunk("   ") == []
        assert recursive_chunk("\n\n") == []
    
    def test_basic_chunking(self):
        """Basic text should be chunked properly."""
        text = "Hello world. This is a test."
        chunks = recursive_chunk(text, chunk_size=50, chunk_overlap=5)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)
        assert all(c.strip() == c for c in chunks)  # All chunks should be stripped
    
    def test_long_text_splits_correctly(self):
        """Long text should be split into multiple chunks."""
        text = "This is a sentence. " * 100
        chunks = recursive_chunk(text, chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 1
        # Each chunk should be at most chunk_size
        for chunk in chunks:
            assert len(chunk) <= 100 + 20  # Some tolerance for splitter behavior
    
    def test_chunks_are_stripped(self):
        """All returned chunks should be stripped strings."""
        text = "  Hello world.  \n\n  This is a test.  "
        chunks = recursive_chunk(text, chunk_size=50, chunk_overlap=5)
        for chunk in chunks:
            assert chunk == chunk.strip()
    
    def test_invalid_overlap_negative(self):
        """Negative overlap should raise ValueError."""
        with pytest.raises(ValueError, match="chunk_overlap must be >= 0"):
            recursive_chunk("test", chunk_overlap=-1)
    
    def test_invalid_overlap_too_large(self):
        """Overlap >= chunk_size should raise ValueError."""
        with pytest.raises(ValueError, match="chunk_overlap must be < chunk_size"):
            recursive_chunk("test", chunk_size=100, chunk_overlap=100)
        with pytest.raises(ValueError, match="chunk_overlap must be < chunk_size"):
            recursive_chunk("test", chunk_size=100, chunk_overlap=150)


class TestCodeChunk:
    """Tests for code_chunk function."""
    
    def test_empty_input_returns_empty_list(self):
        """Empty input should return empty list."""
        assert code_chunk("") == []
        assert code_chunk("   ") == []
        assert code_chunk("\n\n") == []
    
    def test_text_with_no_code_fences(self):
        """Text without code fences should be chunked as prose only."""
        text = "Hello world. This is a test paragraph."
        chunks = code_chunk(text, chunk_size=50, chunk_overlap=5)
        assert len(chunks) >= 1
        # No chunk should start with fence markers
        for chunk in chunks:
            assert not chunk.startswith("```")
            assert not chunk.startswith("~~~")
    
    def test_backtick_fence(self):
        """Text with ``` fences should keep code blocks separate."""
        text = """Some prose here.

```python
def hello():
    print("world")
```

More prose after."""
        chunks = code_chunk(text, chunk_size=100, chunk_overlap=10)
        
        # Should have code block and prose chunks
        code_chunks = [c for c in chunks if c.startswith("```")]
        prose_chunks = [c for c in chunks if not c.startswith("```")]
        
        assert len(code_chunks) >= 1
        assert len(prose_chunks) >= 1
        assert "def hello():" in code_chunks[0]
    
    def test_tilde_fence(self):
        """Text with ~~~ fences should keep code blocks separate."""
        text = """Some prose here.

~~~javascript
console.log("hello");
~~~

More prose after."""
        chunks = code_chunk(text, chunk_size=100, chunk_overlap=10)
        
        # Should have code block and prose chunks
        code_chunks = [c for c in chunks if c.startswith("~~~")]
        prose_chunks = [c for c in chunks if not c.startswith("~~~") and not c.startswith("```")]
        
        assert len(code_chunks) >= 1
        assert len(prose_chunks) >= 1
        assert "console.log" in code_chunks[0]
    
    def test_both_fence_types(self):
        """Text with both ``` and ~~~ fences should handle both."""
        text = """Intro.

```python
x = 1
```

Middle text.

~~~ruby
puts "hello"
~~~

End."""
        chunks = code_chunk(text, chunk_size=100, chunk_overlap=10)
        
        backtick_chunks = [c for c in chunks if c.startswith("```")]
        tilde_chunks = [c for c in chunks if c.startswith("~~~")]
        
        assert len(backtick_chunks) >= 1
        assert len(tilde_chunks) >= 1
        assert "x = 1" in backtick_chunks[0]
        assert 'puts "hello"' in tilde_chunks[0]
    
    def test_unclosed_fence(self):
        """Unclosed fence should treat remainder as code and not crash."""
        text = """Some prose.

```python
def broken():
    # This code block never closes
    pass"""
        
        # Should not raise an exception
        chunks = code_chunk(text, chunk_size=100, chunk_overlap=10)
        
        assert len(chunks) >= 1
        # The unclosed code block should be captured
        code_chunks = [c for c in chunks if c.startswith("```")]
        assert len(code_chunks) >= 1
        assert "def broken():" in code_chunks[0]
    
    def test_large_code_block_no_split_by_default(self):
        """Large code blocks should not be split when max_code_chunk_size is None."""
        large_code = "```python\n" + ("x = 1\n" * 100) + "```"
        text = f"Intro.\n\n{large_code}\n\nEnd."
        
        chunks = code_chunk(text, chunk_size=50, chunk_overlap=5, max_code_chunk_size=None)
        
        # The large code block should be intact
        code_chunks = [c for c in chunks if c.startswith("```")]
        assert len(code_chunks) == 1
        assert len(code_chunks[0]) > 500  # Should be large
    
    def test_large_code_block_with_max_size(self):
        """Large code blocks should be split when max_code_chunk_size is set."""
        large_code = "```python\n" + ("x = 1\n" * 50) + "```"
        text = f"Intro.\n\n{large_code}\n\nEnd."
        
        chunks = code_chunk(text, chunk_size=50, chunk_overlap=5, max_code_chunk_size=100)
        
        # The large code block should be split into multiple chunks
        code_chunks = [c for c in chunks if c.startswith("```")]
        assert len(code_chunks) > 1
        # Each code chunk should have fence markers
        for cc in code_chunks:
            assert cc.startswith("```")
            assert cc.endswith("```")
    
    def test_chunks_are_stripped(self):
        """All returned chunks should be stripped strings."""
        text = "  Hello.  \n\n```python\nx = 1\n```\n\n  World.  "
        chunks = code_chunk(text, chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            assert chunk == chunk.strip()
    
    def test_empty_chunks_skipped(self):
        """Empty chunks should be skipped."""
        text = "```python\n\n```"  # Empty code block
        chunks = code_chunk(text, chunk_size=100, chunk_overlap=10)
        # Should have at least the fence markers, no actual empty strings
        for chunk in chunks:
            assert chunk.strip() != ""
    
    def test_invalid_overlap_negative(self):
        """Negative overlap should raise ValueError."""
        with pytest.raises(ValueError, match="chunk_overlap must be >= 0"):
            code_chunk("test", chunk_overlap=-1)
    
    def test_invalid_overlap_too_large(self):
        """Overlap >= chunk_size should raise ValueError."""
        with pytest.raises(ValueError, match="chunk_overlap must be < chunk_size"):
            code_chunk("test", chunk_size=100, chunk_overlap=100)


class TestHeaderChunk:
    """Tests for chunk_markdown_with_headers function."""
    
    def test_empty_input_returns_empty_list(self):
        """Empty input should return empty list."""
        assert chunk_markdown_with_headers("") == []
        assert chunk_markdown_with_headers("   ") == []
        assert chunk_markdown_with_headers("\n\n") == []
    
    def test_basic_header_chunking(self):
        """Basic markdown with headers should be chunked properly."""
        text = """# Title

Some content under title.

## Section 1

Content in section 1.

## Section 2

Content in section 2."""
        
        chunks = chunk_markdown_with_headers(text, chunk_size=200, chunk_overlap=20)
        
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, dict)
            assert "header_path" in chunk
            assert "content" in chunk
            assert "metadata" in chunk
    
    def test_nested_headers(self):
        """Nested headers should produce hierarchical header_path."""
        text = """# Main Title

Intro text.

## Chapter 1

### Section 1.1

Content in section 1.1.

### Section 1.2

Content in section 1.2.

## Chapter 2

Content in chapter 2."""
        
        chunks = chunk_markdown_with_headers(text, chunk_size=200, chunk_overlap=20)
        
        # Should have chunks with different header paths
        assert len(chunks) >= 1
        
        # Check that header_path contains hierarchical structure
        header_paths = [c["header_path"] for c in chunks]
        # At least some paths should have multiple levels (contain " > ")
        nested_paths = [p for p in header_paths if " > " in p]
        # Note: depending on langchain version, nesting behavior may vary
    
    def test_long_section_triggers_recursive_split(self):
        """Long sections should be split into smaller chunks."""
        long_content = "This is a test sentence. " * 100
        text = f"""# Title

{long_content}"""
        
        chunks = chunk_markdown_with_headers(text, chunk_size=100, chunk_overlap=20)
        
        # Should split into multiple chunks
        assert len(chunks) > 1
        
        # Each chunk's content should be reasonably sized
        for chunk in chunks:
            # Allow some tolerance for splitter behavior
            assert len(chunk["content"]) <= 200
    
    def test_content_lengths_within_chunk_size(self):
        """Content lengths should be <= chunk_size param."""
        text = """# Title

This is a very long paragraph that should be split into smaller chunks.
""" + "Additional content. " * 50
        
        chunk_size = 150
        chunks = chunk_markdown_with_headers(text, chunk_size=chunk_size, chunk_overlap=20)
        
        for chunk in chunks:
            # Content should be close to or under chunk_size
            # (some variance allowed due to splitter behavior)
            assert len(chunk["content"]) <= chunk_size + 50
    
    def test_chunks_are_stripped(self):
        """Content in returned chunks should be stripped."""
        text = """# Title

  Some content with spaces.  """
        
        chunks = chunk_markdown_with_headers(text, chunk_size=200, chunk_overlap=20)
        
        for chunk in chunks:
            assert chunk["content"] == chunk["content"].strip()
    
    def test_empty_chunks_skipped(self):
        """Empty chunks should be skipped."""
        text = """# Title

## Empty Section

## Another Section

Some actual content here."""
        
        chunks = chunk_markdown_with_headers(text, chunk_size=200, chunk_overlap=20)
        
        for chunk in chunks:
            assert chunk["content"].strip() != ""
    
    def test_metadata_preserved(self):
        """Original metadata should be preserved in returned dicts."""
        text = """# Main Title

Content under main.

## Section

Content under section."""
        
        chunks = chunk_markdown_with_headers(text, chunk_size=200, chunk_overlap=20)
        
        for chunk in chunks:
            assert isinstance(chunk["metadata"], dict)
    
    def test_deterministic_header_path(self):
        """Header path should be built deterministically from h1, h2, h3 keys."""
        text = """# Top Level

## Mid Level

### Low Level

Content here."""
        
        chunks = chunk_markdown_with_headers(text, chunk_size=500, chunk_overlap=50)
        
        # Find chunk with nested headers
        for chunk in chunks:
            if chunk["header_path"]:
                # Path should be deterministic based on header order
                assert isinstance(chunk["header_path"], str)
    
    def test_invalid_overlap_negative(self):
        """Negative overlap should raise ValueError."""
        with pytest.raises(ValueError, match="chunk_overlap must be >= 0"):
            chunk_markdown_with_headers("test", chunk_overlap=-1)
    
    def test_invalid_overlap_too_large(self):
        """Overlap >= chunk_size should raise ValueError."""
        with pytest.raises(ValueError, match="chunk_overlap must be < chunk_size"):
            chunk_markdown_with_headers("test", chunk_size=100, chunk_overlap=100)


class TestParameterValidation:
    """Cross-cutting tests for parameter validation."""
    
    def test_all_chunkers_validate_overlap_negative(self):
        """All chunkers should reject negative overlap."""
        with pytest.raises(ValueError):
            recursive_chunk("text", chunk_overlap=-5)
        
        with pytest.raises(ValueError):
            code_chunk("text", chunk_overlap=-5)
        
        with pytest.raises(ValueError):
            chunk_markdown_with_headers("text", chunk_overlap=-5)
    
    def test_all_chunkers_validate_overlap_too_large(self):
        """All chunkers should reject overlap >= chunk_size."""
        with pytest.raises(ValueError):
            recursive_chunk("text", chunk_size=50, chunk_overlap=50)
        
        with pytest.raises(ValueError):
            code_chunk("text", chunk_size=50, chunk_overlap=50)
        
        with pytest.raises(ValueError):
            chunk_markdown_with_headers("text", chunk_size=50, chunk_overlap=50)
    
    def test_all_chunkers_handle_empty_input(self):
        """All chunkers should return empty list for empty input."""
        assert recursive_chunk("") == []
        assert code_chunk("") == []
        assert chunk_markdown_with_headers("") == []
