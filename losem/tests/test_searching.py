from pathlib import Path

import pytest

from unittest.mock import patch
from losem.models import DEFAULT_EMBEDDING_MODEL, generate_document_id, KeywordConnector
from losem.search import (
    convert_query_results_to_text_chunks,
    KeywordSearch,
    SemanticSearch,
)
from losem.sql_statements import SELECT_TEXT_CHUNKS, SELECT_EMBEDDINGS


@pytest.mark.parametrize(
    "keywords,conjunction,expected_query,expected_params",
    [
        # Test with single keyword (OR conjunction)
        (
            ["hello"],
            "OR",
            f"{SELECT_TEXT_CHUNKS} WHERE content LIKE ? ORDER BY enumeration ASC",
            ["%hello%"],
        ),
        # Test with multiple keywords (OR conjunction)
        (
            ["hello", "world"],
            "OR",
            f"{SELECT_TEXT_CHUNKS} WHERE content LIKE ? OR content LIKE ? ORDER BY enumeration ASC",
            ["%hello%", "%world%"],
        ),
        # Test with multiple keywords (AND conjunction)
        (
            ["hello", "world"],
            "AND",
            f"{SELECT_TEXT_CHUNKS} WHERE content LIKE ? AND content LIKE ? ORDER BY enumeration ASC",
            ["%hello%", "%world%"],
        ),
    ],
)
def test_generate_sql_statement_for_keyword_search_normal_cases(
    keywords: list[str], conjunction: KeywordConnector, expected_query, expected_params
):
    keyword_search = KeywordSearch(
        database=Path("dummy.db"),
        keywords=keywords,
        keyword_connector=conjunction,
        is_verbose=True,
    )
    query, params = keyword_search.generate_sql_search_statement()
    assert query == expected_query
    assert params == expected_params


@pytest.mark.parametrize(
    "keywords,conjunction,expected_exception",
    [
        # Test with empty keywords list - should raise ValueError
        ([], "OR", ValueError),
        # Test with invalid conjunction - should raise ValueError
        (["hello"], "XOR", ValueError),
    ],
)
def test_generate_sql_statement_for_keyword_search_error_cases(
    keywords, conjunction: KeywordConnector, expected_exception
):
    with pytest.raises(expected_exception):
        keyword_search = KeywordSearch(
            database=Path("dummy.db"),
            keywords=keywords,
            keyword_connector=conjunction,
            is_verbose=True,
        )
        keyword_search.generate_sql_search_statement()


def test_generate_sql_statement_for_semantic_search():
    # Mock the OllamaEmbeddings to avoid calling ollama
    with patch("losem.search.OllamaEmbeddings") as mock_embeddings, patch.object(
        SemanticSearch, "_read_embedding_model", return_value=DEFAULT_EMBEDDING_MODEL
    ):
        # Set up the mock to return a simple vector
        mock_vector = [1.0, 2.0, 3.0]
        mock_instance = mock_embeddings.return_value
        mock_instance.embed_query.return_value = mock_vector
        mock_semantic_search = SemanticSearch(
            database=Path("dummy.db"), query="test query", is_verbose=True
        )

        # Test the function
        result = mock_semantic_search.generate_sql_search_statement()

        # Verify the result contains expected elements
        expected_embedding = f"{mock_vector}::FLOAT[3]"
        expected_query = f"{SELECT_EMBEDDINGS} ORDER BY array_cosine_distance(vector, {expected_embedding})"
        assert result == expected_query

        # Verify that OllamaEmbeddings was called correctly
        mock_embeddings.assert_called_once_with(model=DEFAULT_EMBEDDING_MODEL)
        mock_instance.embed_query.assert_called_once_with(mock_semantic_search.query)

        # Verify that 'limit' also works
        mock_semantic_search.limit = 5
        result = mock_semantic_search.generate_sql_search_statement()
        expected_query = f"{expected_query} LIMIT 5"
        assert result == expected_query


def test_tuple_to_text_chunk(tmp_path):
    # Mock tuple results that would come from a DuckDB query
    # Create a temporary file to avoid "Path does not point to a file" error
    mock_file = tmp_path / "test.txt"
    mock_file.write_text("Hello world")
    mock_filepath = str(mock_file)
    mock_document_id = generate_document_id("Hello world", mock_filepath)
    mock_results = [
        (
            mock_document_id,  # document_id
            generate_document_id("Hello", mock_filepath),  # (chunk) id
            1,  # enumeration
            "Hello",  # content
            "test.txt",  # filename
            mock_filepath,  # filepath
            "text",  # data_type
            [1.0, 2.0, 3.0],  # vector (not used in TextChunk construction)
        ),
        (
            mock_document_id,
            generate_document_id("World", mock_filepath),
            2,
            "World",
            "test2.txt",
            mock_filepath,
            "text",
            [4.0, 5.0, 6.0],
        ),
    ]

    # Call the function
    result_chunks = convert_query_results_to_text_chunks(mock_results)

    # Verify the results
    assert len(result_chunks) == 2

    # Check first chunk
    mock_result1 = mock_results[0]
    chunk1 = result_chunks[0]
    assert chunk1.document_id == mock_document_id
    assert chunk1.id == mock_result1[1]
    assert chunk1.enumeration == 1
    assert chunk1.content == "Hello"
    assert chunk1.filename == "test.txt"
    assert chunk1.filepath == Path(mock_filepath)
    assert chunk1.data_type == "text"

    # Check second chunk
    mock_result2 = mock_results[1]
    chunk2 = result_chunks[1]
    assert chunk2.document_id == mock_document_id
    assert chunk2.id == mock_result2[1]
    assert chunk2.enumeration == 2
    assert chunk2.content == "World"
    assert chunk2.filename == "test2.txt"
    assert chunk2.filepath == Path(mock_filepath)
    assert chunk2.data_type == "text"
