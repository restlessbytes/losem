import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from losem.ingestion import IngestionPipeline


def test_ingestion_pipeline_run_success(tmp_path: Path):
    """Test successful execution of IngestionPipeline."""
    # Create a temporary text file
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document for ingestion.")

    # Mock the pipeline stages to avoid actual processing
    with patch("losem.ingestion.DocumentIngestionPipeline") as mock_pipeline_class:
        # Create a mock pipeline instance
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        # Mock the load() -> split() -> embed() -> index() chain
        mock_pipeline.load.return_value.split.return_value.embed.return_value.index.return_value = None

        # Create the ingestion pipeline
        pipeline = IngestionPipeline(
            files_or_folders=[test_file],
            database=tmp_path / "test.db",
            document_type="text",
            embedding_model="mxbai-embed-large:latest",
            is_verbose=False,
        )

        # Run the pipeline
        result = pipeline.run()

        # Verify results
        assert len(result) == 1
        mock_pipeline_class.assert_called_once_with(
            file=test_file,
            database=tmp_path / "test.db",
            embedding_model="mxbai-embed-large:latest",
            is_verbose=False,
        )
        mock_pipeline.load.assert_called_once()


def test_ingestion_pipeline_run_failure(tmp_path: Path):
    """Test handling of pipeline failure during execution."""
    # Create a temporary text file
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document for ingestion.")

    # Mock the pipeline to raise an exception
    with patch("losem.ingestion.DocumentIngestionPipeline") as mock_pipeline_class:
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.load.side_effect = Exception("Processing failed")

        # Create the ingestion pipeline
        pipeline = IngestionPipeline(
            files_or_folders=[test_file],
            database=tmp_path / "test.db",
            document_type="text",
            embedding_model="mxbai-embed-large:latest",
            is_verbose=False,
        )

        # Run the pipeline - should not raise but log error
        result = pipeline.run()

        # Verify that no files were successfully processed
        assert len(result) == 0


def test_ingestion_pipeline_filter_by_document_type(tmp_path: Path):
    """Test filtering of files by document type."""
    # Create test files
    txt_file = tmp_path / "test.txt"
    pdf_file = tmp_path / "test.pdf"
    txt_file.write_text("text content")
    pdf_file.write_text(
        "pdf content"
    )  # This won't be read as PDF but that's ok for this test

    pipeline = IngestionPipeline(
        files_or_folders=[txt_file, pdf_file],
        database=tmp_path / "test.db",
        document_type="text",  # Only text files
        embedding_model="mxbai-embed-large:latest",
        is_verbose=False,
    )

    # Test filtering
    filtered_files = pipeline.filter_by_document_type([txt_file, pdf_file])

    assert len(filtered_files) == 1
    assert filtered_files[0] == txt_file


def test_ingestion_pipeline_read_files(tmp_path: Path):
    """Test reading files from paths including directories."""
    # Create directory and files
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()

    file1 = test_dir / "file1.txt"
    file2 = test_dir / "file2.txt"
    file1.write_text("content1")
    file2.write_text("content2")

    # Create a non-file path
    not_a_file = tmp_path / "not_a_file"

    pipeline = IngestionPipeline(
        files_or_folders=[test_dir, not_a_file],
        database=tmp_path / "test.db",
        document_type="text",
        embedding_model="mxbai-embed-large:latest",
        is_verbose=False,
    )

    # Should raise ValueError for non-file path
    with pytest.raises(ValueError):
        pipeline.read_files()
