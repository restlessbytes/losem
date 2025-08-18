from pydantic.types import FilePath
from losem.models import DEFAULT_CHUNK_OVERLAP, TextDocumentType, ConditionalMessage
from losem.models import DEFAULT_CHUNK_SIZE
import duckdb
import more_itertools as mit

from pathlib import Path
from losem.models import DEFAULT_EMBEDDING_MODEL
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
)
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import NLTKTextSplitter
from losem.models import (
    TextDocument,
    TextChunk,
    TextEmbedding,
    EmbeddingVector,
    generate_document_id,
)
from losem.sql_statements import (
    CREATE_EMBEDDINGS_TABLE,
    CREATE_EMBEDDING_MODELS_TABLE,
    INSERT_CHUNKS,
    INSERT_EMBEDDING_MODEL,
    CREATE_HNSW_INDEX,
)


def is_pdf(path: Path) -> bool:
    """Check if `path` points to a PDF file.

    File type is determined by file extension.

    Returns:
        `True` if file is a PDF; `False` otherwise.
    """
    return path.suffix.lower() == ".pdf"


def is_text(path: Path) -> bool:
    """Check if 'path' points to a text file.

    File type is determined by file extension and covers plain text as well as markdown.

    Returns:
        `True` if file is a text file; `False` otherwise.
    """
    # Markdown file types: https://superuser.com/a/285878
    file_extensions = {
        ".txt",
        ".text",
        ".md",
        ".mkd",
        ".mdwn",
        ".mdown",
        ".mdtxt",
        ".mdtext",
        ".markdown",
        ".text",
    }
    return path.suffix.lower() in file_extensions


class PipelineStage(ConditionalMessage):
    file: FilePath
    database: Path
    embedding_model: str = DEFAULT_EMBEDDING_MODEL


class IndexingStage(PipelineStage):
    """Stores text embeddings into a DuckDB table with HNSW indexing enabled.

    This stage takes embedded text chunks and stores them in a DuckDB database
    along with their vector representations. It also creates an HNSW index for
    enabling semantic search operations.
    """

    text_embeddings: list[TextEmbedding]

    def index(self):
        """Store text embeddings in a DuckDB table with HNSW indexing enabled.

        Inserts all text embeddings into the database, creates necessary tables,
        stores embedding model information, and builds an HNSW index for efficient
        semantic search operations.

        The process includes:
        1. Creating database tables for embeddings and model information
        2. Inserting all text chunks with their embeddings in batches
        3. Storing the embedding model name
        4. Creating an HNSW index ready for doing cosine similarity searches

        Raises:
            ValueError: If multiple embedding models are detected during insertion
        """
        with duckdb.connect(self.database) as connection:
            # Create tables
            connection.execute(CREATE_EMBEDDINGS_TABLE)
            connection.execute(CREATE_EMBEDDING_MODELS_TABLE)
            # Insert text embeddings (text chunks + embedding vectors)
            records = [embedding.to_tuple() for embedding in self.text_embeddings]
            for num, batch in enumerate(mit.chunked(records, n=500), start=1):
                self.print_message(f"Now inserting batch no. {num} ...")
                connection.executemany(INSERT_CHUNKS, batch)
            # Collect embedding model names (fingers crossed that there's only a single model name)
            embedding_model_names: set[str] = {
                embedding.embedding_model for embedding in self.text_embeddings
            }
            if (models_count := len(embedding_model_names)) != 1:
                raise ValueError(
                    f"Expected only a single embedding model but found {models_count}"
                )
            # Insert the pairs into the document_embedding_models table
            connection.execute(INSERT_EMBEDDING_MODEL, tuple(embedding_model_names))
            self.print_message("Creating HNSW index ...")
            connection.execute(CREATE_HNSW_INDEX)
            self.print_message("HNSW index created.")


class EmbeddingStage(PipelineStage):
    """Generates embeddings for text chunks using an embedding model through ollama.

    This stage takes a list of text chunks and converts them into vector embeddings
    that can be used for semantic search. It serves as the bridge between the
    splitting stage and the indexing stage in the ingestion pipeline.
    """

    text_chunks: list[TextChunk]

    def embed(self) -> IndexingStage:
        """Generate embeddings for all text chunks.

        Uses an embedding model to convert each text chunk into a vector representation.
        The embeddings preserve semantic meaning and enable similarity searches.

        Returns:
            An IndexingStage instance containing the embedded text chunks representing
             the next stage in the pipeline (embed -> index).
        """
        model = OllamaEmbeddings(model=self.embedding_model)
        embeddings: list[EmbeddingVector] = model.embed_documents(
            [chunk.content for chunk in self.text_chunks]
        )
        if not (len(self.text_chunks) == len(embeddings)):
            raise Exception(
                f"Number of chunks differs from number of embeddings: {len(self.text_chunks)} vs {len(embeddings)}"
            )
        # NOTE This works only because `model.embed_documents` preserves the order of chunks (afaik)
        results: list = []
        for chunk, vector in zip(self.text_chunks, embeddings):
            if not isinstance(chunk, TextChunk):
                raise ValueError(
                    f"Embedding error: TextChunk expected but {type(chunk)} found"
                )
            results.append(
                TextEmbedding.from_text_chunk(
                    text_chunk=chunk,
                    vector=vector,
                    embedding_model=self.embedding_model,
                )
            )
        return IndexingStage(
            file=self.file,
            database=self.database,
            embedding_model=self.embedding_model,
            text_embeddings=results,
            is_verbose=self.is_verbose,
        )


class SplittingStage(PipelineStage):
    """Represents the document splitting stage in the ingestion pipeline.

    This stage takes a loaded TextDocument and splits it into smaller text chunks
    based on the configured chunk size and overlap parameters. It serves as the
    bridge between the loading stage and the embedding stage in the pipeline.
    """

    text_document: TextDocument

    def split(self) -> EmbeddingStage:
        """Split documents into (smaller) text chunks.

        Uses `NLTKTextSplitter` to divide the document content into smaller
        chunks of around 1000 to 1200 tokens.

        Each chunk is associated with its original document and an enumeration number
        that indicates its position relative to other chunks in the original document.

        Returns:
            An EmbeddingStage instance containing the split text chunks representing
            the next stage in the pipeline (split -> embed).
        """
        splitter = NLTKTextSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP
        )
        text_chunks: list[TextChunk] = []
        for enumeration, chunk in enumerate(
            splitter.split_text(self.text_document.content), start=1
        ):
            text_chunk = TextChunk.from_text_document(
                content=chunk, enumeration=enumeration, document=self.text_document
            )
            text_chunks.append(text_chunk)
        return EmbeddingStage(
            file=self.file,
            database=self.database,
            embedding_model=self.embedding_model,
            text_chunks=text_chunks,
            is_verbose=self.is_verbose,
        )


class DocumentIngestionPipeline(PipelineStage):
    """Handles the complete ingestion process for a single document.

    This class represents a pipeline that processes one individual document through
    all stages of ingestion: loading, splitting, embedding, and indexing.
    It is designed to be used by the `IngestionPipeline` orchestrator.
    """

    def load(self) -> SplittingStage:
        """Load file from disk and turn it into a `TextDocument` object.

        Returns:
            A `SplittingStage` instance containing the loaded document representing
             the next stage in the pipeline (load -> split).
        """
        if is_pdf(path=self.file):
            document = self.load_pdf_document()
        else:
            document = self.load_text_document()
        return SplittingStage(
            file=self.file,
            database=self.database,
            embedding_model=self.embedding_model,
            text_document=document,
            is_verbose=self.is_verbose,
        )

    def load_text_document(self) -> TextDocument:
        """Load a text document from disk.

        Reads the content of the file and creates a `TextDocument` object
        with appropriate metadata.

        Returns:
            A `TextDocument` instance containing the file's content and metadata.
        """
        content = self.file.read_text(encoding="utf-8")
        document_id = generate_document_id(content=content, filepath=str(self.file))
        return TextDocument(
            id=document_id, content=content, filename=self.file.name, filepath=self.file
        )

    def load_pdf_document(self) -> TextDocument:
        """Load a PDF document from disk.

        Uses the UnstructuredPDFLoader to extract text content from the PDF file
        and converts it into a `TextDocument` object.

        Returns:
            A `TextDocument` instance containing the PDF's content and metadata.
        """
        loader = UnstructuredPDFLoader(self.file)
        document = loader.load()[0]
        return TextDocument.from_langchain_document(document, document_type="pdf")


class IngestionPipeline(ConditionalMessage):
    """Orchestrates the end-to-end document ingestion process for multiple files.

    This class serves as a manager that coordinates the ingestion of multiple
    documents into a DuckDB database. For each individual document, it creates and
    runs a separate `DocumentIngestionPipeline` which handles the actual
    loading, splitting, embedding, and storage operations.
    """

    files_or_folders: list[Path]
    database: Path
    document_type: TextDocumentType
    embedding_model: str

    def filter_by_document_type(self, files: list[Path]) -> list[Path]:
        """Filter a list of files based on the specified document type.

        Args:
            files: A list of file paths to filter.

        Returns:
            A new list containing only the files that match the specified document type.
            If document_type is None, returns all files (unfiltered).
        """
        if not self.document_type:
            return files
        filtered: list[Path] = []
        for file in files:
            if self.document_type == "pdf" and not is_pdf(file):
                continue
            if self.document_type == "text" and not is_text(file):
                continue
            filtered.append(file)
        return filtered

    def read_files(self) -> list[Path]:
        """Read all files from the specified paths (field `IngestionPipeline.files_or_folders`).

        If a path is a directory, all files in that directory are included.
        If a path is a file, it's included directly.

        Returns:
            A list of file paths to process.

        Raises:
            ValueError: If any path is neither a file nor a folder.
        """
        results: list[Path] = []
        for path in self.files_or_folders:
            if path.is_file():
                results.append(path)
            elif path.is_dir():
                results += list(file for file in path.iterdir() if file.is_file())
            else:
                raise ValueError(f"Neither file nor folder: {path}")
        return self.filter_by_document_type(files=results)

    def prepare_document_ingestion_pipelines(self) -> list[DocumentIngestionPipeline]:
        """Prepare individual ingestion pipelines for each file.

        Creates a `DocumentIngestionPipeline` for each file identified by `read_files()`.
        Failed pipeline setups are logged but do not stop the process.

        Returns:
            A list of `DocumentIngestionPipeline` objects ready to be executed.
        """
        pipelines: list[DocumentIngestionPipeline] = []
        for file in self.read_files():
            try:
                pipeline = DocumentIngestionPipeline(
                    file=file,
                    database=self.database,
                    embedding_model=self.embedding_model,
                    is_verbose=self.is_verbose,
                )
                pipelines.append(pipeline)
            except Exception as ex:
                self.print_message(
                    f"Setting up pipeline for {file} failed -- reason: {ex}"
                )
        return pipelines

    def run(self):
        """Execute the document ingestion pipeline.

        Processes each file sequentially through the following stages:
        1. Load the document
        2. Split into text chunks
        3. Generate embeddings for the chunks
        4. Store both chunks and embeddings in DuckDB

        NOTE: Documents are processed sequentially, one at a time.

        Returns:
            A list containing the file paths for the files that were successfully ingested.
        """
        pipelines = self.prepare_document_ingestion_pipelines()
        results: list[Path] = []
        for pipeline in pipelines:
            file = pipeline.file
            try:
                pipeline.load().split().embed().index()
                results.append(file)
            except Exception as ex:
                self.print_message(f"Failed to run pipeline '{file}' -- reason: {ex}")
        return results
