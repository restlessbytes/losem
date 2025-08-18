from pydantic.types import FilePath
import hashlib

from typing import Literal, TypeAlias, Annotated, override
from pathlib import Path
from pydantic.types import PositiveInt

from langchain_core.documents import Document
from pydantic import BaseModel, AfterValidator


DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large:latest"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

EmbeddingVector: TypeAlias = list[float]
TextDocumentType: TypeAlias = Literal["text", "docx", "pdf"]
KeywordConnector: TypeAlias = Literal["OR", "AND"]
SearchOrder: TypeAlias = Literal["enumeration", "score"]


def generate_document_id(content: str, filepath: str) -> str:
    """Generate MD5 hash ID from content and file path."""
    id_content = f"{filepath}\n{content}"
    return hashlib.md5(id_content.encode(), usedforsecurity=False).hexdigest()


def is_valid_document_id(document_id: str) -> str:
    """Validate that `document_id` represents a valid MD5 hash.

    This function checks if the provided string is a valid MD5 hash by verifying:
    1. The string is exactly 32 characters long
    2. All characters are valid hexadecimal digits (0-9, a-f, A-F)

    Args:
        document_id: A string to validate as an MD5 hash

    Returns:
        The original `document_id` string if it passes validation

    Raises:
        ValueError: If the `document_id` is not a valid 32-character hexadecimal string
    """
    if (size := len(document_id)) != 32:
        raise ValueError(f"Expected document ID to be 32 chars long but was {size}.")
    if not all(c in "0123456789abcdef" for c in document_id.lower()):
        raise ValueError(f"Document ID is not a valid MD5 hash: {document_id}")
    return document_id


class ConditionalMessage(BaseModel):
    is_verbose: bool

    def print_message(self, message: str) -> None:
        """Print message only if `is_verbose` is `True`."""
        if self.is_verbose:
            print(message)


class TextDocument(BaseModel):
    """Represents a simple plain text document with explicit metadata fields.

    This class is called 'TextDocument' in oder to prevent confusions with
    `langchain_core.documents.Document`.
    """

    # Document ID (str, md5 hash)
    id: Annotated[str, AfterValidator(is_valid_document_id)]

    # Content of the document
    content: str

    # File name
    filename: str

    # File path
    filepath: FilePath

    # Data type (plain text (including markdown) or pdf)
    data_type: TextDocumentType = "text"

    @staticmethod
    def from_langchain_document(
        document: Document, document_type: TextDocumentType = "text"
    ) -> "TextDocument":
        """Convert a LangChain Document object to a TextDocument instance.

        The method extracts the content, source metadata, and filename from
        the LangChain document, then generates a document ID based on the
        content and file path.

        Args:
            document: A LangChain Document object containing page content and metadata
            document_type: The expected type of document (default: "text")

        Returns:
            TextDocument: A new TextDocument instance with extracted data

        Raises:
            ValueError: If the document type in metadata doesn't match the expected type
        """
        content = document.page_content
        source = document.metadata["source"]
        filepath = Path(source)
        filename = document.metadata.get("filename", filepath.name)

        type_val: str = document.metadata.get("type", document_type).strip().lower()
        # TODO Document Type 'docx' (for "office" documents)
        match type_val:
            case "pdf":
                data_type: TextDocumentType = "pdf"
            case _:
                data_type: TextDocumentType = "text"

        if data_type != document_type:
            raise ValueError(
                f"Expected document type to be {document_type} but was {data_type}."
            )

        return TextDocument(
            id=generate_document_id(content, str(filepath)),
            content=content,
            filename=filename,
            filepath=filepath,
            data_type=data_type,
        )

    def to_langchain_document(self) -> Document:
        """This method creates and returns a LangChain Document instance with the
        same content and metadata as this TextDocument. It's useful for
        interoperability with LangChain-based workflows and tools.

        Returns:
            Document: A LangChain Document object with the same data as this TextDocument
        """
        return Document(
            id=self.id,
            page_content=self.content,
            metadata={
                "source": str(self.filepath),
                "filename": self.filename,
                "filepath": str(self.filepath.absolute()),
                "data_type": self.data_type,
            },
        )


class TextChunk(BaseModel):
    # ID of the original TextDocument that this chunk belongs to
    document_id: Annotated[str, AfterValidator(is_valid_document_id)]
    # ID of this text chunk (assumed to be md5(<source path> + <content>))
    id: Annotated[str, AfterValidator(is_valid_document_id)]
    # Position of this chunk of text in the original document
    enumeration: PositiveInt
    # Content of the text chunk
    content: str
    # File name of the original document
    filename: str
    # File path to the original document
    filepath: FilePath
    # Data type; must be "text"
    data_type: TextDocumentType = "text"

    @staticmethod
    def from_text_document(
        content: str, enumeration: int, document: TextDocument
    ) -> "TextChunk":
        """Create a new TextChunk instance from document content and metadata.

        This static method also generates a unique chunk ID based on the
        content and file path.

        Args:
            content: The text content for this chunk.
            enumeration: Position of this chunk in the original document.
            document: The source TextDocument this chunk belongs to.

        Returns:
            TextChunk: A new TextChunk instance with the provided data
        """
        chunk_id = generate_document_id(
            content=content, filepath=str(document.filepath)
        )
        return TextChunk(
            document_id=document.id,
            id=chunk_id,
            content=content,
            enumeration=enumeration,
            filename=document.filename,
            filepath=document.filepath,
        )

    def to_dict(self) -> dict[str, str | Path | int]:
        """Turn this `TextChunk` object into a Python `dict`.

        **NOTE** Pydantic's `BaseModel.model_dump_json` returns a JSON
        representation of this class, too, but *as a string*. This method
        returns plain Python `dict` instead.

        Returns:
            Python `dict` representation of this `TextChunk` object.
        """
        dict_result = self.__dict__
        # "TypeError: Object of type PosixPath is not JSON serializable"
        dict_result["filepath"] = str(self.filepath)
        return dict_result


class TextEmbedding(TextChunk):
    vector: EmbeddingVector
    embedding_model: str

    @staticmethod
    @override
    def from_text_document(
        content: str, enumeration: int, document: TextDocument
    ) -> TextChunk:
        """Text embeddings are supposed to be created from text chunks not whole documents.
        It's therefore effectively disabled."""
        raise NotImplementedError("Not applicable to text embeddings.")

    @staticmethod
    def from_text_chunk(
        text_chunk: TextChunk, vector: EmbeddingVector, embedding_model: str
    ) -> "TextEmbedding":
        """Create a TextEmbedding from a TextChunk and its vector representation.

        This static method creates a new TextEmbedding instance by combining
        a TextChunk with its vector representation and embedding model information.

        Args:
            text_chunk: The source TextChunk to convert
            vector: The vector representation of the chunk content
            embedding_model: The name of the embedding model used

        Returns:
            TextEmbedding: A new TextEmbedding instance with combined data
        """
        return TextEmbedding(
            vector=vector, embedding_model=embedding_model, **text_chunk.__dict__
        )

    def to_tuple(self) -> tuple:
        """Return a tuple representation of this TextEmbedding instance.

        Useful for storing it in DuckDB.

        Returns:
            tuple: A tuple containing (document_id, id, enumeration, content, filename,
                   filepath, data_type, vector)
        """
        return (
            self.document_id,
            self.id,
            self.enumeration,
            self.content,
            self.filename,
            str(self.filepath),
            self.data_type,
            self.vector,
        )
