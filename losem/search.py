from losem.models import ConditionalMessage
import duckdb

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic.types import NonNegativeInt

from langchain_ollama import OllamaEmbeddings

from losem.models import TextChunk, KeywordConnector, SearchOrder
from losem.sql_statements import (
    SELECT_TEXT_CHUNKS,
    SELECT_EMBEDDINGS,
    SELECT_MODEL_NAME,
)


def convert_query_results_to_text_chunks(results: list[tuple]) -> list[TextChunk]:
    """
    Convert database query results to a list of TextChunk objects.

    Args:
        results: List of tuples containing data from the following columns:
                   document_id, enumeration, content, filename, filepath, data_type

    Returns:
        List of TextChunk objects populated with the data from the query results.
    """
    text_chunks: list[TextChunk] = []
    for row in results:
        text_chunk = TextChunk(
            document_id=row[0],
            id=row[1],
            enumeration=row[2],
            content=row[3],
            filename=row[4],
            filepath=Path(row[5]),
            data_type=row[6],
        )
        text_chunks.append(text_chunk)
    return text_chunks


class Search(ABC, ConditionalMessage):
    """Abstract base class for searches."""

    @abstractmethod
    def run(self) -> list[TextChunk]:
        """Execute the search operation.

        Returns:
            List of TextChunk objects that match the search criteria.
        """
        pass

    @abstractmethod
    def generate_sql_search_statement(self) -> str | tuple[str, list[str]]:
        """Generate the SQL statement for the search operation.

        Returns:
            Either a SQL string or a tuple of (sql_query, query_parameters)
        """
        pass


class KeywordSearch(Search):
    """Performs keyword-based searches over text chunks.

    This search type looks for specific keywords within the text content of documents
    using SQL LIKE operations. It supports both 'OR' and 'AND' conjunctions between keywords.
    """

    database: Path
    keywords: list[str]
    keyword_connector: KeywordConnector = "OR"

    def run(self) -> list[TextChunk]:
        """
        Perform a keyword search on the 'embeddings' database.

        Returns:
            List of text snippets matching the search criteria.
        """
        self.print_message(f"Performing keyword search for '{self.keywords}'")
        query, query_parameters = self.generate_sql_search_statement()
        with duckdb.connect(self.database) as conn:
            results: list[tuple] = conn.execute(query, query_parameters).fetchall()
            text_chunks: list[TextChunk] = convert_query_results_to_text_chunks(results)
            self.print_message(f"Found {len(text_chunks)} matching text snippets.")
            return text_chunks

    def generate_sql_search_statement(self) -> str | tuple[str, list[str]]:
        """
        Generate SQL statement for keyword search.

        Returns:
            Tuple of (sql_query, query_parameters) for the keyword search.
        """
        if not self.keywords:
            raise ValueError("Keywords list cannot be empty")

        query_parameters = [f"%{k}%" for k in self.keywords]
        like_conditions = ["content LIKE ?"] * len(query_parameters)
        separator = f" {self.keyword_connector.upper()} "
        query = f"{SELECT_TEXT_CHUNKS} WHERE " + separator.join(like_conditions)
        query = f"{query} ORDER BY enumeration ASC"
        return query, query_parameters


class SemanticSearch(Search):
    """Performs semantic searches using vector similarity.

    This search type converts a query into an embedding vector and finds
    documents with the most similar content using cosine similarity.
    """

    database: Path
    query: str
    limit: NonNegativeInt = 0
    order_by: SearchOrder = "enumeration"

    def run(self) -> list[TextChunk]:
        """
        Perform semantic search on a DuckDB database file.

        Returns:
            List of text snippets matching the search criteria.
        """
        self.print_message(f"Performing semantic search for query '{self.query}'")
        search_statement = self.generate_sql_search_statement()
        with duckdb.connect(self.database) as conn:
            results: list[tuple] = conn.execute(search_statement).fetchall()
            text_chunks: list[TextChunk] = convert_query_results_to_text_chunks(results)
            # NOTE 'order by enumeration' needs to be applied here because DuckDB returns results
            # ordered by 'similarity score'
            if self.order_by == "enumeration":
                text_chunks.sort(key=lambda item: item.enumeration)
            self.print_message(f"Found {len(text_chunks)} matching text snippets.")
            return text_chunks

    def generate_sql_search_statement(self) -> str | tuple[str, list[str]]:
        """
        Generate a (DuckDB specific) SQL statement for semantic search.

        Returns:
            DuckDB specific SQL statement that performs semantic search using cosine similarity.
        """
        model_name = self._read_embedding_model()
        vector = OllamaEmbeddings(model=model_name).embed_query(self.query)
        vector_length = len(vector)
        embedded_query = f"{vector}::FLOAT[{vector_length}]"
        order_by_statement = f"ORDER BY array_cosine_distance(vector, {embedded_query})"
        result_statement = f"{SELECT_EMBEDDINGS} {order_by_statement}"
        if 0 < self.limit:
            return f"{result_statement} LIMIT {self.limit}"
        return result_statement

    def _read_embedding_model(self) -> str:
        """Read the embedding model name from the database.

        Returns:
            The name of the embedding model used for indexing.

        Raises:
            ValueError: If no embedding model is found in the database.
        """
        with duckdb.connect(self.database) as conn:
            result = conn.execute(SELECT_MODEL_NAME).fetchone()
            if result:
                return result[0]
            raise ValueError("No embedding model found in database")
