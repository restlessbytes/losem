#!/usr/bin/env python3

from losem.models import KeywordConnector
import json
import argparse
import typing

from losem.models import (
    DEFAULT_EMBEDDING_MODEL,
    TextChunk,
    SearchOrder,
    TextDocumentType,
)
from losem.ingestion import IngestionPipeline
from losem.search import KeywordSearch, SemanticSearch

OUTPUT_FORMATS = ["text", "json"]


def is_ingest(args: argparse.Namespace) -> bool:
    return args.command == "ingest"


def is_keyword_search(args: argparse.Namespace) -> bool:
    return args.command == "search" and args.by_keywords


def is_semantic_search(args: argparse.Namespace) -> bool:
    return args.command == "search" and args.by_similarity


def output_search_results(output_format: str, results: list[TextChunk]):
    output_format = output_format.strip().lower()
    # NOTE This function assumes that results are already ordered according to CLI params
    # (i.e. ordered by enumeration or score)
    if output_format == "text":
        for result in results:
            print(result.content)
    elif output_format == "json":
        print(json.dumps([result.to_dict() for result in results], indent=2))
    else:
        raise ValueError(
            f"Invalid output format - expected {OUTPUT_FORMATS} but was {output_format}"
        )


def add_ingest_subparser(subparsers):
    ingest_parser = subparsers.add_parser("ingest", help="Handle data ingestion")
    ingest_parser.add_argument(
        "--database", "-db", help="Path to DuckDB file", required=True
    )
    ingest_parser.add_argument(
        "--type",
        "-t",
        choices=typing.get_args(TextDocumentType),
        default="text",
        help="Filter files by file type (optional)",
    )
    ingest_parser.add_argument(
        "--embedding-model",
        "-emb",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Name of the embedding model (optional; defaults to {DEFAULT_EMBEDDING_MODEL})",
    )
    ingest_parser.add_argument(
        "files_or_folder",
        nargs="+",
        help="Files or folder(s) to be loaded and 'ingested'. Note that `losem` does not recurse into subfolders!",
    )


def add_search_subparser(subparsers):
    search_parser = subparsers.add_parser("search", help="Handle searching")
    search_parser.add_argument(
        "--database", "-db", help="Path to DuckDB file", required=True
    )
    search_parser.add_argument(
        "--by-keywords",
        "-kw",
        action="store_true",
        help="Set search method to keyword search",
    )
    search_parser.add_argument(
        "--by-similarity",
        "-sim",
        action="store_true",
        help="Set search method to similarity search",
    )
    search_parser.add_argument(
        "--keyword-connector",
        "-kwc",
        choices=typing.get_args(KeywordConnector),
        default="OR",
        help="How keywords are connected (OR/AND) - only used in keyword searches",
    )
    search_parser.add_argument(
        "--limit",
        "-lim",
        type=int,
        default=0,
        help="Maximum number of results to return (0 = return everything)",
    )
    search_parser.add_argument(
        "--order-by",
        "-ord",
        choices=typing.get_args(SearchOrder),
        default="enumeration",
        help=(
            "Order results by enumeration or score "
            "(keyword search will always use enumeration while "
            "semantic search defaults to enumeration)"
        ),
    )
    search_parser.add_argument(
        "--format",
        "-fmt",
        choices=OUTPUT_FORMATS,
        default="text",
        help=f"Output format (one of {OUTPUT_FORMATS})",
    )
    search_parser.add_argument(
        "query",
        nargs="+",
        help="Search query (keywords for keyword search, text for semantic search)",
    )


def main():
    parser = argparse.ArgumentParser(
        description="losem - LOcal Semantic (and keyword) search tool"
    )
    parser.add_argument(
        "--verbose", "-v", help="Display logs and messages", action="store_true"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # "ingest" command
    add_ingest_subparser(subparsers)

    # "search" command
    add_search_subparser(subparsers)

    args = parser.parse_args()

    if is_ingest(args):
        pipeline = IngestionPipeline(
            files_or_folders=args.files_or_folder,
            document_type=args.type,
            database=args.database,
            embedding_model=args.embedding_model,
            is_verbose=args.verbose,
        )
        pipeline.run()
    elif is_keyword_search(args):
        if args.order_by != "enumeration" and args.verbose:
            print(
                "[WARNING] Keyword search will always use 'enumeration' to order results"
            )
        pipeline = KeywordSearch(
            database=args.database,
            keywords=args.query,
            keyword_connector=args.keyword_connector.upper(),
            is_verbose=args.verbose,
        )
        results = pipeline.run()
        output_search_results(output_format=args.format, results=results)
    elif is_semantic_search(args):
        pipeline = SemanticSearch(
            database=args.database,
            query=" ".join(args.query),
            limit=args.limit,
            order_by=args.order_by,
            is_verbose=args.verbose,
        )
        results = pipeline.run()
        output_search_results(output_format=args.format, results=results)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
