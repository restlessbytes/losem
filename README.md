# `losem` - your friendly, local vector search engine. 

`losem` is a CLI tool to perform semantic (_and keyword_) search on local documents (text, Markdown, PDFs), powered by
[LangChain](https://www.langchain.com/) as well as [DuckDb's new vector similarity search extension](https://duckdb.org/docs/stable/core_extensions/vss)

It’s designed with two core workflows in mind:

1. Local document search (_semantic & keyword_)
2. Context provisioning for local LLMs (_e.g. via_ `losem search ... | ollama run ...`)

## Features

- **Semantic Search**: Find relevant content based on meaning, not just keywords.
- **Keyword Search**: Perform standard full-text search.
- **Local Storage**: Documents and embeddings are stored in a local DuckDB.
- **LLM Context**: Provide contextual information to local LLMs.

## Getting started

### Requirements

Make sure you have these programs installed:

* Python 3.12+
* Ollama + embedding model [mxbai-embed-large](https://ollama.com/library/mxbai-embed-large)
* DuckDB (`pip install duckdb`)

### Installation

There's no official `losem` Python package yet, so the easiest way to install it is to cloen the repo
and then create a symlink from `main.py` into your user's `bin/` folder: 

```bash
$ cd ~
$ git clone git@github.com:restlessbytes/losem.git
$ chmod +x losem/main.py
$ ln -s ~/losem/main.py ~/.local/bin/losem
```

**NOTE** This assumes your bin folder is in ~/.local/, though this location may vary depending on 
your Linux distribution.

## Usage

### Ingest Documents

`losem ingest` allows you to insert documents (_text files, markdown files or PDFs_) into a local DuckDB:

```shell
losem ingest notes.txt -db my_docs.db file1.txt file2.txt
```

### Search Documents

After you've added documents to the database, `losem search` lets you perform **keyword** or **semantic search** on them:

1. Keyword Search

```shell
losem search -db my_docs.db --by-keywords 'cat' 'food' 
```

2.Semantic Search

```shell
losem search -db my_docs.db --by-similarity --limit 20 "What do cats like to eat?"
```

## Use with Ollama

As noted earlier, `losem` was designed with local LLMs in mind. For example, search results can be made 
available through piping, like this:

```shell
losem search --database cat_info.db -sim "What do cats like to eat?" | ollama run qwen3 "Tell me what cats like to eat based on the following information: "
```

## How it works

Loading and searching documents is basically a three-stage process:

1. Ingest: Documents → chunks → embedded (mxbai-embed-large) → DuckDB storage
2. Index: HNSW index for fast vector search
3. Search:
   * Semantic: Query embedding → vector comparison
   * Keyword: DuckDB SQL full-text matching
