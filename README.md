[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/superlinear-ai/raglite) [![Open in GitHub Codespaces](https://img.shields.io/static/v1?label=GitHub%20Codespaces&message=Open&color=blue&logo=github)](https://github.com/codespaces/new/superlinear-ai/raglite)

# 🥤 RAGLite

RAGLite is a Python package for Retrieval-Augmented Generation (RAG) with PostgreSQL or SQLite.

## Features

##### Configurable

- 🧠 Choose any LLM provider with [LiteLLM](https://github.com/BerriAI/litellm), including local [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) models
- 💾 Choose either [PostgreSQL](https://github.com/postgres/postgres) or [SQLite](https://github.com/sqlite/sqlite) as a keyword & vector search database
- 🥇 Choose any reranker with [rerankers](https://github.com/AnswerDotAI/rerankers), including multi-lingual [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) as the default

##### Fast and permissive

- ❤️ Only lightweight and permissive open source dependencies (e.g., no [PyTorch](https://github.com/pytorch/pytorch) or [LangChain](https://github.com/langchain-ai/langchain))
- 🚀 Acceleration with Metal on macOS, and CUDA on Linux and Windows

##### Unhobbled

- 📖 PDF to Markdown conversion on top of [pdftext](https://github.com/VikParuchuri/pdftext) and [pypdfium2](https://github.com/pypdfium2-team/pypdfium2)
- 🧬 Multi-vector chunk embedding with [late chunking](https://weaviate.io/blog/late-chunking) and [contextual chunk headings](https://d-star.ai/solving-the-out-of-context-chunk-problem-for-rag)
- ✂️ Optimal [level 4 semantic chunking](https://medium.com/@anuragmishra_27746/five-levels-of-chunking-strategies-in-rag-notes-from-gregs-video-7b735895694d) by solving a [binary integer programming problem](https://en.wikipedia.org/wiki/Integer_programming)
- 🔍 [Hybrid search](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) with the database's native keyword & vector search ([tsvector](https://www.postgresql.org/docs/current/datatype-textsearch.html)+[pgvector](https://github.com/pgvector/pgvector), [FTS5](https://www.sqlite.org/fts5.html)+[sqlite-vec](https://github.com/asg017/sqlite-vec)[^1])
- 🌀 Optimal [closed-form linear query adapter](src/raglite/_query_adapter.py) by solving an [orthogonal Procrustes problem](https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem)

##### Extensible

- ✍️ Optional conversion of any input document to Markdown with [Pandoc](https://github.com/jgm/pandoc)
- ✅ Optional evaluation of retrieval and generation performance with [Ragas](https://github.com/explodinggradients/ragas)

[^1]: We use [PyNNDescent](https://github.com/lmcinnes/pynndescent) until [sqlite-vec](https://github.com/asg017/sqlite-vec) is more mature.

## Installing

First, begin by installing spaCy's multilingual sentence model:

```sh
# Install spaCy's xx_sent_ud_sm:
pip install https://github.com/explosion/spacy-models/releases/download/xx_sent_ud_sm-3.7.0/xx_sent_ud_sm-3.7.0-py3-none-any.whl
```

Next, it is optional but recommended to install [an accelerated llama-cpp-python precompiled binary](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#supported-backends) with:

```sh
# Configure which llama-cpp-python precompiled binary to install (⚠️ only v0.2.88 is supported right now):
LLAMA_CPP_PYTHON_VERSION=0.2.88
PYTHON_VERSION=310
ACCELERATOR=metal|cu121|cu122|cu123|cu124
PLATFORM=macosx_11_0_arm64|linux_x86_64|win_amd64

# Install llama-cpp-python:
pip install "https://github.com/abetlen/llama-cpp-python/releases/download/v$LLAMA_CPP_PYTHON_VERSION-$ACCELERATOR/llama_cpp_python-$LLAMA_CPP_PYTHON_VERSION-cp$PYTHON_VERSION-cp$PYTHON_VERSION-$PLATFORM.whl"
```

Finally, install RAGLite with:

```sh
pip install raglite
```

To add support for filetypes other than PDF, use the `pandoc` extra:

```sh
pip install raglite[pandoc]
```

To add support for evaluation, use the `ragas` extra:

```sh
pip install raglite[ragas]
```

## Using

### Overview

1. [Configuring RAGLite](#1-configuring-raglite)
2. [Inserting documents](#2-inserting-documents)
3. [Searching and Retrieval-Augmented Generation (RAG)](#3-searching-and-retrieval-augmented-generation-rag)
4. [Computing and using an optimal query adapter](#4-computing-and-using-an-optimal-query-adapter)
5. [Evaluation of retrieval and generation](#5-evaluation-of-retrieval-and-generation)

### 1. Configuring RAGLite

> [!TIP]
> 🧠 RAGLite extends [LiteLLM](https://github.com/BerriAI/litellm) with support for [llama.cpp](https://github.com/ggerganov/llama.cpp) models using [llama-cpp-python](https://github.com/abetlen/llama-cpp-python). To select a llama.cpp model (e.g., from [bartowski's collection](https://huggingface.co/bartowski)), use a model identifier of the form `"llama-cpp-python/<hugging_face_repo_id>/<filename>@<n_ctx>"`, where `n_ctx` is an optional parameter that specifies the context size of the model.

> [!TIP]
> 💾 You can create a PostgreSQL database in a few clicks at [neon.tech](https://neon.tech).

First, configure RAGLite with your preferred PostgreSQL or SQLite database and [any LLM supported by LiteLLM](https://docs.litellm.ai/docs/providers/openai):

```python
from raglite import RAGLiteConfig

# Example 'remote' config with a PostgreSQL database and an OpenAI LLM:
my_config = RAGLiteConfig(
    db_url="postgresql://my_username:my_password@my_host:5432/my_database"
    llm="gpt-4o-mini",  # Or any LLM supported by LiteLLM.
    embedder="text-embedding-3-large",  # Or any embedder supported by LiteLLM.
)

# Example 'local' config with a SQLite database and a llama.cpp LLM:
my_config = RAGLiteConfig(
    db_url="sqlite:///raglite.sqlite",
    llm="llama-cpp-python/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/*Q4_K_M.gguf@8192",
    embedder="llama-cpp-python/lm-kit/bge-m3-gguf/*F16.gguf",
)
```

You can also configure [any reranker supported by rerankers](https://github.com/AnswerDotAI/rerankers):

```python
from rerankers import Reranker

# Example remote API-based reranker:
my_config = RAGLiteConfig(
    db_url="postgresql://my_username:my_password@my_host:5432/my_database"
    reranker=Reranker("cohere", lang="en", api_key=COHERE_API_KEY)
)

# Example local cross-encoder reranker per language (this is the default):
my_config = RAGLiteConfig(
    db_url="sqlite:///raglite.sqlite",
    reranker=(
        ("en", Reranker("ms-marco-MiniLM-L-12-v2", model_type="flashrank")),  # English
        ("other", Reranker("ms-marco-MultiBERT-L-12", model_type="flashrank")),  # Other languages
    )
)
```

### 2. Inserting documents

> [!TIP]
> ✍️ To insert documents other than PDF, install the `pandoc` extra with `pip install raglite[pandoc]`.

Next, insert some documents into the database. RAGLite will take care of the [conversion to Markdown](src/raglite/_markdown.py), [optimal level 4 semantic chunking](src/raglite/_split_chunks.py), and [multi-vector embedding with late chunking](src/raglite/_embed.py):

```python
# Insert documents:
from pathlib import Path
from raglite import insert_document

insert_document(Path("On the Measure of Intelligence.pdf"), config=my_config)
insert_document(Path("Special Relativity.pdf"), config=my_config)
```

### 3. Searching and Retrieval-Augmented Generation (RAG)

Now, you can search for chunks with vector search, keyword search, or a hybrid of the two. You can also rerank the search results with the configured reranker. And you can use any search method of your choice (`hybrid_search` is the default) together with reranking to answer questions with RAG:

```python
# Search for chunks:
from raglite import hybrid_search, keyword_search, vector_search

prompt = "How is intelligence measured?"
chunk_ids_vector, _ = vector_search(prompt, num_results=20, config=my_config)
chunk_ids_keyword, _ = keyword_search(prompt, num_results=20, config=my_config)
chunk_ids_hybrid, _ = hybrid_search(prompt, num_results=20, config=my_config)

# Retrieve chunks:
from raglite import retrieve_chunks

chunks_hybrid = retrieve_chunks(chunk_ids_hybrid, config=my_config)

# Rerank chunks:
from raglite import rerank

chunks_reranked = rerank(prompt, chunks_hybrid, config=my_config)

# Answer questions with RAG:
from raglite import rag

prompt = "What does it mean for two events to be simultaneous?"
stream = rag(prompt, config=my_config)
for update in stream:
    print(update, end="")

# You can also pass a search method or search results directly:
stream = rag(prompt, search=hybrid_search, config=my_config)
stream = rag(prompt, search=chunks_reranked, config=my_config)
```

### 4. Computing and using an optimal query adapter

RAGLite can compute and apply an [optimal closed-form query adapter](src/raglite/_query_adapter.py) to the prompt embedding to improve the output quality of RAG. To benefit from this, first generate a set of evals with `insert_evals` and then compute and store the optimal query adapter with `update_query_adapter`:

```python
# Improve RAG with an optimal query adapter:
from raglite import insert_evals, update_query_adapter

insert_evals(num_evals=100, config=my_config)
update_query_adapter(config=my_config)  # From here, simply call vector_search to use the query adapter.
```

### 5. Evaluation of retrieval and generation

If you installed the `ragas` extra, you can use RAGLite to answer the evals and then evaluate the quality of both the retrieval and generation steps of RAG using [Ragas](https://github.com/explodinggradients/ragas):

```python
# Evaluate retrieval and generation:
from raglite import answer_evals, evaluate, insert_evals

insert_evals(num_evals=100, config=my_config)
answered_evals_df = answer_evals(num_evals=10, config=my_config)
evaluation_df = evaluate(answered_evals_df, config=my_config)
```

## Contributing

<details>
<summary>Prerequisites</summary>

<details>
<summary>1. Set up Git to use SSH</summary>

1. [Generate an SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key) and [add the SSH key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).
1. Configure SSH to automatically load your SSH keys:
    ```sh
    cat << EOF >> ~/.ssh/config
    
    Host *
      AddKeysToAgent yes
      IgnoreUnknown UseKeychain
      UseKeychain yes
      ForwardAgent yes
    EOF
    ```

</details>

<details>
<summary>2. Install Docker</summary>

1. [Install Docker Desktop](https://www.docker.com/get-started).
    - _Linux only_:
        - Export your user's user id and group id so that [files created in the Dev Container are owned by your user](https://github.com/moby/moby/issues/3206):
            ```sh
            cat << EOF >> ~/.bashrc
            
            export UID=$(id --user)
            export GID=$(id --group)
            EOF
            ```

</details>

<details>
<summary>3. Install VS Code or PyCharm</summary>

1. [Install VS Code](https://code.visualstudio.com/) and [VS Code's Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers). Alternatively, install [PyCharm](https://www.jetbrains.com/pycharm/download/).
2. _Optional:_ install a [Nerd Font](https://www.nerdfonts.com/font-downloads) such as [FiraCode Nerd Font](https://github.com/ryanoasis/nerd-fonts/tree/master/patched-fonts/FiraCode) and [configure VS Code](https://github.com/tonsky/FiraCode/wiki/VS-Code-Instructions) or [configure PyCharm](https://github.com/tonsky/FiraCode/wiki/Intellij-products-instructions) to use it.

</details>

</details>

<details open>
<summary>Development environments</summary>

The following development environments are supported:

1. ⭐️ _GitHub Codespaces_: click on _Code_ and select _Create codespace_ to start a Dev Container with [GitHub Codespaces](https://github.com/features/codespaces).
1. ⭐️ _Dev Container (with container volume)_: click on [Open in Dev Containers](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/superlinear-ai/raglite) to clone this repository in a container volume and create a Dev Container with VS Code.
1. _Dev Container_: clone this repository, open it with VS Code, and run <kbd>Ctrl/⌘</kbd> + <kbd>⇧</kbd> + <kbd>P</kbd> → _Dev Containers: Reopen in Container_.
1. _PyCharm_: clone this repository, open it with PyCharm, and [configure Docker Compose as a remote interpreter](https://www.jetbrains.com/help/pycharm/using-docker-compose-as-a-remote-interpreter.html#docker-compose-remote) with the `dev` service.
1. _Terminal_: clone this repository, open it with your terminal, and run `docker compose up --detach dev` to start a Dev Container in the background, and then run `docker compose exec dev zsh` to open a shell prompt in the Dev Container.

</details>

<details>
<summary>Developing</summary>

- This project follows the [Conventional Commits](https://www.conventionalcommits.org/) standard to automate [Semantic Versioning](https://semver.org/) and [Keep A Changelog](https://keepachangelog.com/) with [Commitizen](https://github.com/commitizen-tools/commitizen).
- Run `poe` from within the development environment to print a list of [Poe the Poet](https://github.com/nat-n/poethepoet) tasks available to run on this project.
- Run `poetry add {package}` from within the development environment to install a run time dependency and add it to `pyproject.toml` and `poetry.lock`. Add `--group test` or `--group dev` to install a CI or development dependency, respectively.
- Run `poetry update` from within the development environment to upgrade all dependencies to the latest versions allowed by `pyproject.toml`.
- Run `cz bump` to bump the package's version, update the `CHANGELOG.md`, and create a git tag.

</details>
