[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/radix-ai/raglite) [![Open in GitHub Codespaces](https://img.shields.io/static/v1?label=GitHub%20Codespaces&message=Open&color=blue&logo=github)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=812973394&skip_quickstart=true)

# 🧵 RAGLite

RAGLite is a Python package for Retrieval-Augmented Generation (RAG) with SQLite.

## Features

1. ❤️ Only lightweight and permissive open source dependencies (e.g., no [PyTorch](https://github.com/pytorch/pytorch), [LangChain](https://github.com/langchain-ai/langchain), or [PyMuPDF](https://github.com/pymupdf/PyMuPDF))
2. 🔒 Fully local RAG with [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) as an LLM provider and [SQLite](https://github.com/sqlite/sqlite) as a local database
3. 🚀 Acceleration with Metal on macOS and with CUDA on Linux and Windows
4. 📖 PDF to Markdown conversion on top of [pdftext](https://github.com/VikParuchuri/pdftext) and [pypdfium2](https://github.com/pypdfium2-team/pypdfium2)
5. ✂️ Optimal [level 4 semantic chunking](https://medium.com/@anuragmishra_27746/five-levels-of-chunking-strategies-in-rag-notes-from-gregs-video-7b735895694d) by solving a [binary integer programming problem](https://en.wikipedia.org/wiki/Integer_programming)
6. 📌 Markdown-based [contextual chunk headings](https://d-star.ai/solving-the-out-of-context-chunk-problem-for-rag)
7. 🌈 Combined sentence-level and chunk-level matching with [multi-vector chunk retrieval](https://python.langchain.com/v0.2/docs/how_to/multi_vector/)
8. 🌀 Optimal [closed-form linear query adapter](src/raglite/_query_adapter.py) by solving an [orthogonal Procrustes problem](https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem)
9. 🔍 [Hybrid search](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) that combines [SQLite's BM25 full-text search](https://sqlite.org/fts5.html) with [PyNNDescent's ANN vector search](https://github.com/lmcinnes/pynndescent)
10. ✍️ Optional support for conversion of any input document to Markdown with [Pandoc](https://github.com/jgm/pandoc)

## Installing

To install this package (including Metal acceleration if on macOS), run:

```sh
pip install raglite
```

To add CUDA 12.4 support, use the `cuda124` extra:

```sh
pip install raglite[cuda124]
```

To add support for filetypes other than PDF, use the `pandoc` extra:

```sh
pip install raglite[pandoc]
```

## Using

Example usage:

```python
# Configure the database and LLM:
from raglite import RAGLiteConfig

my_config = RAGLiteConfig(db_url="sqlite:///raglite.sqlite")

# Insert documents:
from pathlib import Path
from raglite import insert_document

insert_document(Path("On the Measure of Intelligence.pdf"), config=my_config)
insert_document(Path("Special Relativity.pdf"), config=my_config)

# Search for chunks:
from raglite import hybrid_search, keyword_search, vector_search

prompt = "How is intelligence measured?"
results_vector = vector_search(prompt, num_results=5, config=my_config)
results_keyword = keyword_search(prompt, num_results=5, config=my_config)
results_hybrid = hybrid_search(prompt, num_results=5, config=my_config)

# Answer questions with RAG:
from raglite import rag

prompt = "What does it mean for two events to be simultaneous?"
stream = rag(prompt, search=hybrid_search, config=my_config)
for update in stream:
    print(update, end="")

# Improve RAG with an optimal query adapter (optional):
from raglite import insert_evals, update_query_adapter

insert_evals(num_evals=100, config=my_config)
update_query_adapter(config=my_config)
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
1. ⭐️ _Dev Container (with container volume)_: click on [Open in Dev Containers](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/radix-ai/raglite) to clone this repository in a container volume and create a Dev Container with VS Code.
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
