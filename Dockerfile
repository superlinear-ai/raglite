# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.10
FROM ghcr.io/astral-sh/uv:python$PYTHON_VERSION-bookworm AS dev

# Create and activate a virtual environment [1].
# [1] https://docs.astral.sh/uv/concepts/projects/config/#project-environment-path
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=$VIRTUAL_ENV/bin:$PATH
ENV UV_PROJECT_ENVIRONMENT=$VIRTUAL_ENV

# Tell Git that the workspace is safe to avoid 'detected dubious ownership in repository' warnings.
RUN git config --system --add safe.directory '*'

# Configure the user's shell.
RUN echo 'HISTFILE=~/.history/.bash_history' >> ~/.bashrc && \
    echo 'bind "\"\e[A\": history-search-backward"' >> ~/.bashrc && \
    echo 'bind "\"\e[B\": history-search-forward"' >> ~/.bashrc && \
    mkdir ~/.history/

FROM python:$PYTHON_VERSION-slim AS app

# Configure Python to print tracebacks on crash [1], and to not buffer stdout and stderr [2].
# [1] https://docs.python.org/3/using/cmdline.html#envvar-PYTHONFAULTHANDLER
# [2] https://docs.python.org/3/using/cmdline.html#envvar-PYTHONUNBUFFERED
ENV PYTHONFAULTHANDLER=1
ENV PYTHONUNBUFFERED=1

# Remove docker-clean so we can manage the apt cache with the Docker build cache.
RUN rm /etc/apt/apt.conf.d/docker-clean

# Install compilers that may be required for certain packages or platforms.
RUN --mount=type=cache,target=/var/cache/apt/ \
    --mount=type=cache,target=/var/lib/apt/ \
    apt-get update && \
    apt-get install --no-install-recommends --yes build-essential

# Create a non-root user and switch to it [1].
# [1] https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
ARG UID=1000
ARG GID=$UID
RUN groupadd --gid $GID user && \
    useradd --create-home --gid $GID --uid $UID user --no-log-init
USER user

# Set the working directory.
WORKDIR /workspaces/raglite/

# Copy the app source code to the working directory.
COPY --chown=user:user . .

# Install the application and its dependencies [1].
# [1] https://docs.astral.sh/uv/guides/integration/docker/#optimizations
RUN --mount=type=cache,uid=$UID,gid=$GID,target=/home/user/.cache/uv \
    --mount=from=ghcr.io/astral-sh/uv,source=/uv,target=/bin/uv \
    uv sync \
    --all-extras \
    --compile-bytecode \
    --frozen \
    --link-mode copy \
    --no-dev \
    --no-editable \
    --python-preference only-system

# Configure the non-root user's shell.
ENV ANTIDOTE_VERSION 1.8.6
RUN git clone --branch v$ANTIDOTE_VERSION --depth=1 https://github.com/mattmc3/antidote.git ~/.antidote/ && \
    echo 'zsh-users/zsh-syntax-highlighting' >> ~/.zsh_plugins.txt && \
    echo 'zsh-users/zsh-autosuggestions' >> ~/.zsh_plugins.txt && \
    echo 'source ~/.antidote/antidote.zsh' >> ~/.zshrc && \
    echo 'antidote load' >> ~/.zshrc && \
    echo 'eval "$(starship init zsh)"' >> ~/.zshrc && \
    echo 'HISTFILE=~/.history/.zsh_history' >> ~/.zshrc && \
    echo 'HISTSIZE=1000' >> ~/.zshrc && \
    echo 'SAVEHIST=1000' >> ~/.zshrc && \
    echo 'setopt share_history' >> ~/.zshrc && \
    echo 'bindkey "^[[A" history-beginning-search-backward' >> ~/.zshrc && \
    echo 'bindkey "^[[B" history-beginning-search-forward' >> ~/.zshrc && \
    mkdir ~/.history/ && \
    zsh -c 'source ~/.zshrc'

# Expose the app.
ENTRYPOINT ["/workspaces/raglite/.venv/bin/poe"]
CMD ["serve"]