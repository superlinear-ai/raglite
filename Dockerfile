# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.10-bookworm AS dev

# Create and activate a virtual environment [1].
# [1] https://docs.astral.sh/uv/concepts/projects/config/#project-environment-path
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=$VIRTUAL_ENV/bin:$PATH
ENV UV_PROJECT_ENVIRONMENT=$VIRTUAL_ENV

# Tell Git that the workspace is safe to avoid 'detected dubious ownership in repository' warnings.
RUN git config --system --add safe.directory '*'

# Create a non-root user and give it passwordless sudo access [1].
# [1] https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
RUN --mount=type=cache,target=/var/cache/apt/ \
    --mount=type=cache,target=/var/lib/apt/ \
    groupadd --gid 1000 user && \
    useradd --create-home --no-log-init --gid 1000 --uid 1000 --shell /usr/bin/bash user && \
    chown user:user /opt/ && \
    apt-get update && apt-get install --no-install-recommends --yes sudo && \
    echo 'user ALL=(root) NOPASSWD:ALL' > /etc/sudoers.d/user && chmod 0440 /etc/sudoers.d/user
USER user

# Configure the non-root user's shell.
RUN mkdir ~/.history/ && \
    echo 'HISTFILE=~/.history/.bash_history' >> ~/.bashrc && \
    echo 'bind "\"\e[A\": history-search-backward"' >> ~/.bashrc && \
    echo 'bind "\"\e[B\": history-search-forward"' >> ~/.bashrc && \
    echo 'eval "$(starship init bash)"' >> ~/.bashrc