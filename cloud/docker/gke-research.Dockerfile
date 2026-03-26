FROM ghcr.io/prefix-dev/pixi:latest

WORKDIR /app

# Copy dependency and project metadata first for better layer caching.
COPY pixi.toml pixi.lock pyproject.toml README.md ./
COPY sp26_gke ./sp26_gke

RUN pixi install --frozen

CMD ["pixi", "run", "research-agent"]
