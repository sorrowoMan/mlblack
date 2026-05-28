FROM python:3.12-slim

LABEL org.opencontainers.image.title="mlblack"
LABEL org.opencontainers.image.description="Optimization-first ML framework test environment"

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /workspace/
WORKDIR /workspace

RUN pip install --no-cache-dir -e ".[all]"

RUN pip install --no-cache-dir pytest

COPY . /workspace/

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]
