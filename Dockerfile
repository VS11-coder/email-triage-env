# ── Base image ────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Metadata
LABEL maintainer="openenv-hackathon"
LABEL description="Email Triage OpenEnv — real-world email RL environment"

# ── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────
COPY . .

# ── Build env/ package structure from flat files ───────────────────────────
# Source files are flat; the code imports from env.* package.
# Create the package with symlinks so both flat and package imports work.
RUN mkdir -p env/data && \
    touch env/__init__.py env/data/__init__.py && \
    ln -sf ../email_env.py env/email_env.py && \
    ln -sf ../models.py env/models.py && \
    ln -sf ../graders.py env/graders.py && \
    ln -sf ../../emails.py env/data/emails.py

# ── HuggingFace Spaces: non-root user ─────────────────────────────────────
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# ── Expose port ────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Health check ──────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# ── Start server ───────────────────────────────────────────────────────────
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
