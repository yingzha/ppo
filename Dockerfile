FROM python:3.9.0 as BASE

ARG POETRY_VERSION=1.5.1

WORKDIR /app
ENV PYTHONPATH "${PYTHONPATH}:/app/"

# copy work directories
COPY configs /app/configs

# poetry stuff
RUN pip install "poetry==$POETRY_VERSION"
COPY poetry.lock pyproject.toml /app/
RUN poetry config virtualenvs.create false && poetry install --no-interaction

