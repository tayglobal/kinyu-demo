FROM python:3

WORKDIR /kinyu-demo

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src .

ENV PYTHONPATH "${PYTHONPATH}:/kinyu-demo"