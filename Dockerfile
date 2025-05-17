FROM python:3.12.4-slim
WORKDIR counter
COPY clip /root/.cache/clip
COPY project/requirements.txt .
RUN pip install -r requirements.txt
COPY project .