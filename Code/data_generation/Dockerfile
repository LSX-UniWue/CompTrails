FROM python:3.7

WORKDIR "/workdir/"

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY Code .
RUN chown -R 1147:1003 .