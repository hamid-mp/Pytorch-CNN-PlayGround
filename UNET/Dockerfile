FROM python:latest

RUN mkdir /app
ADD . /app
WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "app.py", "-hs","0.0.0.0", "-ps", "8080"]