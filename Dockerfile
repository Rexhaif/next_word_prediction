FROM python:3.7-buster

WORKDIR app
COPY ./ ./

RUN ls /app

RUN pip install -r ./requirements.txt

EXPOSE 8000

CMD python app.py