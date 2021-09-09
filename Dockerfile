FROM python:3
RUN apt update -y
RUN pip install --upgrade pip
COPY . /code
WORKDIR /code
ADD RAW /
RUN pip install -r requirements.txt
CMD [ "python", "./desafio.py" ]
