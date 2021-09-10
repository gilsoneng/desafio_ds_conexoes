FROM python:3.8.3
RUN pip install --upgrade pip
COPY . /code
COPY RAW/conexoes_espec.csv /code/RAW/
COPY RAW/individuos_espec.csv /code/RAW/
WORKDIR /code
RUN ls
RUN cd RAW/
RUN ls
RUN cd ..
RUN pip install -r requirements.txt
CMD ["python", "desafio.py", "cat > log.txt"] 
