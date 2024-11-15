FROM python:3.8
WORKDIR /home

RUN apt-get -y update
RUN apt-get -y install vim nano

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models ./models
COPY train_codes ./train_codes

COPY test.py ./
COPY app.py ./

CMD ["python","app.py"]