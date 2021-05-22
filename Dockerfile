FROM python:3.8-slim-buster

RUN mkdir -p usr/app

WORKDIR /app

ADD . /app

RUN pip3 install --trusted-host nexusoss.wistron.com --trusted-host nexusoss.wistron.com -i https://nexusoss.wistron.com/repository/pypi-proxy/simple -r requirements.txt

EXPOSE 80


CMD ["python", "train.py"]
