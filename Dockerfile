FROM ubuntu:20.04
FROM openjdk:8

RUN apt-get update && apt-get install -y \
	python3 \
	python3-pip

RUN pip3 --no-cache-dir install \
	pyspark \
	pandas \
	sklearn 

ADD my-app.py /
ADD test.csv /
ADD train.csv /
ADD lrModel.tar.xz /
ADD rfModel.tar.xz /

CMD ["spark-submit", "./my-app.py"]
