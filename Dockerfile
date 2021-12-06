FROM ubuntu:20.04
FROM openjdk:8

RUN apt-get update && apt-get install -y \
	python3 \
	python3-pip

RUN pip3 --no-cache-dir install \
	pyspark \
	pandas \
	sklearn 

ADD entry3.sh /
ADD my-app.py /
ADD Wine_forest.py /
ADD Wine_linear.py /
ADD dataSplitter.py /
ADD test.csv /
ADD train.csv /
ADD lrModel.tar.xz /
ADD rfModel.tar.xz /

RUN ["chmod", "+x", "/entry3.sh"]

ENTRYPOINT ["./entry3.sh"]
