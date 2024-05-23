# image: spark-python

FROM spark:3.5.0-scala2.12-java17-ubuntu

USER root

RUN set -ex; \
    apt-get update; \
    apt-get install -y sudo python3 python3-pip; \
    pip3 cache purge; \
    pip3 install pyspark; \
    rm -rf /var/lib/apt/lists/*
# RUN echo "spark ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
ENV JAVA_HOME /opt/java/openjdk
RUN export JAVA_HOME

USER spark
