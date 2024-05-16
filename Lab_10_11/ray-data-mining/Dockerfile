# image: ray-data-mining

FROM ubuntu:22.04

USER root

RUN set -ex; 
RUN apt-get update
RUN    apt-get install -y sudo python3 python3-pip
RUN    pip3 cache purge
RUN    pip3 install numpy 
RUN    pip3 install scikit-learn 
RUN    pip3 install scipy 
RUN    pip3 install pandas 
RUN    pip3 install matplotlib 
RUN    pip3 install ray[air] 
RUN    pip3 install py_spy
RUN    rm -rf /var/lib/apt/lists/*

# Create a group named 'users' and add user 'ray' to the 'users' group

#RUN useradd -m -g users ray && \
#    echo "ray ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/ray && \
#    chmod 0440 /etc/sudoers.d/ray

#USER ray

