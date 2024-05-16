# image: ray-jupyter

FROM ray-data-mining

USER root

RUN pip3 install matplotlib 
RUN pip3 install seaborn 
RUN pip3 install jupyterlab 
RUN pip3 install ipywidgets tqdm

#RUN useradd -m -g users ray

RUN useradd -m -g users ray && \
    echo "ray ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/ray && \
    chmod 0440 /etc/sudoers.d/ray

USER ray

# RUN mkdir notebooks

WORKDIR /home/ray/notebooks

