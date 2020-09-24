FROM pytorch/pytorch

RUN apt-get update
RUN apt-get install -y build-essential cmake
RUN apt-get install -y libopenblas-dev liblapack-dev
RUN apt-get install -y libx11-dev libgtk-3-dev
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
