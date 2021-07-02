FROM tensorflow/tensorflow:2.3.0

RUN mkdir /root/workspace
WORKDIR /root/workspace

ADD . / education2skill/
WORKDIR /root/workspace/education2skill/

RUN pip install --use-feature=2020-resolver --no-input -r requirements.txt

