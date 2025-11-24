FROM ubuntu

WORKDIR /tmp/
RUN apt update && apt install telnet -y

USER root

RUN touch file.txt
RUN echo "Hello-world" > file.txt
#CMD  

