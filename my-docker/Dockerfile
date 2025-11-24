FROM nginx:latest
WORKDIR /opt
COPY hosts myhosts 
ADD mydata.tar.gz .
USER root 
RUN echo "Hello world" >> /usr/share/nginx/html/index.html
RUN useradd john
USER john
RUN touch /tmp/file1.txt
USER root
#RUN echo "Hello world" >> /usr/share/nginx/html/index.html
