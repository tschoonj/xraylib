FROM ubuntu:bionic

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y && \
	apt-get install -y autoconf automake libtool python-all-dev python-numpy cython python3-all-dev python3-numpy fpc gfortran-7 gcc-7 g++-7 php-dev lua5.2 liblua5.2-dev swig clang openjdk-8-jre openjdk-8-jdk

WORKDIR /root
