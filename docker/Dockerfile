FROM debian:unstable
MAINTAINER Vladimir Polushin (vovapolu@gmail.com)

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update \
    && apt-get install -y nano mc build-essential curl git vim wget \
    && apt-get clean
RUN curl -L https://github.com/just-containers/s6-overlay/releases/download/v1.16.0.0/s6-overlay-amd64.tar.gz | tar xz -C /
ENTRYPOINT ["/init"]

RUN apt-get install --no-install-recommends -y openssh-server \
    && apt-get clean
RUN mkdir /var/run/sshd
RUN echo 'root:tmnavartmdev' | chpasswd
RUN sed -i 's/PermitRootLogin without-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config
RUN mkdir /etc/services.d/sshd \
    && echo '#!/bin/sh' > /etc/services.d/sshd/run \
    && echo 'exec /usr/sbin/sshd -D 2>&1' >> /etc/services.d/sshd/run \
    && chmod +x /etc/services.d/sshd/run
EXPOSE 22

RUN apt-get install --no-install-recommends -y postgresql postgresql-contrib libpq-dev \
    && apt-get clean
RUN mkdir /var/run/postgresql/9.5-main.pg_stat_tmp/ && chown postgres /var/run/postgresql/9.5-main.pg_stat_tmp/
RUN sed -i 's|local   all             all                                     peer|local   all             all                                     trust|' /etc/postgresql/9.5/main/pg_hba.conf \
    && sed -i 's|host    all             all             127.0.0.1/32            md5|host    all             all             127.0.0.1/32            trust|' /etc/postgresql/9.5/main/pg_hba.conf \
    && sed -i 's|host    all             all             ::1/128                 md5|host    all             all             ::1/128                 trust|' /etc/postgresql/9.5/main/pg_hba.conf
RUN mkdir /etc/services.d/postgresql \
    && echo '#!/bin/sh' > /etc/services.d/postgresql/run \
    && echo 'exec su -c "/usr/lib/postgresql/9.5/bin/postmaster -D /var/lib/postgresql/9.5/main -c config_file=/etc/postgresql/9.5/main/postgresql.conf" postgres 2>&1' >> /etc/services.d/postgresql/run \
    && chmod +x /etc/services.d/postgresql/run

RUN apt-get install -y python3 python3-dev python3-pip python3-setuptools

RUN apt-get install -y python python-dev python-pip python-setuptools
RUN apt-get install -y python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas

# set virmc
RUN curl -o ~/.vimrc https://raw.githubusercontent.com/omtcyf0/dotfiles/master/.vimrc

ENV VERSION bum
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN cd /root && git clone https://github.com/omtcyf0/tm_navigator
RUN cd /root && git clone https://github.com/vovapolu/artm-experiments

RUN apt-get install -y git make cmake build-essential libboost-all-dev
RUN cd /root && git clone --branch=stable https://github.com/bigartm/bigartm.git
WORKDIR /root/bigartm
RUN mkdir build && cd build && cmake .. && make
RUN cd /root/bigartm && cp build/3rdparty/protobuf-cmake/protoc/protoc 3rdparty/protobuf/src/
RUN cd /root/bigartm/3rdparty/protobuf/python && python setup.py build && python setup.py install

ENV BIGARTM_HOME /root

WORKDIR /root/tm_navigator
RUN pip3 install --editable .
RUN mkdir /etc/services.d/tm_navigator \
    && echo '#!/bin/sh' > /etc/services.d/tm_navigator/run \
    && echo 'cd /root/tm_navigator/' >> /etc/services.d/tm_navigator/run \
    && echo 'git pull' >> /etc/services.d/tm_navigator/run \
    && echo 'cd tm_navigator' >> /etc/services.d/tm_navigator/run \
    && echo 'python3 main.py' >> /etc/services.d/tm_navigator/run \
    && chmod +x /etc/services.d/tm_navigator/run

WORKDIR /root/artm-experiments
RUN mkdir /etc/services.d/artm-experiments \
    && echo '#!/bin/sh' > /etc/services.d/artm-experiments/run \
    && echo 'cd /root/artm-experiments/' >> /etc/services.d/artm-experiments/run \
    && echo 'git pull' >> /etc/services.d/artm-experiments/run \
    && chmod +x /etc/services.d/tm_navigator/run

RUN pip install csvwriter