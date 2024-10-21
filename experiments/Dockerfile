FROM ubuntu:24.10
MAINTAINER Leni Aniva <aniva@stanford.edu>

# Install Python and Poetry
RUN apt update &&\
	apt install -y python3 curl &&\
	curl -sSL https://install.python-poetry.org | python3 -

# Install Lean
RUN curl -sSfL https://github.com/leanprover/elan/releases/download/v3.1.1/elan-x86_64-unknown-linux-gnu.tar.gz | tar xz &&\
	./elan-init -y --default-toolchain none
ENV PATH="$PATH:~/.local/bin:~/.elan/bin"

COPY .. ~/pantograph
WORKDIR ~/pantograph

RUN poetry build &&\
	poetry install
