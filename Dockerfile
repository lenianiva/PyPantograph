FROM ubuntu:24.10

# Install Python and Poetry
RUN apt update &&\
	apt install -y python3 curl git &&\
	curl -sSL https://install.python-poetry.org | python3 -

# Install Lean
RUN curl -sSfL https://github.com/leanprover/elan/releases/download/v3.1.1/elan-x86_64-unknown-linux-gnu.tar.gz | tar xz &&\
	./elan-init -y --default-toolchain none
ENV PATH="$PATH:/root/.local/bin:/root/.elan/bin"

COPY . /root/pantograph
WORKDIR /root/pantograph

RUN poetry build &&\
	poetry install --with dev 
