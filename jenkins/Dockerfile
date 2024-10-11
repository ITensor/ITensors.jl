FROM nvidia/cuda:12.0.0-devel-ubuntu20.04

# julia
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install --yes --no-install-recommends \
                    # basic stuff
                    curl ca-certificates \
                    libcutensor-dev \
                    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG JULIA=1.6
RUN curl -fsSL https://install.julialang.org | sh -s -- --path /usr/local/.juliaup --yes
RUN /usr/local/.juliaup/bin/juliaup add ${JULIA}
RUN /usr/local/.juliaup/bin/juliaup default ${JULIA}