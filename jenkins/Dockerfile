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
RUN curl -s -L https://julialang-s3.julialang.org/bin/linux/x64/${JULIA}/julia-${JULIA}-latest-linux-x86_64.tar.gz | \
    tar -C /usr/local -x -z --strip-components=1 -f -
