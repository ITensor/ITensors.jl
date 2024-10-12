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

RUN rm -rf /usr/local/.juliaup
RUN rm -rf /usr/local/.julia