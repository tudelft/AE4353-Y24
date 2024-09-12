FROM continuumio/miniconda3 as upstream

# Temporary: Upgrade python packages due to mentioned CVEs
# They are installed by the base image (continuumio/miniconda3) which does not have the patch.
# RUN conda install \ 
# https://github.com/advisories/<CVE_ID>
# <package_name> = <version>

RUN python3 -m pip install --upgrade \
    # https://github.com/advisories/GHSA-6vqw-3v5j-54x4
    cryptography==42.0.4 \
    # installed for compatibility with cryptography v42.0.4
    pyopenssl==24.0.0 

# Reset and copy updated files with updated privs to keep image size down
FROM mcr.microsoft.com/devcontainers/base:ubuntu

ARG USERNAME=vscode

# Create the conda group and add remote user to the group
RUN groupadd -r conda --gid 900 \ 
    && usermod -aG conda ${USERNAME}

# Copy opt folder, set ownership and group permissions
COPY --chown=:conda --chmod=775 --from=upstream /opt/conda /opt/conda
RUN chmod =2775 /opt/conda

USER root


# Setup conda to mirror contents from https://github.com/ContinuumIO/docker-images/blob/main/miniconda3/debian/Dockerfile
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH
ARG USERNAME=vscode
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    mercurial \
    openssh-client \
    procps \
    subversion \
    tmux \
    wget \    
    && apt-get upgrade -y \
    # && mv -f "/tmp/library-scripts/meta.env" /usr/local/etc/vscode-dev-containers/meta.env \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

COPY env.yml /tmp/conda-tmp/
RUN conda env create -f /tmp/conda-tmp/env.yml && rm -rf /tmp/conda-tmp

RUN if [ "$(uname -m)" = "aarch64" ]; then \
    echo "Running on ARM architecture"; \
    # Replace OpenBLAS with the version with OpenMP support on ARM systems
    # AMD64 systems use the libBLAS with MKL which is installed by default
    conda install -n AE4353 -c conda-forge libopenblas=*=openmp_* -y; \
fi

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>