FROM jupyter/base-notebook

ENV TRANSFORMERS_CACHE=/tmp/.cache
ENV TOKENIZERS_PARALLELISM=true

# Add RUN statements to install packages as the $NB_USER defined in the base images.

# Add a "USER root" statement followed by RUN statements to install system packages using apt-get,
# change file permissions, etc.

# If you do switch to root, always be sure to add a "USER $NB_USER" command at the end of the
# file to ensure the image runs as a unprivileged user by default.

USER root

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    jupyter \
    tensorflow \
    torch \ 
    torchvision \
    torchaudio

RUN python3 -m pip install --no-cache-dir \
    scikit-learn \
    keras \
    pandas

RUN python3 -m pip install --no-cache-dir \
    transformers \
    datasets\
    nltk \
    pytorch_lightning \
    gradio \
    sentencepiece \
    seqeval

RUN python3 -m pip install --upgrade jupyterlab-vim


USER ${NB_UID}

WORKDIR "${HOME}"
