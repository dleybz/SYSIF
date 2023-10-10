FROM nvidia/cuda:11.8.0-base-ubuntu20.04

LABEL name Corentin Kervadec "https://registry.sb.upf.edu"

LABEL authors="Corentin Kervadec"
LABEL version="0.10"
LABEL description="CUDA 11.8 Pytorch 2.0.1 image with \
all dependancies requires for SISYF"

# /!\ If building the image from  macOs, don't forget to add --platform linux/amd64 

# Update
RUN apt-get update

################### I N S T A L L - L I B #############################
# Python builder
RUN apt install -y build-essential
RUN apt install -y python3-pip

# Pytorch 
RUN pip3 install --no-cache-dir MarkupSafe
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu118


# Python utils
RUN pip3 install --no-cache-dir tqdm termcolor colorama overrides cached-path gdown 
# Unit test
RUN pip3 install --no-cache-dir pytest
# python formatting
RUN pip3 install --no-cache-dir black
# Scientific libs
RUN pip3 install --no-cache-dir Cython spacy scipy pandas
# Viz
RUN pip3 install --no-cache-dir seaborn
# Hugging face
RUN pip3 install --no-cache-dir transformers datasets
# Acceleration
RUN pip3 install --no-cache-dir accelerate bitsandbytes
# RUN pip3 install --no-cache-dir datasets apache_beam mwparserfromhell
RUN ln -s /usr/bin/python3 /usr/bin/python
#######################################################################


# Cleaning
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN mkdir /offload_folder