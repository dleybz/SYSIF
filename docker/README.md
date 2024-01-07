# Debugging in container with visual code

Instructions to Danny:
Start by manually creating an image from the Dockerfile using the VSCode console or run `docker build - < SYSIF/docker/Dockerfile -t sysif:latest --platform linux`  to build the image
Then run `docker run -m 16g --platform linux/amd64 -it -dp 80:80 -v '/Users/dleybz/Documents/UPF Courses/Research/Neural Activation Patterns/Analysis 0/SYSIF':/analysis sysif` to run a container


TO BE ABLE TO RUN ON LINUX/AMD64 THE PLATFORM, CAN'T BUILD THE IMAGE LOCALLY https://stackoverflow.com/questions/68342427/unable-to-find-image-namelatest-locally
docker build - < SYSIF/docker/Dockerfile -t sysif:latest --platform linux/amd64
docker run -m 16g --platform linux/amd64 -it -dp 80:80 -v '/Users/dleybz/Documents/UPF Courses/Research/Neural Activation Patterns/Analysis 0/' sysif


if the image is "registry.sb.upf.edu/colt/sysif:0.1", do:
> docker run -m 16g --platform linux/amd64 -it -dp 80:80 -v "/Users/dleybz/Documents/UPF Courses/Research/Neural Activation Patterns/Analysis 0/SYSIF/docker/Dockerfile" 

Then in visual code: docker > container > [select sysif image] > Attach Visual Studio Code

It will open a new windows. Open the '/sysif' folder and launch the debugger.

# HPC cluster + singluarity

Load the image from the registry (require to be in interactive mode)
> singularity pull --docker-login docker://registry.sb.upf.edu/colt/sysif:0.2
> mkdir SYSIF/docker/images
> mv sysif_0.2.sif SYSIF/docker/images

When you are in interactive mode, open the image using:
> singularity run --nv docker/images/sysif_0.2.sif