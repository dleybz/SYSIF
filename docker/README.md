# Debugging in container with visual code

> docker build --platform linux/amd64 .

if the image is "registry.sb.upf.edu/colt/sysif:0.1", do:
> docker run -m 16g --platform linux/amd64 -it -dp 80:80 -v /Users/corentk/UNLACE/SYSIF:/sysif registry.sb.upf.edu/colt/sysif:0.3

Then in visual code: docker > container > [select sysif image] > Attach Visual Studio Code

It will open a new windows. Open the '/sysif' folder and launch the debugger.

# HPC cluster + singluarity

Load the image from the registry (require to be in interactive mode)
> singularity pull --docker-login docker://registry.sb.upf.edu/colt/sysif:0.2
> mkdir SYSIF/docker/images
> mv sysif_0.2.sif SYSIF/docker/images

When you are in interactive mode, open the image using:
> singularity run --nv docker/images/sysif_0.2.sif