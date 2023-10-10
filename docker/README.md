# Debugging in container with visual code

if the image is "registry.sb.upf.edu/colt/sysif:0.1", do:
> docker run --platform linux/amd64 -it -dp 80:80 -v /Users/corentk/UNLACE/SYSIF:/sysif registry.sb.upf.edu/colt/sysif:0.1

Then in visual code: docker > container > [select sysif image] > Attach Visual Studio Code

It will open a new windows. Open the '/sysif' folder and launch the debugger.