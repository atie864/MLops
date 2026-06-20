# HW03 Docker Image Size Report
 subprocess in the notebook can't see your Docker images because the notebook is running in a different environment so I check the actual repository names by running `docker images | grep qbc12` manually.

- *qbc12-airbnb-serving---optimized--->834MB*

- *qbc12-airbnb-serving---naive--->1.71GB*


## Analysis
The optimized multi-stage image is significantly smaller than the naive image because the builder stage installs compilers and build tools that are never copied into the final runtime image. The naive image uses the full `python:3.8` base (which includes gcc, make, and many development libraries), while the optimized image uses `python:3.8-slim` and only copies the installed packages — not the build toolchain.
