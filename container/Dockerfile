# Base image
FROM nvidia/cuda:12.9.0-base-ubuntu20.04

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/	

# Solving timezone 
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install python
RUN apt-get update && apt-get install -y python3.12 python3-pip

# Install git
RUN apt-get -y install git

# Set the working directory
WORKDIR /app


# Clone 
RUN git clone https://github.com/maurinl26/dwarf-p-ice3.git /app

# Sync requirements 
RUN uv sync --extra test 

# Set uv as entry point to run scripts
CMD ["uv", "run"]



