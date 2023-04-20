# Use the official NVIDIA CUDA image as the base image
FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu18.04

# Set the working directory
WORKDIR /app

# Update and install necessary packages
RUN apt-get clean \
    && apt-get update --fix-missing \
    && apt-get install -y \
    python3-pip \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Azure Kinect SDK
RUN wget https://packages.microsoft.com/config/ubuntu/18.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb \
    && dpkg -i packages-microsoft-prod.deb \
    && apt-get update \
    && apt-get install -y k4a-tools

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of your project files into the container
COPY . .

# Set the default command to run when starting the container
CMD ["python3", "app.py"]
