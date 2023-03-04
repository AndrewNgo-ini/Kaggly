# Use an official Ubuntu as a parent image
FROM ubuntu:20.04

# Update the package list and install dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set the command to run when the container starts
CMD ["bash"]
