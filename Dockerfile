FROM python:3.9-slim-buster

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    vim

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the source code to the container
COPY . .

# Set the default command to run when the container starts
CMD ["bash"]

#docker run -it -v /Users/ngohieu/nlp_kaggle_framework:/app myimage

