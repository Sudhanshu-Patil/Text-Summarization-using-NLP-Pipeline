# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download necessary NLTK data
RUN python -m nltk.downloader wordnet stopwords

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Add execute permissions to the start.sh script
RUN chmod +x start.sh

# Run the start.sh script when the container launches
CMD ["./start.sh"]
