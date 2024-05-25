# Use an official Python runtime as a parent image
FROM python:3.10-alpine

# Copy the current directory contents into the container at /text_to_image_plugin
COPY . /text_to_image_plugin

# Set the working directory in the container
WORKDIR /text_to_image_plugin

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
ENV OPEN_AI_API_KEY = 
ENV LANGCHAIN_TRACING_V2 =TRUE
ENV LANGCHAIN_API_KEY = 
ENV LANGCHAIN_PROJECT = "Langchain-Ops PROJECT"

# Run app.py when the container launches
CMD ["streamlit", "run","main.py"]