# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Copy the current directory contents into the container at /text_to_image_plugin
COPY . /text_to_image_plugin

# Set the working directory in the container
WORKDIR /text_to_image_plugin

# Install any needed dependencies specified in requirements.txt
RUN pip install -r text2image_service_requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define the Built time argument 
ARG OPENAI_API_KEY
ARG LANGCHAIN_TRACING_V2
ARG LANGCHAIN_API_KEY
ARG LANGCHAIN_PROJECT

# Define environment variable for the run time (in the container)
ENV OPEN_AI_API_KEY : ${OPENAI_API_KEY} 
ENV LANGCHAIN_TRACING_V2 : ${LANGCHAIN_TRACING_V2}
ENV LANGCHAIN_API_KEY : ${LANGCHAIN_API_KEY}
ENV LANGCHAIN_PROJECT : ${LANGCHAIN_PROJECT}

# Run app.py when the container launches
CMD ["streamlit", "run","main.py"]