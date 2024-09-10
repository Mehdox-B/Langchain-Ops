# Streamlit Dockerfile
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the Marketing_Expert folder to app code
COPY /Marketing_Expert/. /app/
COPY .env /app/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# ADD execution permissions to run.sh script to container
RUN chmod +x /app/run.sh

# Expose Streamlit port
EXPOSE 8501
# Expose Fastapi docs port
EXPOSE 8000

# Command to run the streamlit and fastapi docs servies
CMD ["bash","/app/run.sh" ]