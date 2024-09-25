# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files (not the virtual environment)
COPY app.py .
COPY .env .
# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables (if needed, customize as required)
# If you don't need any specific environment variables, you can remove this
#ENV MY_ENV_VAR=january

# Expose port 80 (or whatever port your app uses, adjust as needed)
EXPOSE 80

# Run your Python app
CMD ["python", "app.py"]
