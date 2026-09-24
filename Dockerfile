FROM apify/actor-python:3.11

# Copy requirements.txt
COPY requirements.txt ./

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code into container
COPY . ./

# Execute the actor
CMD ["python", "-m", "src.main"]
