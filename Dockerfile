FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Create explicit nltk data directory
RUN mkdir -p /usr/share/nltk_data

# Set environment variable
ENV NLTK_DATA=/usr/share/nltk_data

# Download required corpora into that directory
RUN python -m nltk.downloader -d /usr/share/nltk_data vader_lexicon stopwords punkt

#COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
