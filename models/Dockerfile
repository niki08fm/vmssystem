FROM python:3.8

WORKDIR /app

COPY requirements.txt .

# Pillow fix for Python 3.8
RUN sed -i 's/Pillow.*/Pillow<10/g' requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
