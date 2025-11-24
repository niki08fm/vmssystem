FROM python:3.8

WORKDIR /app

COPY requirements.txt .

# Pillow fix for Python 3.8
RUN sed -i 's/Pillow.*/Pillow<10/g' requirements.txt

# 1. Install system dependencies for OpenCV (cv2)
# The 'python:3.8' image is based on Debian, so we use apt-get
# libgl1 is the package that provides libGL.so.1
# libSM and libXext are also commonly needed by OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libsm6 \
        libxext6 && \
    rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
