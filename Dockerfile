# Gunakan base image Python versi spesifik
FROM python:3.12.5-slim

# Tetapkan working directory dalam container
WORKDIR /code

# Salin file requirements.txt ke dalam image
COPY ./requirements.txt /code/requirements.txt

# Install dependencies dengan pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Salin semua kode aplikasi ke dalam container
COPY ./app /code/app

# Tambahkan perintah untuk memastikan file penting ada (opsional, untuk debugging)
RUN ls /code/app

# Ekspos port aplikasi (tidak diperlukan untuk Cloud Run, tapi tetap baik untuk debugging lokal)
EXPOSE 8000

# Default command untuk menjalankan aplikasi
CMD ["uvicorn", "app.main:application", "--host", "0.0.0.0", "--port", "8000"]
