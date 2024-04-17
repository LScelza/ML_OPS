# Usa la imagen base de Python desde Docker Hub
FROM python:3.9-slim

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia el archivo de requerimientos al contenedor en el directorio /app
COPY requirements.txt /app

# Copia el archivo de Dataset al contenedor en el directorio /app / Dataset
COPY Dataset /app/Dataset

# Instala las dependencias desde requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia el archivo de Python al contenedor en el directorio /app
COPY main.py /app

# Expone el puerto 8000
EXPOSE 8000

# Ejecuta el comando uvicorn al iniciar el contenedor
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]



