# Imagen para servir un modelo entrenado con Blackcode (API estilo OpenAI).
#
# Construir desde la raíz del repo:
#   docker build -t blackcode-serve .
#
# Ejecutar montando la config y el modelo del host:
#   docker run --rm -p 8000:8000 \
#     -v "$PWD/config.yaml:/app/config.yaml" \
#     -v "$PWD/blackcode-output:/app/blackcode-output" \
#     blackcode-serve
FROM python:3.12-slim

WORKDIR /app

# Solo lo necesario para instalar el paquete: el núcleo + extras de inferencia
# (sin las dependencias de entrenamiento).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[serve,llm]'

EXPOSE 8000
ENTRYPOINT ["blackcode", "serve"]
CMD ["config.yaml"]
