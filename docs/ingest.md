# De documentos a un asistente

Blackcode puede convertir el conocimiento de tu organización (páginas web,
PDFs, documentos Word, imágenes, textos) en un asistente que responde
preguntas. El flujo tiene cuatro pasos, todos en local:

```
documentos → corpus → pares Q&A → modelo entrenado → asistente
   ingest    ingest    generate-qa     train            serve
```

## 1. Ingerir los documentos

`blackcode ingest` recorre archivos, carpetas y URLs, extrae el texto de cada
uno y escribe un corpus JSONL:

```bash
blackcode ingest ./documentos-empresa https://miempresa.com --output corpus.jsonl
```

Formatos soportados (cada uno necesita su extra):

| Tipo | Extensiones | Extra |
|---|---|---|
| Texto / Markdown | `.txt` `.md` `.rst` | — (stdlib) |
| PDF | `.pdf` | `blackcode[pdf]` |
| Word | `.docx` | `blackcode[docx]` |
| HTML / web | `.html`, URLs | `blackcode[web]` |
| Imágenes (OCR) | `.png` `.jpg` … | `blackcode[ocr]` + Tesseract |

`pip install 'blackcode[ingest]'` instala todos. Las imágenes y los PDF
escaneados requieren OCR; el resto extrae el texto directamente. Cada
documento produce un registro `{"text": ..., "source": ...}`.

### Rastrear un sitio completo

Por defecto una URL descarga solo esa página. Con `--crawl-depth` el
rastreador sigue los enlaces del mismo dominio:

```bash
blackcode ingest https://miempresa.com --crawl-depth 2 --output corpus.jsonl
```

- `--crawl-depth N` — saltos de enlace desde la URL inicial (0 = solo esa página).
- `--max-pages N` — tope de páginas por sitio (50 por defecto).
- `--delay S` — pausa entre descargas, para no saturar el servidor.
- `--ignore-robots` — no respetar `robots.txt` (por defecto sí se respeta).

Solo se siguen enlaces del mismo dominio; los externos se ignoran.

### Sitemap.xml

Si tu sitio publica un sitemap, es la vía más completa y eficiente: pásale la
URL del sitemap y se leerán todas las páginas listadas (incluyendo sitemaps
anidados de un `sitemap-index`):

```bash
blackcode ingest https://miempresa.com/sitemap.xml --output corpus.jsonl
```

Cualquier URL terminada en `.xml` se trata como sitemap.

### Bases de datos SQL

Ingiere filas como texto. La contraseña **no** va en el YAML — se pasa por
`--sql-url` o por la variable `BLACKCODE_SQL_URL`.

| Motor       | Esquema URL              | Extra                  |
|-------------|--------------------------|------------------------|
| SQLite      | `sqlite:///./mi.db`      | — (stdlib)             |
| PostgreSQL  | `postgresql://u:p@h/db`  | `blackcode[postgres]`  |
| MySQL       | `mysql://u:p@h/db`       | `blackcode[mysql]`     |
| SQL Server  | `mssql://u:p@h/db`       | `blackcode[mssql]`     |
| Oracle      | `oracle://u:p@h:1521/serv`| `blackcode[oracle]`   |

```bash
export BLACKCODE_SQL_URL='postgresql://usuario:secreta@host/db'
blackcode ingest \
  --sql "SELECT titulo, cuerpo FROM articulos" \
  --sql-template "{titulo}\n\n{cuerpo}" \
  --output corpus.jsonl
```

Para Oracle el path de la URL es el *service name*, no una base de datos. El
driver (`oracledb`) usa por defecto el modo "thin" puro Python — sin Oracle
Instant Client.

`--sql` se puede repetir; cada fila no vacía produce un registro. Sin
`--sql-template` se concatenan todas las columnas no nulas.

### Notion

```bash
export NOTION_TOKEN=secret_...
blackcode ingest --notion PAGE_ID_1 --notion PAGE_ID_2 --output corpus.jsonl
```

Crea una integración interna en
[notion.so/my-integrations](https://www.notion.so/my-integrations) y
compártela con las páginas que quieras ingerir. Extra: `blackcode[notion]`.

### Confluence

```bash
export CONFLUENCE_URL=https://miempresa.atlassian.net
export CONFLUENCE_USER=tu@email
export CONFLUENCE_TOKEN=tu-api-token
blackcode ingest --confluence SPACE_KEY --output corpus.jsonl
```

Genera el token en `id.atlassian.com/manage-profile/security/api-tokens`.
Extra: `blackcode[confluence]`.

### Google Docs

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/al/service-account.json
blackcode ingest --gdocs DRIVE_FOLDER_ID --output corpus.jsonl
```

Necesitas una cuenta de servicio de Google con permisos de Drive + Docs
(read-only) y compartir la carpeta con su correo. Extra: `blackcode[gdocs]`.

### Combinar fuentes

Puedes mezclar todo en una sola invocación; cada fuente añade sus registros
al mismo `corpus.jsonl`:

```bash
blackcode ingest \
  ./docs-empresa \
  https://miempresa.com/sitemap.xml \
  --notion PAGE_ID \
  --confluence ESPACIO \
  --output corpus.jsonl
```

## 2. Generar pares pregunta/respuesta

Un documento no trae preguntas. `blackcode generate-qa` trocea el corpus y,
por cada trozo, le pide a un LLM local que genere pares Q&A:

```bash
blackcode generate-qa corpus.jsonl --output dataset.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct --device auto
```

- `--device` — `auto`, `cpu`, `cuda` o `mps`. La generación es exigente: con
  GPU es cómoda, en CPU funciona pero es lenta.
- `--per-chunk` — cuántos pares por trozo. `--chunk-size` — tamaño del trozo.
- La calidad del asistente depende directamente de la de estos pares; conviene
  un modelo generador capaz (7B o más).

La salida es un JSONL `{"instruction": ..., "output": ...}`.

## 3. Entrenar

Apunta una config al dataset generado y entrena con el trainer `llm`. El
`instruction_template` combina los campos en el texto de entrenamiento:

```yaml
data:
  path: ./dataset.jsonl
  instruction_template: "### Pregunta: {instruction}\n### Respuesta: {output}"
train:
  trainer: llm
  model: Qwen/Qwen2.5-1.5B-Instruct
```

```bash
blackcode train config.yaml
```

## 4. Servir

```bash
blackcode serve config.yaml
```

Tu asistente queda disponible en una API estilo OpenAI, en local.

## Una nota sobre expectativas

El fine-tuning enseña al modelo el tono, el vocabulario y los temas de tu
empresa, pero su memoria de datos exactos (cifras, nombres, fechas) es
imperfecta — habrá errores puntuales. Se mejora con más documentos, más
épocas y un modelo base más grande.
