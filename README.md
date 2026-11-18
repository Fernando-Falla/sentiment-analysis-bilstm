# Análisis de sentimiento con Bi-LSTM (español / portugués)

Modelo Bi-LSTM que clasifica reseñas en **negativo, neutro o positivo**, servido con FastAPI y desplegado en Oracle Cloud (OCI) con Docker.

> **Estado: modelo en reentrenamiento.** El notebook original tiene problemas de rigor (ver [Limitaciones conocidas](#limitaciones-conocidas)), así que este repositorio **no publica métricas ni pesos** hasta corregirlo. Lo que sí está validado es la API y su deploy.

## Contexto y mi rol

**Hackathon ONE G8** (Alura Latam / Oracle Next Education en la plataforma No Country, equipo H12-25-L-Equipo68, dic 2025 – ene 2026). El equipo construyó una API de sentimiento con un backend en Java/Spring y un motor de ML en Python.

Mi parte:
- el **modelo Bi-LSTM** y su API en **FastAPI**, con un contrato JSON para el backend en Java;
- el **deploy completo en OCI**: dos motores de IA en contenedores Docker, el backend Spring Boot y Nginx como reverse proxy, en 2 VMs comunicadas por IP privada.

Para la entrega final, el equipo eligió otro motor (ML clásico con reglas, desarrollado por otro integrante). El Bi-LSTM quedó integrado como **motor alternativo**. Este repositorio contiene solo mi trabajo.

## El modelo

- **Datos:** Andalusian Hotels Reviews. Se usa el título de cada reseña; rating 1–2 → negativo, 3 → neutro, 4–5 → positivo.
- **Aumento bilingüe:** el 25 % de las filas se tradujo al portugués con traducción automática (22.715 filas en total).
- **Arquitectura:** `Embedding(15.000 × 128)` → `SpatialDropout1D(0,4)` → `Bidirectional(LSTM(32))` → `Dense(16)` → `Dropout(0,5)` → `softmax` de 3 clases. `EarlyStopping` y pesos de clase.
- **Inferencia reproducible:** el tokenizer y los metadatos (`max_len`, etiquetas) se exportan en el entrenamiento, para que la API preprocese igual.

## La API

`POST /predict/v2/sentiment`, con el header `X-API-KEY`:

```bash
curl -X POST http://localhost:8080/predict/v2/sentiment \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: <tu-clave>" \
  -d '{"text": "El personal fue muy amable y la habitación estaba limpia"}'
```

Responde con `prevision` (clase), `probabilidad`, `texto_limpio` y `top_features`.

Decisiones de implementación:
- **`top_features` por ablación:** quita cada palabra, vuelve a predecir y devuelve las 3 cuya ausencia más reduce la probabilidad de la clase. Es una explicabilidad simple y legible, pero cuesta una inferencia por palabra.
- **TensorFlow limitado a 1 hilo**, porque las VMs tenían 1 OCPU Ampere.
- **Autenticación por API key**, leída del entorno (`.env`), nunca escrita en el código.

## Cómo ejecutarlo

1. Copia los artefactos del modelo en `data/models/` (ver [`data/models/README.md`](data/models/README.md)).
2. Configura la clave y levanta el contenedor:

```bash
cp .env.example .env      # edita API_KEY_SENTIMENT
docker compose up -d --build
```

Swagger queda en `http://localhost:8080/docs`.

## Deploy en OCI

Ene 2026, dos VMs `VM.Standard.A1.Flex` (ARM64, Oracle Linux 9):

```text
dps-vm1: Nginx (:80) ── /api/ ──► Spring Boot (:8000, systemd, H2) + frontend estático
            │ IP privada (VCN)
dps-vm2: Docker ── ml-bi-lstm (:8080, este repo) · ml-hibrido (:8081, motor del equipo)
```

Problemas resueltos: dos motores en la misma VM, comunicación por la red privada, SELinux bloqueando el JIT de Java bajo systemd y Nginx como reverse proxy. Detalle paso a paso en la [bitácora de despliegues en OCI](https://github.com/Fernando-Falla/bitacora-despliegues-oci#caso-2--sentiment-api-en-2-vms).

## Limitaciones conocidas

El deploy funcionaba; el modelo necesita reentrenarse. En la prueba de integración de punta a punta clasificó como **negativo** un texto claramente positivo. Los problemas identificados en el notebook:

1. **Fuga de datos:** la traducción al portugués se agrega antes de separar entrenamiento y validación. Una reseña y su traducción pueden quedar en conjuntos distintos, y las métricas salen optimistas.
2. **Sin conjunto de prueba independiente:** `EarlyStopping` usa la misma validación con la que se reportan los resultados.
3. **Preprocesamiento que borra señal:** se eliminan las stopwords de NLTK en español y portugués, que incluyen negaciones e intensificadores ("no", "sin", "muy", "não", "muito"). "No me gustó" queda como "gustó". Afecta el entrenamiento y la inferencia por igual, así que se corrige reentrenando, no solo en la API.
4. **Reproducibilidad:** sin semillas fijas ni versiones de librerías en el notebook.
5. **Portugués sintético:** el aumento bilingüe es traducción automática, no reseñas reales en portugués.

## Próximos pasos

- Notebook v2: separar los datos antes de traducir, test independiente, semillas, conservar las negaciones y reportar **F1 macro** (con 72 % de reseñas positivas, el accuracy engaña).
- Línea base **TF-IDF + LinearSVC** propia, evaluada sobre el mismo test, para comparar con datos.
- Publicar las métricas y los artefactos cuando ambos estén evaluados.
