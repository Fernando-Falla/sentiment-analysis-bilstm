# Artefactos del modelo

La API carga tres archivos desde esta carpeta (montada en el contenedor como `/app/data/models`):

| Archivo | Qué es | ¿Versionado? |
|---|---|---|
| `modelo_lstm_v2.keras` | Modelo Bi-LSTM entrenado | No |
| `tokenizer.json` | Tokenizer de Keras exportado en el entrenamiento | No |
| `metadata.json` | Etiquetas de clase y `max_len` | Sí |

El modelo y el tokenizer no se publican porque salieron de un entrenamiento con problemas de rigor (ver "Limitaciones conocidas" en el README principal). Se publicarán cuando esté listo el notebook corregido.
