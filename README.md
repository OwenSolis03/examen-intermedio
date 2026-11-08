# Examen intermedio · Reconocimiento de Patrones (Examen B)

**Owen Adiel Solis Zatarain**

| Parámetro | Valor |
| :--- | :--- |
| Expediente | 223201053 |
| a (último dígito) | 3 |
| b (penúltimo dígito) | 5 |
| Semilla | 223201053 |

## Contenido

- [`problema-1/`](problema-1/): *El reporte demasiado bueno* (Bank Marketing). Notebook, gráficas y [README](problema-1/README.md) con resultados y conclusiones.
- [`problema-2/`](problema-2/): *Árboles que cortan en diagonal* (árbol oblicuo; datos de vino tinto). Notebook, gráficas y [README](problema-2/README.md) con resultados y conclusiones.
- `reporte.tex` y `reporte.pdf`: parte teórica, carta al banco, bitácoras de decisiones y declaración de uso de IA.

## Cómo reproducir todo

Hace falta [uv](https://docs.astral.sh/uv/) y conexión a internet: los datos se descargan del UCI Machine Learning Repository desde el propio código.

```bash
uv sync
uv run --with jupyter jupyter nbconvert --to notebook --execute --inplace problema-1/problema_1.ipynb
uv run --with jupyter jupyter nbconvert --to notebook --execute --inplace problema-2/problema_2.ipynb
```

- **`uv sync`:** crea el entorno con las versiones fijadas en `pyproject.toml` y `uv.lock`.
- **Los otros dos comandos:** ejecutan cada notebook de principio a fin y guardan sus gráficas en la carpeta `figuras/` del problema. Cada uno tarda alrededor de 1–2 minutos.

También se pueden abrir los notebooks en VS Code o Jupyter con el kernel de `.venv` y usar *Run All*.

Para compilar el reporte se necesita una distribución de LaTeX:

```bash
pdflatex reporte.tex
pdflatex reporte.tex
```
