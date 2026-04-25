# Stroke Rehab Web — Inicio rapido

Este proyecto necesita **2 procesos** para funcionar:
- Frontend Next.js (`http://localhost:3000`)
- Backend FastAPI (`http://localhost:8000`)

## Requisitos

- Node.js 18+ (recomendado 20+)
- npm
- Python 3.10+
- pip

## 1) Instalar dependencias

Desde la carpeta `stroke-rehab-web`:

```bash
npm install
```

Para el backend (desde la raiz del repo `stroke-rehab`):

```bash
python -m pip install -r stroke-rehab-web/backend/requirements.txt
```

## 2) Levantar backend

Desde la raiz del repo `stroke-rehab`:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## 3) Levantar frontend

Desde `stroke-rehab-web`:

```bash
npm run dev
```

Abre:
- [http://localhost:3000](http://localhost:3000)

## Flujo minimo para probar

1. Cargar los 4 archivos (`pre-train`, `pre-test`, `post-train`, `post-test`)
2. Ajustar filtros/hiperparametros
3. Ejecutar `Run Analysis`

Si el frontend no conecta, verifica que el backend este corriendo en `:8000`.
