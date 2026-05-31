from fastapi import FastAPI, Query
from src.predictor import SimuladorMundial

app = FastAPI(
    title="Mundial 2026 Forecasting API",
    description="API de Clean Architecture para simular pronósticos probabilísticos en cancha neutral."
)

# Inicializamos el simulador (Carga los modelos una sola vez al encender el servidor)
simulador = SimuladorMundial()

@app.get("/")
def inicio():
    return {"mensaje": "¡Bienvenido a la API del Mundial 2026! Ve a /docs para probar los pronósticos."}

@app.get("/api/v1/pronostico")
def obtener_pronostico(
    equipo1: str = Query(..., description="Nombre del primer equipo (ej: Argentina)"),
    equipo2: str = Query(..., description="Nombre del segundo equipo (ej: France)")
):
    # Capitalizar nombres para evitar errores de tipeo del usuario
    eq1 = equipo1.strip().title()
    eq2 = equipo2.strip().title()
    
    resultado = simulador.simular(eq1, eq2)
    return resultado