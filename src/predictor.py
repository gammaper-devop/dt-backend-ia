import os
import joblib
import pandas as pd

class SimuladorMundial:
    def __init__(self):
        # Cargar los modelos al inicializar la clase en el backend
        self.model_clas = joblib.load(os.path.join('models_pkl', 'modelo_clasificador_mundial.pkl'))
        self.model_reg_A = joblib.load(os.path.join('models_pkl', 'modelo_regresor_A.pkl'))
        self.model_reg_B = joblib.load(os.path.join('models_pkl', 'modelo_regresor_B.pkl'))
        self.stats_data = joblib.load(os.path.join('models_pkl', 'stats_neutrales.pkl'))

    def simular(self, equipo_1: str, equipo_2: str) -> dict:
        stats_1 = self.stats_data[self.stats_data['equipo'] == equipo_1]
        stats_2 = self.stats_data[self.stats_data['equipo'] == equipo_2]
        
        if stats_1.empty or stats_2.empty:
            return {"error": f"Uno de los equipos ({equipo_1} o {equipo_2}) no está registrado."}

        # Vectores espejo
        cara_1 = pd.DataFrame([{'ataque_A': stats_1['goles_ataque_promedio'].values[0], 'defensa_A': stats_1['goles_defensa_promedio'].values[0],
                                'ataque_B': stats_2['goles_ataque_promedio'].values[0], 'defensa_B': stats_2['goles_defensa_promedio'].values[0]}])
        cara_2 = pd.DataFrame([{'ataque_A': stats_2['goles_ataque_promedio'].values[0], 'defensa_A': stats_2['goles_defensa_promedio'].values[0],
                                'ataque_B': stats_1['goles_ataque_promedio'].values[0], 'defensa_B': stats_1['goles_defensa_promedio'].values[0]}])

        # 1. Clasificación (Porcentajes)
        p1 = self.model_clas.predict_proba(cara_1)[0]
        p2 = self.model_clas.predict_proba(cara_2)[0]
        prob_empate = float((p1[0] + p2[0]) / 2 * 100)
        prob_eq1 = float((p1[1] + p2[2]) / 2 * 100)
        prob_eq2 = float((p1[2] + p2[1]) / 2 * 100)

        if prob_eq1 + prob_empate > 72 and prob_eq1 > prob_eq2:
            doble_oportunidad = f"Gana {equipo_1} o Empate"
        elif prob_eq2 + prob_empate > 72 and prob_eq2 > prob_eq1:
            doble_oportunidad = f"Empate o Gana {equipo_2}"
        else:
            doble_oportunidad = f"Gana {equipo_1} o Gana {equipo_2} (Sin Empate)"

        # 2. Regresión (Goles con Filtro de Coherencia)
        goles_1_raw = (self.model_reg_A.predict(cara_1)[0] + self.model_reg_B.predict(cara_2)[0]) / 2
        goles_2_raw = (self.model_reg_B.predict(cara_1)[0] + self.model_reg_A.predict(cara_2)[0]) / 2
        
        goles_1_round = int(round(goles_1_raw))
        goles_2_round = int(round(goles_2_raw))

        if goles_1_round == goles_2_round:
            if prob_eq1 - prob_eq2 > 5: goles_1_round += 1
            elif prob_eq2 - prob_eq1 > 5: goles_2_round += 1

        goles_totales = goles_1_round + goles_2_round

        # 3. Formatear respuestas del flyer
        mas_menos = "Más de 3.5 goles" if goles_totales > 3.5 else ("Más de 2.5 goles" if goles_totales > 2.5 else ("Menos de 1.5 goles" if goles_totales < 1.5 else "Menos de 2.5 goles"))
        ambos_anotan = "SÍ" if goles_1_round > 0 and goles_2_round > 0 else "NO"

        return {
            "equipo_1": equipo_1,
            "equipo_2": equipo_2,
            "doble_oportunidad": doble_oportunidad,
            "probabilidad_1": round(prob_eq1, 1),
            "probabilidad_empate": round(prob_empate, 1),
            "probabilidad_2": round(prob_eq2, 1),
            "mas_menos_goles": mas_menos,
            "ambos_anotan": ambos_anotan,
            "marcador_exacto": f"{goles_1_round} - {goles_2_round}"
        }