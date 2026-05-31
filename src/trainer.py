import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from src.data_processor import procesar_datos_neutrales

def entrenar_y_guardar_modelos():
    """Entrena los clasificadores y regresores y los persiste en disco."""
    ruta_csv = os.path.join('data', 'results.csv')
    df_neutral, stats_neutrales = procesar_datos_neutrales(ruta_csv)

    # Definir etiquetas
    def definir_resultado(row):
        if row['home_score'] > row['away_score']: return 1
        elif row['home_score'] < row['away_score']: return 2
        else: return 0

    df_neutral['resultado'] = df_neutral.apply(definir_resultado, axis=1)

    X = df_neutral[['ataque_A', 'defensa_A', 'ataque_B', 'defensa_B']]
    y_clas = df_neutral['resultado']
    y_goles_A = df_neutral['home_score']
    y_goles_B = df_neutral['away_score']

    # Entrenar con todo el histórico disponible para máxima precisión en producción
    modelo_clasificador = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    modelo_clasificador.fit(X, y_clas)

    modelo_regresor_A = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    modelo_regresor_A.fit(X, y_goles_A)

    modelo_regresor_B = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    modelo_regresor_B.fit(X, y_goles_B)

    # Asegurar que exista la carpeta de destino
    os.makedirs('models_pkl', exist_ok=True)

    # Exportar
    joblib.dump(modelo_clasificador, os.path.join('models_pkl', 'modelo_clasificador_mundial.pkl'))
    joblib.dump(modelo_regresor_A, os.path.join('models_pkl', 'modelo_regresor_A.pkl'))
    joblib.dump(modelo_regresor_B, os.path.join('models_pkl', 'modelo_regresor_B.pkl'))
    joblib.dump(stats_neutrales, os.path.join('models_pkl', 'stats_neutrales.pkl'))
    
    print("🎯 ¡Modelos entrenados y guardados con éxito en 'models_pkl/'!")

if __name__ == '__main__':
    entrenar_y_guardar_modelos()