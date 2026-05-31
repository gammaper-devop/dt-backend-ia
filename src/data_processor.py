import pandas as pd

def procesar_datos_neutrales(ruta_csv: str):
    """Carga, limpia y genera las estadísticas neutrales del dataset."""
    df = pd.read_csv(ruta_csv)
    df['date'] = pd.to_datetime(df['date'])

    # Filtrar era moderna
    df_moderno = df[df['date'].dt.year >= 2014].copy()
    df_moderno['home_team'] = df_moderno['home_team'].str.strip()
    df_moderno['away_team'] = df_moderno['away_team'].str.strip()

    # Desarmar local/visitante
    goles_como_local = df_moderno[['home_team', 'home_score', 'away_score']].rename(
        columns={'home_team': 'equipo', 'home_score': 'goles_anotados', 'away_score': 'goles_recibidos'}
    )
    goles_como_visitante = df_moderno[['away_team', 'away_score', 'home_score']].rename(
        columns={'away_team': 'equipo', 'away_score': 'goles_anotados', 'home_score': 'goles_recibidos'}
    )
    historial_total = pd.concat([goles_como_local, goles_como_visitante])

    # Agrupar estadísticas netas
    stats_neutrales = historial_total.groupby('equipo').agg(
        goles_ataque_promedio=('goles_anotados', 'mean'),
        goles_defensa_promedio=('goles_recibidos', 'mean')
    ).reset_index()

    # Reconstruir dataset simétrico
    df_neutral = df_moderno.merge(stats_neutrales, left_on='home_team', right_on='equipo', how='left')
    df_neutral = df_neutral.rename(columns={'goles_ataque_promedio': 'ataque_A', 'goles_defensa_promedio': 'defensa_A'}).drop(columns=['equipo'])

    df_neutral = df_neutral.merge(stats_neutrales, left_on='away_team', right_on='equipo', how='left')
    df_neutral = df_neutral.rename(columns={'goles_ataque_promedio': 'ataque_B', 'goles_defensa_promedio': 'defensa_B'}).drop(columns=['equipo'])
    df_neutral = df_neutral.fillna(0)

    return df_neutral, stats_neutrales