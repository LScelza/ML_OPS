"""
En este archivo se encuentras seis endpoints, el ultimo da respuesta al sistema de recomendacion de
videojuegos tomando como referencia el tipo de juego.
"""
import pandas as pd
from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
 
app = FastAPI()

@app.get('/')
async def my_function():
    return 'PROYECTO INDIVIDUAL Nº1 Machine Learning Operations (MLOps)'

# ENDPOINTS

"""
1. Cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora.
"""

@app.get("/developer/{desarrollador}")
async def desarrollador(desarrollador:str):
    
    df1 = pd.read_parquet('./Dataset/steam_games.parquet')
    df_desarrollador = df1[df1['developer'] == desarrollador.capitalize()]
    for index, row in df_desarrollador.iterrows():
        if  row['price'] == 0:
            df_resultado = pd.DataFrame()
            df_resultado['Años'] = df_desarrollador['release_date'].unique()
            df_resultado = df_desarrollador.groupby('release_date')['price'].value_counts().reset_index()
            df_resultado = df_resultado[df_resultado['price'] == 0]
            df_resultado['count_total'] = df_desarrollador.groupby('release_date').size().values
            df_resultado['porcentaje_free'] = round(df_resultado['count']/df_resultado['count_total']*100,2)
            resultado_dict = df_resultado[['release_date', 'count', 'count_total', 'porcentaje_free']].to_dict(orient='records')
            return resultado_dict
        else:        
            return 'No tiene juegos free'

"""
2. Debe devolver cantidad de dinero gastado por el usuario, 
el porcentaje de recomendación en base a reviews.recommend y cantidad de items.       
"""

@app.get('/User_id/{user_id}')
async def userdata(user_id: str):
    df2 = pd.read_parquet('./Dataset/endpoint_2.parquet')
    df_user_id = df2[df2['user_id'] == user_id]
    dinero_gastado= (df_user_id['price'].sum())
    porcentaje_recomen = len(df_user_id['recommend']=='True')/len(df_user_id)*100
    cant_items = df_user_id.iloc[0,3].astype(float)
    
    return {'usuario':user_id, 'Dinero gastado':dinero_gastado, 'porcentaje de recomendación':porcentaje_recomen, 'Cantidad de items': cant_items}

"""
3. Debe devolver el usuario que acumula más horas jugadas 
para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.
"""

@app.get('/Genero/{genero}')
async def UserForGenre(genero: str):
    df3 = pd.read_parquet('./Dataset/endpoint_3.parquet')
    data = df3[df3['genres'] == genero.capitalize()]
    usuario_horas = data.groupby('user_id')['playtime_forever'].sum().idxmax(0)
    lista_horas = data.groupby('release_date')['playtime_forever'].sum().reset_index() 
    resultado ={'Usuario con más horas jugadas para ' + genero: usuario_horas,
              'Horas jugadas': [{'Año': int(row['release_date']),  'Horas':int(row['playtime_forever'])} for index, row in lista_horas.iterrows()]
  }
    return resultado

"""
4. Devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado. 
(reviews.recommend = True y comentarios positivos)
"""

@app.get('/Año')
async def best_developer_year(año:int ): 
    df4 = pd.read_parquet('./Dataset/endpoint_4_5.parquet')
    df4['release_date'] = df4['release_date'].astype(int)
    data = df4[df4['release_date']== año]
    data = data[(data['recommend'] == True) & (data['sentiment_analysis'] == 2)].sort_values(by= 'developer',ascending= False)
    df_dvelopers = data.groupby('developer')['sentiment_analysis'].sum().index[:3]
    lista = df_dvelopers.to_list()
    resultado_dict = {'Primer puesto': lista[0], 'Segundo puesto': lista[1], 'Tercer puesto': lista[2]}
    return resultado_dict    

"""
5. Según el desarrollador, se devuelve un diccionario con el nombre del desarrollador como llave y una lista con la cantidad total 
de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor positivo o negativo
"""   

@app.get('/Desarrolladora/{desarrolladora}')
async def developer_reviews_analysis(desarrolladora: str ):
    df = pd.read_parquet('./Dataset/endpoint_4_5.parquet')
    df_desarrolladora = df[df['developer'] == desarrolladora.capitalize()]
    positivos = (df_desarrolladora['sentiment_analysis'] == 2).sum()
    negativos = (df_desarrolladora['sentiment_analysis'] == 0).sum()
    return {desarrolladora:f'[Negative = {negativos}, Positive = {positivos}]'}

"""
6. Ingresando el id de producto, deberíamos recibir una lista 
con 5 juegos recomendados similares al ingresado.
"""

@app.get('/Recomendacion_juego/{id_juego}')  
async def recomendacion_juego(id_juego:int):
    df = pd.read_parquet('./Dataset/recomendacion.parquet')
    # Verifica si existe el id.
    if id_juego not in df['item_id'].values:
        return "ID de juego no encontrado"
    # Vectoriza (convierte texto en valores numéricos).
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    # Convierte 'genres' a cadena y lo codifica con LabelEncoder.
    label_encoder = LabelEncoder()
    df['genres_str'] = label_encoder.fit_transform(df['genres'].astype(str))
    # Combina 'genres_str', 'title', 'sentiment_analysis', y 'playtime_2weeks' en una nueva columna.
    # Esto es para generar vectores y comparar cosenos.
    df['combined_features'] = (
        df['genres_str'].astype(str) + ' ' +
        df['title'] + ' ' +
        df['sentiment_analysis'].astype(str) + ' ' +
        df['playtime_2weeks'].astype(str)
    )
    # Aplica el vectorizador a la nueva columna.
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])
    # Escala 'sentiment_analysis' y 'playtime_2weeks'.
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[['sentiment_analysis', 'playtime_2weeks']])
    # Agrega las características escaladas.
    tfidf_matrix = pd.concat([pd.DataFrame(tfidf_matrix.toarray()), pd.DataFrame(scaled_features)], axis=1)
    # Calcula la similitud de coseno entre los juegos.
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    # Obtiene el índice del juego en el DataFrame.
    idx = df[df['item_id'] == id_juego].index[0]
    # Obtiene las similitudes de coseno para el juego especificado.
    cosine_scores = list(enumerate(cosine_similarities[idx]))
    # Ordena los juegos por similitud de coseno. Cuanto más cercana a 1, más "parecido" es.
    cosine_scores = sorted(cosine_scores, key=lambda x: x[1], reverse=True)
    # Obtiene los índices de los 5 juegos recomendados (excluyendo el juego actual) por similitud.
    recommended_indices = [i[0] for i in cosine_scores[1:6]]  
    # Obtiene los títulos de los juegos recomendados.
    recommended_titles = df['title'].iloc[recommended_indices].tolist()
    return recommended_titles

    


    
    
    