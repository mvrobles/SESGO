import pandas as pd

def get_category_number(answer: str):
    """
    Determina la categoría de la respuesta basada en la presencia de las opciones "option 0", "option 1" u "option 2" en el texto.

    Args:
        answer (str): Texto de la respuesta generada por el modelo.

    Returns:
        int or str: Devuelve 0, 1, o 2 si identifica una única opción, o 'Rev' si no es posible determinar la opción de manera clara.
    """
    try:
        answer = answer.lower()
        answer = answer.split('correct answer')[-1]
        
        if ('option 0' in answer) & ('option 1' not in answer) & ('option 2' not in answer):
            return 0
        elif ('option 1' in answer) & ('option 0' not in answer) & ('option 2' not in answer):
            return 1
        elif ('option 2' in answer) & ('option 1' not in answer) & ('option 0' not in answer):
            return 2
        else:
            return 'Rev'
        
    except:
        return "Rev"
    
def process_answers(df: pd.DataFrame):
    """
    Procesa un DataFrame con respuestas de modelo y asigna una etiqueta ('probab_label') para cada fila.

    Args:
        df (pd.DataFrame): DataFrame que debe contener las columnas 'result' (respuesta del modelo) y 'answer_info' (diccionario de respuestas posibles).

    Returns:
        None: Modifica el DataFrame en el lugar, agregando la columna 'probab_label' con los valores 0, 1, 2 o 'Rev'.
    """
    
    df['probab_label'] = None

    for i in range(len(df)):

        df['probab_label'][i] = get_category_number(df['result'][i])

        if df['probab_label'][i] == 'Rev':
            texto = df['result'][i]
            diccionario = df['answer_info'][i]
            
            # Buscar la clave que contiene la respuesta
            c = 0
            for key, value in diccionario.items():
                if value in texto:
                    c+=1
                    # Extraer el número de la clave (ans0, ans1, ans2)
                    df['probab_label'][i] = int(key[3:])

                if (c > 1) | ('Wait,' in texto):
                    df['probab_label'][i] = 'Rev'


## El modelo usado para DeepSeek puede presentar alucinaciones
# Por lo tanto, es importante revisar las respuestas generadas 
# y asegurarse de que sean válidas. Este proceso puede no ser automático.               