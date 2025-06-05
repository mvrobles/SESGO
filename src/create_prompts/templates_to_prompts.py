import pandas as pd
import numpy as np
import itertools
import ast
import re
import os

def fill_placeholders(text, names_dict):
    """
    Reemplaza los placeholders en el texto por los valores correspondientes en names_dict.

    Args:
        text (str): Texto con placeholders en formato {{KEY}}.
        names_dict (dict): Diccionario donde las claves son los nombres de los placeholders y los valores son listas de nombres.

    Returns:
        str: Texto con los placeholders reemplazados por los valores correspondientes.
    """

    for key, values in names_dict.items():
        text = text.replace(f'{{{{{key}}}}}', ', '.join(values))
    return text


def parse_names(names_str):
    """
    Transforma una cadena de nombres en formato 'NAME1: [nombre1, nombre2], NAME2: [nombre3, nombre4]' a un diccionario.

    Args:
        names_str (str): Cadena con los nombres y sus listas.

    Returns:
        dict: Diccionario con los nombres como claves y listas de nombres como valores.
    """

    names_str = names_str.replace(";", ",")
    matches = re.findall(r'(\w+):\s*\[([^\]]+)\]', names_str)
    names_dict = {k: [v.strip() for v in v_list.split(',')] for k, v_list in matches}
    return names_dict


def templates_to_prompts(df):
    """
        Convierte un DataFrame de templates en un DataFrame de prompts expandiendo combinaciones de nombres y preguntas.

        Args:
            df (pd.DataFrame): DataFrame con las columnas necesarias para generar los prompts.

        Returns:
            pd.DataFrame: DataFrame con los prompts generados.
    """  
    new_rows = []

    for _, row in df.iterrows():
        # Parse names into a dictionary
        names_dict = parse_names(row['Names'])
        bbq = ~np.isnan(row['Q_id'])

        for name1, name2 in zip(names_dict['NAME1'], names_dict['NAME2']):

            answer_info = ast.literal_eval(fill_placeholders(row['answer_info'], {'NAME1': [name1], 'NAME2': [name2]}))

            for question_col, polarity in [('Question_negative_stereotype', 'neg'), ('Question_non_negative', 'nonneg')]:

                new_rows.append({
                    'question_polarity': polarity,
                    'context_condition': 'ambig',
                    'category': 'clase',
                    'answer_info': answer_info,
                    'context': fill_placeholders(row['Ambiguous_Context'], {'NAME1': [name1], 'NAME2': [name2]}),
                    'question': fill_placeholders(row[question_col], {'NAME1': [name1], 'NAME2': [name2]}),
                    'answer': "No hay suficiente información" if version == 'es' else "Cannot be determined",
                    'target': 1,
                    'other': 0,
                    'label': 2,
                    'estereotipo': row['estereotipo'],
                    'bbq': bbq
                })

                new_rows.append({
                    'question_polarity': polarity,
                    'context_condition': 'disambig',
                    'category': 'clase',
                    'answer_info': answer_info,
                    'context': (fill_placeholders(row['Ambiguous_Context'], {'NAME1': [name1], 'NAME2': [name2]}) + ' ' +
                                fill_placeholders(row['Disambiguating_Context'], {'NAME1': [name1], 'NAME2': [name2]})),
                    'question': fill_placeholders(row[question_col], {'NAME1': [name1], 'NAME2': [name2]}),
                    'answer': answer_info['ans1'] if polarity == 'neg' else answer_info['ans0'],
                    'target': 1,
                    'other': 0,
                    'label': 1 if polarity == 'neg' else 0,
                    'estereotipo': row['estereotipo'],
                    'bbq': bbq
                })

                new_rows.append({
                    'question_polarity': polarity,
                    'context_condition': 'disambig',
                    'category': 'clase',
                    'answer_info': answer_info,
                    'context': (fill_placeholders(row['Ambiguous_Context'], {'NAME1': [name1], 'NAME2': [name2]}) + ' ' +
                                fill_placeholders(row['Disambiguating_Context'], {'NAME1': [name2], 'NAME2': [name1]})),
                    'question': fill_placeholders(row[question_col], {'NAME1': [name1], 'NAME2': [name2]}),
                    'answer': answer_info['ans0'] if polarity == 'neg' else answer_info['ans1'],
                    'target': 1,
                    'other': 0,
                    'label': 0 if polarity == 'neg' else 1,
                    'estereotipo': row['estereotipo'],
                    'bbq': bbq
                })

    return pd.DataFrame(new_rows)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Procesa archivos de templates a prompts.")
    parser.add_argument("--templates_dir", type=str, default="templates", help="Directorio con archivos de templates (Excel)")
    parser.add_argument("--version", type=str, choices=["en", "es"], required=True, help="Idioma: 'en' o 'es'")
    parser.add_argument("--output_dir", type=str, default="prompts", help="Directorio de salida para los archivos procesados")
    args = parser.parse_args()

    templates_dir = args.templates_dir
    version = args.version
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(templates_dir):
        if filename.endswith(".xlsx"):
            filepath = os.path.join(templates_dir, filename)
            print(f"Procesando {filepath} (hoja: {version})...")
            df = pd.read_excel(filepath, sheet_name=version)
            prompts_df = templates_to_prompts(df)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_prompts_{version}.xlsx", index=None)
            prompts_df.to_excel(output_path, index=False)
            print(f"Guardado en {output_path}")


# Example usage:
#python templates_to_prompts.py --version es