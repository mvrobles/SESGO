import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from compute_metrics import get_type_metrics
import os


def load_and_process_metrics(df_english, df_spanish, models,output_path=None):
    """
    Processes metrics of type 'ambig' and 'disamb', and returns a dictionary with dataframes filtered by 'xpooled'.

    Returns:
        dict: {
            'ambig': {'en': df_xpooled_en_amb, 'es': df_xpooled_es_amb},
            'disamb': {'en': df_xpooled_en_disam, 'es': df_xpooled_es_disam}
        }
    """
    df_en_disamb, df_en_ambig = get_type_metrics(df_english, models)
    df_es_disamb, df_es_ambig = get_type_metrics(df_spanish, models)

    df_metrics = {
        'ambig': {'en': df_en_ambig, 'es': df_es_ambig},
        'disamb': {'en': df_en_disamb, 'es': df_es_disamb}
    }

    results = {}

    for metric_type in ['ambig', 'disamb']:
        df_en = df_metrics[metric_type]['en'].query("type == 'xpooled'")
        df_es = df_metrics[metric_type]['es'].query("type == 'xpooled'")

        if output_path:
            df_en.to_excel(os.path.join(output_path, f"{metric_type}_english.xlsx"), index=False)
            df_es.to_excel(os.path.join(output_path, f"{metric_type}_spanish.xlsx"), index=False)

        results[metric_type] = {
            'en': df_en,
            'es': df_es
        }


    return results

def get_merged_bias_scores(df_en_type, df_es_type, df_es_full_type, bias_label_prefix):
    """
    Join the English and Spanish DataFrames by model/type and add the full Spanish data.

    Args:
        df_en_type (pd.DataFrame): English filtered DF (specific type, e.g., 'ambig').
        df_es_type (pd.DataFrame): Filtered DF in Spanish (specific type).
        df_es_full_type (pd.DataFrame): DF with full Spanish data (specific type).
        bias_label_prefix (str): Prefix to rename columns ('amb' or 'disamb').

    Returns:
        pd.DataFrame: DataFrame combined with renamed columns.
    """

    enes = df_en_type.merge(df_es_type, on=['model', 'type'], suffixes=('_en', '_es'))

    enes_full = enes.merge(
        df_es_full_type[['model', 'type', 'bias_score']],
        on=['model', 'type'],
        how='left'
    )

    enes_full.rename(columns={
        'bias_score_en': f'{bias_label_prefix}_bias_english',
        'bias_score_es': f'{bias_label_prefix}_bias_spanish',
        'bias_score': f'{bias_label_prefix}_bias_spanish_full'
    }, inplace=True)

    return enes_full[[ 'model', f'{bias_label_prefix}_bias_english',
                       f'{bias_label_prefix}_bias_spanish',
                       f'{bias_label_prefix}_bias_spanish_full']]



def plot_bias_scores(df, bias_prefix, title=None, output_path=None):
    """
    Graph bias scores with given prefix, for English and Spanish (matched and full).
    """

    x_pos = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#36A2EB','#FFCE56','#228B22','#FF6384','#9966FF','#A0522D'] * 2

    for i, (_, row) in enumerate(df.iterrows()):
        color = colors[i % len(colors)]
        eng = row[f'{bias_prefix}_bias_english']
        esp = row[f'{bias_prefix}_bias_spanish']
        esp_full = row[f'{bias_prefix}_bias_spanish_full']

        ax.scatter(x_pos[i], eng, color=color, s=100, edgecolors='black', linewidth=1.2, zorder=3)
        ax.scatter(x_pos[i], esp, color=color, marker='s', s=100, edgecolors='black', linewidth=1.2, zorder=3)
        ax.hlines(y=esp_full, xmin=x_pos[i] - 0.1, xmax=x_pos[i] + 0.1, colors=color, linestyles='-', linewidth=2, zorder=1)
        ax.annotate('', xy=(x_pos[i], esp), xytext=(x_pos[i], eng), arrowprops=dict(facecolor=color, arrowstyle='->', lw=2), zorder=2)

        if esp < eng:
            ax.text(x_pos[i], esp - 0.06, f"{esp:.2f}", fontsize=11, ha='center', va='top')
            va_pos_eng, offset_eng = 'bottom', 0.06
        else:
            ax.text(x_pos[i], esp + 0.06, f"{esp:.2f}", fontsize=11, ha='center', va='bottom')
            va_pos_eng, offset_eng = 'top', -0.06

        ax.text(x_pos[i], eng + offset_eng, f"{eng:.2f}", fontsize=11, ha='center', va=va_pos_eng)
        ax.text(x_pos[i] + 0.15, esp_full, f"{esp_full:.2f}", fontsize=11, ha='left', va='center')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(df['model'], rotation=45, ha='right', fontsize=12)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.7)

    y_min = df[[f'{bias_prefix}_bias_english', f'{bias_prefix}_bias_spanish', f'{bias_prefix}_bias_spanish_full']].min().min() - 0.2
    y_max = df[[f'{bias_prefix}_bias_english', f'{bias_prefix}_bias_spanish', f'{bias_prefix}_bias_spanish_full']].max().max() + 0.2
    ax.set_ylim(y_min, y_max)

    ax.set_ylabel('Bias Score', fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='English', markerfacecolor='gray', markeredgecolor='black', markersize=10),
        plt.Line2D([0], [0], marker='s', color='w', label='Spanish (matched)', markerfacecolor='gray', markeredgecolor='black', markersize=10),
        plt.Line2D([0], [0], color='black', lw=2, label='Spanish (full)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.grid(axis='y', linestyle='--', alpha=0.2)
    ax.set_xlim(-0.5, len(df) - 0.5 + 0.3)
    plt.tight_layout()
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        file_path = os.path.join(output_path, f"{bias_prefix}_Multilingual_bias_scores.pdf")
        plt.savefig(file_path, format='pdf', bbox_inches='tight')
        print(f"Figure saved in: {file_path}")


