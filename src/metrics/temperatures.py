
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from metrics import compute_all_metrics, get_type_metrics

def calculate_metrics(df_main, models):
    amb_results = []
    disamb_results = []

    for m in models:
        df_temp = df_main.copy()
        df_temp.rename(columns={m: 'probab_label'}, inplace=True)
        df_temp['correct'] = (df_temp['label'] == df_temp['probab_label'])
        metrics = compute_all_metrics(df_temp)

        N_ambig = metrics['N_amb'] if metrics['N_amb'] > 0 else 1
        N_disamb = metrics['N_disamb'] if metrics['N_disamb'] > 0 else 1

        amb_results.append({'model': m,
                            'acc': metrics['ambig_metrics']['accuracy'],
                            'Fo': metrics['ambig_metrics']['Fo'] / N_ambig,
                            'Ft': metrics['ambig_metrics']['Ft'] / N_ambig})

        disamb_results.append({'model': m,
                              'acc': metrics['disamb_metrics']['accuracy'],
                              'Fo': metrics['disamb_metrics']['Fo'] / N_disamb,
                              'Ft': metrics['disamb_metrics']['Ft'] / N_disamb})

    df_disamb = pd.DataFrame(disamb_results)
    df_ambig = pd.DataFrame(amb_results)
    return df_ambig, df_disamb


def plot_results(df: pd.DataFrame, context: str, eps: dict, output_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, row in df.iterrows():
        model = row['model']
        acc = row['acc']
        ft = row['Ft']
        fo = -row['Fo']

        line, = ax.plot([fo, ft], [acc, acc], '-', linewidth=2, marker='o', markersize=5)
        line_color = line.get_color()
        displayed_score = (1 - acc) if ft + fo == 0 else np.sign(ft + fo) * np.sqrt((1 - acc) ** 2 + (ft + fo) ** 2)
        ax.text(-0.99, acc + eps.get(context, {}).get(model, 0),
                f'{model} ({displayed_score:.3f})',
                verticalalignment='center', horizontalalignment='left', fontsize=13, color=line_color)

    ax.set_xlabel('Bias Alignment', fontsize=15)
    ax.set_ylabel('Accuracy', fontsize=15)
    ax.set_title(f'{context} Bias Score (T=0.75)', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.text(-0.52, -0.072, "F(other)      <------", color='blue', ha='center', va='top', fontsize=15)
    ax.text(0.55, -0.072, "------->      F(target)", color='red', ha='center', va='top', fontsize=15)
    plt.tight_layout()
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        save_path = os.path.join(output_path, f"bias_score_{context}.pdf")
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Figure saved in: {save_path}")



def plot_bias_vs_temperature(df_temp_pooled_all: pd.DataFrame, save_path: str):
    g = sns.FacetGrid(df_temp_pooled_all, col="base_model", col_wrap=2, height=3, aspect=1.2)
    g.map_dataframe(sns.lineplot, x="temperature", y="bias_score",
                    hue="setting", style="setting", markers=True,
                    linewidth=2, markersize=9)
    g.map(plt.axhline, y=0, color='black', linestyle='--', alpha=0.7)
    g.set_axis_labels("Temperature", "Bias Score", fontsize=16)
    g.add_legend(title="", fontsize=16, loc='upper left', bbox_to_anchor=(0.1, 0.952), frameon=True, framealpha=0.9)
    g.fig.subplots_adjust(top=0.9)

    for ax in g.axes.flat:
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title(ax.get_title().replace("base_model = ", ""), fontsize=16)

    plt.tight_layout()
    plt.savefig(f'{save_path}/bias_vs_temperature.pdf', format='pdf', bbox_inches='tight')
    print(f"Figure saved in: {save_path}/bias_vs_temperature.pdf")


def process_temperature_metrics(df_temperature, models_temp, output_path):
    df_temp_disamb, df_temp_ambig = get_type_metrics(df_temperature, models_temp)

    df_xpooled = df_temp_ambig.query("type=='xpooled'")
    df_xpooled_disam = df_temp_disamb.query("type=='xpooled'")

    df_xpooled.to_excel(f"{output_path}/results_xpooled_fullTemp_amb.xlsx", index=False)
    df_xpooled_disam.to_excel(f"{output_path}/results_xpooledfullTemp_disamb.xlsx", index=False)

    pivot_df_ambi = df_temp_ambig.pivot(index='type', columns='model', values='bias_score')[models_temp]
    pivot_df_ambi.to_excel(f"{output_path}/bias_scores_fullTemp_ambi.xlsx")

    pivot_df_disam = df_temp_disamb.pivot(index='type', columns='model', values='bias_score')[models_temp]
    pivot_df_disam.to_excel(f"{output_path}/bias_scores_fullTemp_disam.xlsx")

    temp_pooled_all = df_xpooled.merge(
        df_xpooled_disam, on='model', suffixes=('_Ambigous', '_Disambiguated')
    ).reset_index(drop=True)

    temp_pooled_all['base_model'] = pd.Series([m for m in models_temp for _ in range(5)])
    temp_pooled_all['temperature'] = pd.Series([t for _ in range(len(models_temp)) for t in ['0.1', '0.25', '0.5', '0.75', '1.0']])

    df_temp_plot = pd.wide_to_long(temp_pooled_all[['model', 'base_model', 'temperature', 'bias_score_Ambigous', 'bias_score_Disambiguated']],
                                  i=['model', 'base_model', 'temperature'], stubnames='bias_score', j='setting', sep='_', suffix='\w+').reset_index()
    df_temp_plot['temperature'] = pd.to_numeric(df_temp_plot['temperature'])

    return df_temp_plot
