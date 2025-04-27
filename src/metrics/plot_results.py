import matplotlib.pyplot as plt
import numpy as np

def plot_results(df, context):

    eps = {'Ambiguous': {'GPT-4o mini': 0.01, 'Llama 3.1 Instruct': 0, 'Llama 3.1 Uncensored': -0.0, 'DeepSeek R1': -0.017, 'Gemini 2.0 Flash': 0.017, 'Claude 3.5 Haiku': -0.01},
           'Disambiguated': {'GPT-4o mini': 0.02, 'Llama 3.1 Instruct': -0.017, 'Llama 3.1 Uncensored': 0.0, 'DeepSeek R1': 0.0, 'Gemini 2.0 Flash': 0.017, 'Claude 3.5 Haiku': 0.0}}


    fig, ax = plt.subplots(figsize=(10, 6))

    # For each model in the DataFrame
    for i, row in df.iterrows():
        model = row['model']
        acc = row['acc']
        ft = row['Ft']
        fo = -row['Fo']

        # Plot a horizontal line from -Fo to Ft at the accuracy level
        line, = ax.plot([fo, ft], [acc, acc], '-', linewidth=2, marker='o', markersize=5)


        # Add model name as text label at the leftmost part of the plot (-1)
        # fo is already negative
        line_color = line.get_color()
        ax.text(-0.99, acc + eps[context][model], f'{model} ({np.sign(ft+fo)*np.sqrt((1-acc)**2 + (ft + fo)**2):.3f})',
                verticalalignment='center', horizontalalignment='left', fontsize = 13, color = line_color)

    # Set labels and title
    ax.set_xlabel('Bias Alignment', fontsize = 15)
    ax.set_ylabel('Accuracy', fontsize = 15)
    ax.set_title(f'{context} Bias Score (T=0.75)', fontsize = 16)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)

    # Add a vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Add "Fo" and "Ft" labels below the x-axis at -1 and +1
    ax.text(-0.52, -0.072, "F(other)      <------", color = 'blue', ha='center', va='top', fontsize = 15)
    ax.text(0.55, -0.072, "------->      F(target)", color = 'red', ha='center', va='top', fontsize = 15)

    plt.show()