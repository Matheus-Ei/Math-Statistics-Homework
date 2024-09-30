import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# Dados fornecidos na imagem
data = [
    58, 152, 80, 57, 126, 136, 96, 144, 108, 86,
    109, 82, 75, 148, 114, 131, 66, 106, 121, 158,
    64, 105, 118, 73, 83, 81, 104, 60, 111, 94,
    100, 110, 125, 130, 123, 105, 140, 104, 96, 115,
    120, 94, 146, 136, 102, 132, 87, 127, 100
]

# Passo 1: Calcular o número de classes pela fórmula de Sturges
n = len(data)
k = math.ceil(1 + 3.33 * math.log10(n))

# Passo 2: Definir o intervalo de classe (amplitude)
min_val = min(data)
max_val = max(data)
amplitude = math.ceil((max_val - min_val) / k)

# Passo 3: Criar as classes
bins = [min_val + i * amplitude for i in range(k+1)]

# Passo 4: Criar a tabela de frequências
freq_abs, bin_edges = np.histogram(data, bins=bins)
freq_rel = freq_abs / n
freq_acum = np.cumsum(freq_abs)
freq_rel_perc = freq_rel * 100
freq_acum_perc = np.cumsum(freq_rel_perc)
mid_points = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]

# Passo 5: Criar DataFrame
df = pd.DataFrame({
    'Classes': [f'{int(bin_edges[i])} - {int(bin_edges[i+1])}' for i in range(len(bin_edges)-1)],
    'Ponto Médio (xmi)': mid_points,
    'Frequência Absoluta (fi)': freq_abs,
    'Frequência Relativa (fri)': freq_rel,
    'Frequência Acumulada (fai)': freq_acum,
    'Frequência Relativa % (frpi)': freq_rel_perc,
    'Frequência Acumulada % (fapi)': freq_acum_perc
})

# (b) Construir o histograma da distribuição de frequência
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
axs[0].hist(data, bins=bins, edgecolor='black', alpha=0.7)
axs[0].set_title('Histograma da Distribuição de Frequência')
axs[0].set_xlabel('Classes (KWH)')
axs[0].set_ylabel('Frequência Absoluta')
axs[0].grid(axis='y', linestyle='--', alpha=0.7)

# (c) Curva polida da distribuição de frequência
axs[1].plot(mid_points, freq_abs, marker='o', linestyle='-', color='blue')
axs[1].set_title('Curva Polida da Distribuição de Frequência')
axs[1].set_xlabel('Ponto Médio (xmi)')
axs[1].set_ylabel('Frequência Absoluta')
axs[1].grid(axis='both', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# (d) Média residual (média dos valores)
media_residual = np.mean(data)

# (e) Mediana (Md), Quartis (Q1, Q3), Percentis (P10, P90)
Md = np.median(data)
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
P10 = np.percentile(data, 10)
P90 = np.percentile(data, 90)

# (f) Modas: Bruta, Czuber, King
# Moda Bruta: valor que mais aparece
moda_bruta = max(set(data), key=data.count)

# Moda de Czuber: aproximada usando fórmula
h = amplitude  # amplitude da classe

f1 = freq_abs[np.argmax(freq_abs)]
f0 = freq_abs[np.argmax(freq_abs) - 1] if np.argmax(freq_abs) > 0 else 0
f2 = freq_abs[np.argmax(freq_abs) + 1] if np.argmax(freq_abs) < len(freq_abs) - 1 else 0
moda_czuber = mid_points[np.argmax(freq_abs)] + (h * (f1 - f0)) / (2 * f1 - f0 - f2)

# Moda de King: aproximação
moda_king = mid_points[np.argmax(freq_abs)] + (h * (f1 - f0)) / (f1 + f0)

# (g) Desvio padrão
desvio_padrao = np.std(data)

# (h) Coeficiente de assimetria de Pearson (usando a moda bruta)
coef_pearson = 3 * (media_residual - moda_bruta) / desvio_padrao

# Identificar tipo de assimetria
assimetria = 'assimetria positiva' if coef_pearson > 0 else 'assimetria negativa' if coef_pearson < 0 else 'simetria'


resultados = {
    'Média Residual (KWH)': media_residual,
    'Mediana (Md)': Md,
    'Q1 (1º Quartil)': Q1,
    'Q3 (3º Quartil)': Q3,
    'P10': P10,
    'P90': P90,
    'Moda Bruta': moda_bruta,
    'Moda Czuber': moda_czuber,
    'Moda King': moda_king,
    'Desvio Padrão': desvio_padrao,
    'Coeficiente de Assimetria de Pearson': coef_pearson,
    'Tipo de Assimetria': assimetria
}

resultados_df = pd.DataFrame(resultados.items(), columns=['Medida', 'Valor'])


print(resultados_df)
print(df)
