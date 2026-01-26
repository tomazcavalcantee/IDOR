import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter

# 1. Carga e Limpeza de Dados
# O r'' evita o erro de 'invalid escape sequence'
path = '/home/al.tomaz.cavalcante/python/IDOR/Questão3_atv1/desmame.txt'
df = pd.read_csv(path, sep=r'\s+')
df.columns = df.columns.str.strip()

# --- (a) ANÁLISE DESCRITIVA AMPLIADA ---

print("--- 1. Estatísticas dos Tempos por Censura ---")
# Comparando medidas de tendência central e dispersão
desc_cens = df.groupby('cens')['tempo'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
print(desc_cens)

# Visualização da Distribuição
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='tempo', hue='cens', kde=True, bins=15, palette='viridis', element="step")
plt.title('Distribuição de Densidade dos Tempos (Observados vs Censurados)')
plt.xlabel('Tempo (meses)')
plt.show()

print("\n--- 2. Curvas de Sobrevivência (Kaplan-Meier) ---")
# Essencial para entender a probabilidade de continuar amamentando ao longo do tempo
kmf = KaplanMeierFitter()
plt.figure(figsize=(10, 6))

# Curva Global
kmf.fit(df['tempo'], event_observed=df['cens'], label='Curva de Aleitamento Materno')
kmf.plot_survival_function(at_risk_counts=True)
plt.title('Função de Sobrevivência Global (Método Kaplan-Meier)')
plt.ylabel('Probabilidade de Aleitamento')
plt.grid(alpha=0.3)
plt.show()

print("\n--- 3. Análise das Covariáveis (V1 a V11) ---")
# Verificando a prevalência de cada fator (0 ou 1)
v_cols = [f'V{i}' for i in range(1, 12)]
proporcoes = df[v_cols].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 4))
proporcoes.plot(kind='bar', color='skyblue')
plt.title('Proporção de Casos com Presença do Fator (Valor = 1)')
plt.ylabel('Frequência Relativa')
plt.show()

# Matriz de Correlação para checar colinearidade (causadora de erros no modelo)
plt.figure(figsize=(10, 8))
sns.heatmap(df[v_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação entre Covariáveis')
plt.show()

# --- (b) MODELO DE COX PENALIZADO (LASSO) ---

# Criação de Interações (Excluindo as que possuem variância quase nula)
df_inter = df.copy()
for i in range(len(v_cols)):
    for j in range(i + 1, len(v_cols)):
        col_name = f"{v_cols[i]}_{v_cols[j]}"
        inter_val = df[v_cols[i]] * df[v_cols[j]]
        if inter_val.var() > 0.05: # Filtro para evitar singularidade de matriz
            df_inter[col_name] = inter_val

# Ajuste com penalidade LASSO mais forte para garantir convergência
df_model = df_inter.drop(columns=['id'])
cph = CoxPHFitter(penalizer=0.2, l1_ratio=1.0) 
cph.fit(df_model, duration_col='tempo', event_col='cens')

print("\n--- 4. Variáveis Selecionadas pelo Modelo ---")
selecionados = cph.params_[cph.params_ != 0].sort_values()
if not selecionados.empty:
    print(selecionados)
else:
    print("Nenhuma variável sobreviveu à penalidade Lasso com este valor de penalizer.")

# --- (c) ERROS PADRÃO ---
# Armazenando os erros padrão dos coeficientes selecionados
print("\nErros Padrão das Estimativas:")
print(cph.standard_errors_[cph.params_ != 0])