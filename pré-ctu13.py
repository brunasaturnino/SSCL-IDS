import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_your_dataset(input_paths, output_path):
    all_dfs = []

    for path in input_paths:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()  # remove espaços extras dos nomes de coluna
        all_dfs.append(df)

    # Concatena todos os DataFrames
    df = pd.concat(all_dfs, ignore_index=True)

    # Remove colunas que têm todos os valores iguais (sem informação)
    nunique = df.nunique()
    constant_cols = nunique[nunique <= 1].index
    df.drop(columns=constant_cols, inplace=True)

    # Substitui valores infinitos e ausentes
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Assegura que Label está no formato correto
    df['Label'] = df['Label'].apply(lambda x: 1 if str(x).strip() == '1' else 0)

    # Separa X e y
    X = df.drop(columns=['Label'])
    y = df['Label']

    # Normaliza os dados
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Junta X e y novamente
    final_df = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)

    # Salva o arquivo processado
    final_df.to_csv(output_path, index=False)
    print(f"✅ Dataset processado salvo em: {output_path}")

# Exemplo de uso:
input_paths = [
    'Dataset/CTU13_Attack_Traffic.csv',
    'Dataset/CTU13_Normal_Traffic.csv',
]

output_path = 'Dataset/dataset-ctu13.csv'

preprocess_your_dataset(input_paths, output_path)

