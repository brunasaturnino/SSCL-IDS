import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_cicids_dataset(input_paths, output_path):
    all_dfs = []

    for path in input_paths:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()  # remove espaços extras
        all_dfs.append(df)

    # Concatena os DataFrames
    df = pd.concat(all_dfs, ignore_index=True)

    # Substitui valores infinitos e ausentes
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Converte coluna Label: BENIGN -> 0, outros -> 1
    df['Label'] = df['Label'].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)

    # Seleciona apenas colunas numéricas (exceto IPs, protocolos, etc.)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Label' not in numeric_cols:
        numeric_cols.append('Label')

    # Separa X e y
    X = df[numeric_cols].drop(columns=['Label'])
    y = df['Label']

    # Normaliza com StandardScaler
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Junta X e y novamente
    final_df = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)

    # Salva o CSV final
    final_df.to_csv(output_path, index=False)
    print(f"Dataset processado salvo em {output_path}")

# Exemplo de uso:
input_paths = [
    'Dataset/CICIDS-2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    'Dataset/CICIDS-2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    'Dataset/CICIDS-2017/Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'Dataset/CICIDS-2017/Monday-WorkingHours.pcap_ISCX.csv',
    'Dataset/CICIDS-2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    'Dataset/CICIDS-2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'Dataset/CICIDS-2017/Tuesday-WorkingHours.pcap_ISCX.csv',
    'Dataset/CICIDS-2017/Wednesday-workingHours.pcap_ISCX.csv',
]

output_path = 'Dataset/CICIDS-2017/dataset.csv'
preprocess_cicids_dataset(input_paths, output_path)
