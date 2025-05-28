import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_cicdos_dataset(input_paths, output_path):
    all_dfs = []

    for path in input_paths:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        all_dfs.append(df)

    df = pd.concat(all_dfs, ignore_index=True)

    # Substitui valores inválidos
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Converte rótulo: BENIGN -> 0, outros -> 1
    df['Label'] = df['Label'].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)

    # Colunas mapeadas
    desired_columns = {
        #"src_ip": "Source IP",
        #"src_port": "Source Port",
        #"dst_ip": "Destination IP",
        "dst_port": "Destination Port",
        #"proto": "Protocol",
        "src_bytes": "Subflow Fwd Bytes",
        "dst_bytes": "Subflow Bwd Bytes",
        "src_ip_bytes": "Init_Win_bytes_forward",
        "dst_ip_bytes": "Init_Win_bytes_backward",
        "Label": "Label"
    }

    # Verifica e filtra colunas existentes
    existing_cols = {k: v for k, v in desired_columns.items() if v in df.columns}
    missing_cols = [v for v in desired_columns.values() if v not in df.columns]
    if missing_cols:
        print(f"⚠️ As seguintes colunas estão ausentes: {missing_cols}")

    df = df[list(existing_cols.values())]

    # Renomeia para nomes padrão
    df.rename(columns={v: k for k, v in existing_cols.items()}, inplace=True)

    # Codifica IPs e protocolo (categóricos)
    #for col in ['src_ip', 'dst_ip', 'proto']:
     #   if col in df.columns:
      #      df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Separa X e y
    X = df.drop(columns=['Label'])
    y = df['Label']

    # Normaliza colunas numéricas
    X_numeric = X.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=X_numeric.columns)

    # Junta X escalado com y
    final_df = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
    final_df.to_csv(output_path, index=False)
    print(f"✅ Dataset processado salvo em: {output_path}")

# Exemplo de uso:
input_paths = [
    'Dataset/CICDOS/LDAP.csv',
    'Dataset/CICDOS/Portmap.csv',
    'Dataset/CICDOS/UDPLag.csv',
]

output_path = 'Dataset/dataset-cicdos.csv'
preprocess_cicdos_dataset(input_paths, output_path)
