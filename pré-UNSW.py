import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_unsw_minimal(input_paths, output_path):
    all_dfs = []

    for path in input_paths:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        all_dfs.append(df)

    df = pd.concat(all_dfs, ignore_index=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Mapeamento das colunas reais para os nomes desejados
    column_mapping = {
        "src_ip": "src_ip",
        "src_port": "src_port",
        "dst_ip": "dst_ip",
        "dst_port": "dst_port",
        "proto": "protocol",
        "src_bytes": "totlen_fwd_pkts",
        "dst_bytes": "totlen_bwd_pkts",
        "src_ip_bytes": "fwd_pkt_len_max",
        "dst_ip_bytes": "bwd_pkt_len_max",
        "label": "label" 
    }

    # Verifica quais colunas existem
    existing_mapping = {k: v for k, v in column_mapping.items() if v in df.columns}
    missing = [v for v in column_mapping.values() if v not in df.columns]
    if missing:
        print(f"⚠️ As seguintes colunas estão ausentes e serão ignoradas: {missing}")

    # Filtra e renomeia
    df = df[list(existing_mapping.values())]
    df.rename(columns={v: k for k, v in existing_mapping.items()}, inplace=True)

    # Codifica IPs e protocolo (categóricos)
    for col in ['src_ip', 'dst_ip', 'proto']:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))


# Separa label para não normalizar
    y = df['label']
    X = df.drop(columns=['label'])

    # Normaliza apenas os dados de entrada
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Junta X normalizado com label original
    final_df = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)

    # Salva
    final_df.to_csv(output_path, index=False)
    print(f"✅ Dataset processado salvo em: {output_path}")


# Exemplo de uso:
input_paths = ["Dataset/UNSW/UNSW-NB15-all.csv"]
output_path = "Dataset/dataset-unsw.csv"
preprocess_unsw_minimal(input_paths, output_path)
