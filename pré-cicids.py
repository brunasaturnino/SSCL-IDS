import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_cicids_dataset(input_paths, output_path):
    all_dfs = []

    for path in input_paths:
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip().str.lower()
        all_dfs.append(df)

    df = pd.concat(all_dfs, ignore_index=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Mapeamento: nome desejado -> nome real no CSV
    column_mapping = {
        "src_ip": "source ip",
        "src_port": "source port",
        "dst_ip": "destination ip",
        "dst_port": "destination port",
        "proto": "protocol",
        "src_bytes": "subflow fwd bytes",
        "dst_bytes": "subflow bwd bytes",
        "src_ip_bytes": "init_win_bytes_forward",
        "dst_ip_bytes": "init_win_bytes_backward",
        "label": "label"
    }

    # Verifica quais colunas realmente existem
    existing_mapping = {k: v for k, v in column_mapping.items() if v in df.columns}
    missing = [v for v in column_mapping.values() if v not in df.columns]
    if missing:
        print(f"⚠️ As seguintes colunas estão ausentes e serão ignoradas: {missing}")

    df = df[[existing_mapping[k] for k in existing_mapping]]
    df.rename(columns={v: k for k, v in existing_mapping.items()}, inplace=True)

    # Codifica IPs e protocolo
    for col in ['src_ip', 'dst_ip', 'proto']:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Converte label para 0 (BENIGN) ou 1 (ataque)
    if 'label' in df.columns:
        df['label'] = df['label'].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)

    # Separa X e y
    if 'label' in df.columns:
        X = df.drop(columns=['label'])
        y = df['label']
    else:
        X = df
        y = None

    # Normaliza dados numéricos
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Junta com a label se existir
    final_df = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1) if y is not None else X_scaled
    final_df.to_csv(output_path, index=False)
    print(f"✅ Dataset processado e salvo em: {output_path}")

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

output_path = 'Dataset/dataset-cicids-minimal.csv'
preprocess_cicids_dataset(input_paths, output_path)
