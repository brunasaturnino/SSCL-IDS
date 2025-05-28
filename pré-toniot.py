import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_network_dataset(input_paths, output_path):
    """
    Mantém apenas colunas específicas, codifica IPs/proto, normaliza dados e salva em CSV.
    """
    all_dfs = []

    # Lista das colunas desejadas
    desired_columns = [
        "src_ip", "src_port", "dst_ip", "proto",
        "dst_port", "src_bytes", "dst_bytes", "src_ip_bytes", "dst_ip_bytes", "label"
    ]

    for path in input_paths:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()

        # Verifica se todas as colunas desejadas existem
        existing_cols = [col for col in desired_columns if col in df.columns]
        missing_cols = [col for col in desired_columns if col not in df.columns]
        if missing_cols:
            print(f"⚠️ Colunas ausentes em {path}: {missing_cols}")

        df = df[existing_cols]

        # Codifica IPs e protocolo, se existirem
        for col in ['src_ip', 'dst_ip', 'proto']:
            if col in df.columns:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        # Garante que label é int
        if 'label' in df.columns:
            df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
        else:
            raise KeyError("Coluna 'label' não encontrada em " + path)

        # Separa X e y
        X = df.drop(columns=['label'])
        y = df['label']

        # Normaliza somente as colunas numéricas
        X_numeric = X.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=X_numeric.columns, index=X.index)

        # Junta com label
        final_df = pd.concat([X_scaled, y], axis=1)
        all_dfs.append(final_df)

    # Concatena e salva
    result = pd.concat(all_dfs, ignore_index=True)
    result.to_csv(output_path, index=False)
    print(f"✅ Dataset processado salvo em {output_path}")

# Exemplo de uso
input_paths = ['Dataset/TONIoT/train_test_network.csv']
output_path = 'Dataset/dataset-toniot.csv'
preprocess_network_dataset(input_paths, output_path)
