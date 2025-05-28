import pandas as pd

# Caminho para o dataset original
input_path = "Dataset/dataset-unsw.csv"

# Caminho para salvar a amostra
output_path = "Dataset/sample-unsw.csv"

# Porcentagem da amostra (ex: 10%)
sample_fraction = 0.1

# Carrega o dataset
df = pd.read_csv(input_path)

# Cria a amostra
sample_df = df.sample(frac=sample_fraction, random_state=42)

# Salva a amostra
sample_df.to_csv(output_path, index=False)

print(f"Amostra salva com {len(sample_df)} linhas em '{output_path}'")
