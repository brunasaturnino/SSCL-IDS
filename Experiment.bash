#!/bin/bash

# Configurações
DATASET_PATH="Dataset/CICIDS-2017/sample.csv"
DATASET="CICIDS-2017/sample.csv"
BATCH=2046
EPOCHS=200
CHECKPOINT_NAME="scarf1_embdd_dim=45_lr=0.001_bs=${BATCH}_epochs=${EPOCHS}_tempr=0.5_V=onlyunsw_cr_rt=0.4_ach_cr_rt0.2_msk_rt0_ach_msk_rt0.pth"
CHECKPOINT_PATH="new_checkpoints/$CHECKPOINT_NAME"
SUPERVISED_MODEL="RF"
LOG_FILE="resultados_execucao.txt"

# Limpa log anterior
rm -f "$LOG_FILE"

# 🟠 Treinamento
echo "========== INÍCIO DO TREINAMENTO ==========" | tee -a "$LOG_FILE"
python3 train.py --dataset_path "$DATASET_PATH" --batch_size "$BATCH" --epochs "$EPOCHS" 2>&1 | tee -a "$LOG_FILE"

# 🟡 Avaliação OOD (AUROC)
echo -e "\n========== INÍCIO DA AVALIAÇÃO OOD ==========" | tee -a "$LOG_FILE"
python3 evaluation.py --test_dataset_name "$DATASET" --batch_size "$BATCH" --model_chkpt "$CHECKPOINT_NAME" --train_from_scratch False 2>&1 | tee -a "$LOG_FILE"

# 🔵 Avaliação supervisionada
echo -e "\n========== INÍCIO DA AVALIAÇÃO SUPERVISIONADA ==========" | tee -a "$LOG_FILE"
python3 evaluation_supervised.py --supervised_training_dataset "$DATASET_PATH" --batch_size "$BATCH" --model_chkpt "$CHECKPOINT_PATH" --supervised_model "$SUPERVISED_MODEL" 2>&1 | tee -a "$LOG_FILE"

# ✅ Fim
echo -e "\n✅ Todos os experimentos foram concluídos com sucesso!" | tee -a "$LOG_FILE"
