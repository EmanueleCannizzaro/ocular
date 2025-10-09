#!/bin/bash

# --- IMPOSTAZIONI E CONFIGURAZIONE ---

# Interrompe lo script immediatamente se un comando fallisce
set -e

# Numero massimo di processi da eseguire in parallelo
MAX_PARALLEL_JOBS=3

# Variabili di configurazione
GCS_BUCKET="gs://models-aecs4u"
HF_HOME="/tmp/huggingface_cache"
BOTO_CONFIG="/tmp/gsutil.boto"

# Lista dei modelli da sincronizzare
MODELS_TO_CHECK=(
    "allenai/olmOCR-7B-0825"
    "reducto/RolmOCR"
    "NousResearch/Hermes-2-Pro-Mistral-7B"
    "NousResearch/Hermes-2-Pro-Llama-3-8B"
    "microsoft/Phi-3-mini-4k-instruct"
)

# --- FUNZIONE PRINCIPALE DI ELABORAZIONE ---

process_model() {
    local model_id=$1
    local gcs_path="${GCS_BUCKET}/${model_id}/"

    printf "[%s] 🔎 Controllo del modello: %s\n" "$(date +%T)" "$model_id"

    # Controlla se il file di configurazione esiste già nel bucket GCS
    if gsutil -q stat "${gcs_path}config.json"; then
        printf "[%s] ✅ Modello '%s' già presente. Salto.\n" "$(date +%T)" "$model_id"
    else
        printf "[%s] 📥 Download di '%s'...\n" "$(date +%T)" "$model_id"

        # Crea una directory temporanea sicura
        local local_dir
        local_dir=$(mktemp -d -p /tmp "${model_id//\//_}-XXXXXX")

        # Scarica il modello da Hugging Face
        python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${model_id}', local_dir='${local_dir}')"

        printf "[%s] 📤 Upload di '%s' su GCS...\n" "$(date +%T)" "$model_id"

        # Carica i file in parallelo su Google Cloud Storage
        gsutil -m cp -r "${local_dir}/"* "${gcs_path}"

        printf "[%s] 🧹 Pulizia dei file temporanei per '%s'.\n" "$(date +%T)" "$model_id"
        rm -rf "${local_dir}"
    fi
}

# --- ESECUZIONE ---

# Configura gsutil per usare /tmp
cat <<EOF > "$BOTO_CONFIG"
[GSUtil]
state_dir = /tmp/.gsutil_state
parallel_composite_upload_threshold = 150M
EOF
export BOTO_CONFIG

# Installa/aggiorna le dipendenze
echo "📦 Aggiornamento delle dipendenze Python..."
pip install --upgrade -q huggingface-hub
pip cache purge

# Avvia il processo di sincronizzazione
echo "🚀 Avvio del processo di sincronizzazione per ${#MODELS_TO_CHECK[@]} modelli..."

for model in "${MODELS_TO_CHECK[@]}"; do
    # Avvia la funzione in background
    process_model "$model" &

    # Controlla il numero di job in esecuzione
    if (($(jobs -p | wc -l) >= MAX_PARALLEL_JOBS)); then
        # Attende che un qualsiasi job in background finisca
        wait -n
    fi
done

# Attende che tutti i restanti job in background siano completati
wait

# --- PULIZIA FINALE ---
printf "[%s] 🧹 Pulizia finale dei file di configurazione...\n" "$(date +%T)"
rm -f "$BOTO_CONFIG"
rm -rf "/tmp/.gsutil_state" "$HF_HOME"

printf "[%s] 🎉 Operazione completata per tutti i modelli!\n" "$(date +%T)"
