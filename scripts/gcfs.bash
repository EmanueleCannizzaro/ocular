#!/bin/bash

# --- CONFIGURAZIONE INIZIALE ---
# [cite_start]1. Dice a Hugging Face di usare /tmp per la sua cache [cite: 1]
export HF_HOME="/tmp/huggingface_cache"

# [cite_start]2. Crea un file di configurazione Boto personalizzato in /tmp [cite: 1]
cat <<EOF > /tmp/gsutil.boto
[GSUtil]
state_dir = /tmp/.gsutil_state
parallel_composite_upload_threshold = 150M
EOF

# [cite_start]3. Dice a gsutil di usare il file di configurazione temporaneo [cite: 1]
export BOTO_CONFIG="/tmp/gsutil.boto"

echo "➡️ Ambiente configurato per usare la cartella /tmp."

# --- DEFINIZIONE DEI MODELLI E DEL BUCKET ---
GCS_BUCKET="gs://models-aecs4u"
MODELS_TO_CHECK=(
    "allenai/olmOCR-7B-0825"
    "reducto/RolmOCR"
    "NousResearch/Hermes-2-Pro-Mistral-7B"
    "NousResearch/Hermes-2-Pro-Llama-3-8B"
    "microsoft/Phi-3-mini-4k-instruct"
)

# --- ESECUZIONE ---
# [cite_start]Installa la libreria di Hugging Face una sola volta [cite: 2]
echo "➡️ Installazione/aggiornamento di huggingface-hub..."
pip install --upgrade -q huggingface-hub

# Cicla su ogni modello definito nell'array
for model_id in "${MODELS_TO_CHECK[@]}"; do
    echo "--------------------------------------------------"
    echo "🔎 Controllo del modello: ${model_id}"

    # Costruisce il percorso di destinazione su GCS
    gcs_path="${GCS_BUCKET}/${model_id}/"

    # Controlla se un file chiave del modello (es. config.json) esiste già in GCS
    # gsutil -q stat ritorna un codice di successo (0) se il file esiste
    if gsutil -q stat "${gcs_path}config.json"; then
        echo "✅ Modello già presente in GCS. Salto."
    else
        echo "📥 Modello non trovato. Avvio del download da Hugging Face..."

        # Crea una cartella temporanea sicura in /tmp
        local_dir=$(mktemp -d -p /tmp "${model_id//\//_}-XXXXXX")

        # [cite_start]Scarica il modello nella cartella temporanea [cite: 2]
        python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${model_id}', local_dir='${local_dir}', local_dir_use_symlinks=False, resume_download=True)"

        if [ $? -eq 0 ]; then
            echo "📤 Download completato. Avvio dell'upload su GCS..."

            # [cite_start]Carica il contenuto della cartella temporanea nel bucket GCS [cite: 3]
            gsutil -m cp -r "${local_dir}/"* "${gcs_path}"

            echo "🧹 Pulizia dei file temporanei per ${model_id}..."
            rm -rf "${local_dir}"
        else
            echo "❌ Errore durante il download di ${model_id}. Salto l'upload."
            # Pulisce comunque in caso di download parziale
            rm -rf "${local_dir}"
        fi
    fi
done

# --- PULIZIA FINALE ---
echo "--------------------------------------------------"
echo "🧹 Pulizia finale dei file di configurazione..."
rm -rf /tmp/gsutil.boto /tmp/.gsutil_state /tmp/huggingface_cache

echo "🎉 Operazione completata per tutti i modelli!"
