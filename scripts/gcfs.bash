
# --- CONFIGURAZIONE INIZIALE ---
# 1. Dice a Hugging Face di usare /tmp per la sua cache
export HF_HOME="/tmp/huggingface_cache"

# 1. Crea un file di configurazione Boto personalizzato in /tmp
cat <<EOF > /tmp/gsutil.boto
[GSUtil]
state_dir = /tmp/.gsutil_state
parallel_composite_upload_threshold = 150M
EOF

# 2. Dice a gsutil di usare il file di configurazione temporaneo
export BOTO_CONFIG="/tmp/gsutil.boto"

echo "gsutil è stato configurato per usare la cartella /tmp."


# 1. Installa la libreria di Hugging Face e scarica il modello in una cartella temporanea
pip install --upgrade huggingface-hub

python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='reducto/RolmOCR', local_dir='/tmp/temp_RolmOCR', local_dir_use_symlinks=False)"

# 2. Carica il contenuto della cartella temporanea nel tuo bucket GCS
# L'opzione -m accelera l'upload eseguendolo in parallelo.
gsutil -m cp -r /tmp/temp_RolmOCR/* gs://models-aecs4u/reducto/RolmOCR/

# 3. Pulisce la cartella temporanea per liberare spazio
rm -rf /tmp/temp_RolmOCR /tmp/.gsutil_state /tmp/huggingface_cache


# 2. Scarica il modello con un piccolo script Python
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='allenai/olmOCR-7B-0825', local_dir='/tmp/temp_allenai', local_dir_use_symlinks=False)"

# 3. Carica su GCS
gsutil -m cp -r /tmp/temp_allenai/* gs://models-aecs4u/allenai/olmOCR-7B-0825/

# 4. Pulisci
rm -rf /tmp/temp_allenai /tmp/.gsutil_state /tmp/huggingface_cache


echo "✅ Operazione completata con successo!"
