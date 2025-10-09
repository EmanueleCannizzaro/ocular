gcloud run deploy mio-servizio-vllm \
    --image gcr.io/tuo-progetto/immagine-vllm \
    --gpu=nvidia-l4 \
    --region=europe-west1 \
    --volume name=modelli-gcs,gcs-fuse,bucket=il-mio-bucket-dei-modelli \
    --volume-mount modelli-gcs=/models \
    --allow-unauthenticated
