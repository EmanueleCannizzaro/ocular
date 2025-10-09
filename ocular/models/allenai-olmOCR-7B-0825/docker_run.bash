docker run --rm -it --gpus all \
  -e AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY" \
  -e AWS_SECRET_ACCESS_KEY="YOUR_SECRET_KEY" \
  -e S3_ENDPOINT_URL="https://s3api-us-west-1.runpod.io" \
  -e S3_BUCKET_NAME="your-bucket-name" \
  -v /path/to/your/local/document.pdf:/app/document.pdf \
  olmocr-extractor