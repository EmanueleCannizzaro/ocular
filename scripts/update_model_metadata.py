#!/usr/bin/env python
"""
Backfill model metadata (license, downloads, etc.) from Hugging Face.

This script will:
  * Ensure the `models` table has the required metadata columns.
  * Fetch metadata for every model that references a Hugging Face repository.
  * Update the database with the fetched information.

Usage:
    UV_CACHE_DIR=.uv-cache uv run python scripts/update_model_metadata.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import requests
from sqlalchemy import inspect, text

from ocular.database.session import SessionLocal, engine
from ocular.database.models import Model

HF_API_URL = "https://huggingface.co/api/models/{repo}"
TIMEOUT = 15

# Columns we expect on the models table (name -> SQL DDL fragment)
REQUIRED_COLUMNS: Dict[str, str] = {
    "license": "TEXT",
    "pipeline_tag": "TEXT",
    "downloads": "INTEGER",
    "likes": "INTEGER",
    "last_modified": "TEXT"
}


def ensure_columns() -> None:
    """Ensure the new metadata columns exist on the models table."""
    inspector = inspect(engine)
    column_names = {column['name'] for column in inspector.get_columns('models')}

    with engine.begin() as conn:
        for name, ddl in REQUIRED_COLUMNS.items():
            if name not in column_names:
                conn.execute(text(f"ALTER TABLE models ADD COLUMN {name} {ddl}"))
                print(f"✓ Added column models.{name}")


def fetch_huggingface_metadata(repo_id: str) -> Optional[dict]:
    """Fetch metadata for a Hugging Face repository."""
    if not repo_id:
        return None

    url = HF_API_URL.format(repo=repo_id)
    try:
        response = requests.get(url, timeout=TIMEOUT)
        if response.status_code != 200:
            print(f"⚠️  Hugging Face returned {response.status_code} for {repo_id}", file=sys.stderr)
            return None
        return response.json()
    except requests.RequestException as exc:
        print(f"⚠️  Failed to fetch metadata for {repo_id}: {exc}", file=sys.stderr)
        return None


def is_open_source(license_name: Optional[str]) -> bool:
    if not license_name:
        return False
    normalized = license_name.lower()
    closed_keywords = {"unknown", "proprietary", "custom", "other"}
    return normalized not in closed_keywords


def update_models() -> None:
    session = SessionLocal()

    try:
        models = session.query(Model).all()
        if not models:
            print("No models found in database.")
            return

        for model in models:
            repo_id = (model.version or '').strip()
            if '/' not in repo_id:
                print(f"- Skipping {model.name}: version '{model.version}' is not a Hugging Face repo")
                continue

            print(f"🔎 Fetching metadata for {repo_id}...")
            metadata = fetch_huggingface_metadata(repo_id)
            if not metadata:
                continue

            model.license = metadata.get('license') or metadata.get('cardData', {}).get('license')
            model.pipeline_tag = metadata.get('pipeline_tag')
            model.downloads = metadata.get('downloads')
            model.likes = metadata.get('likes')

            last_modified = metadata.get('lastModified') or metadata.get('lastModifiedAt')
            if last_modified:
                try:
                    model.last_modified = datetime.fromisoformat(last_modified.replace('Z', '+00:00'))
                except ValueError:
                    model.last_modified = None

            # Update repo_url/documentation if missing
            model.repo_url = model.repo_url or f"https://huggingface.co/{repo_id}"
            model.documentation_url = model.documentation_url or metadata.get('cardData', {}).get('documentation')

            # Update extra metadata
            extra = model.extra_metadata or {}
            hf_stats = {
                'downloads': model.downloads,
                'likes': model.likes,
                'last_modified': last_modified,
                'tags': metadata.get('tags', []),
                'pipeline_tag': model.pipeline_tag,
                'license': model.license,
            }
            extra['huggingface'] = {
                'tags': metadata.get('tags', []),
                'library_name': metadata.get('library_name'),
                'pipeline_tag': metadata.get('pipeline_tag'),
                'license': model.license,
            }
            extra['huggingface_stats'] = hf_stats
            extra['is_opensource'] = is_open_source(model.license)
            model.extra_metadata = extra

            # Ensure open source flag is exposed to front-end
            model.config_template = model.config_template or {}

            print(f"  ↻ Updated {model.name}: license={model.license}, downloads={model.downloads}")

        session.commit()
        print("\n✅ Model metadata updated successfully.")
    finally:
        session.close()


def main() -> None:
    ensure_columns()
    update_models()


if __name__ == "__main__":
    main()
