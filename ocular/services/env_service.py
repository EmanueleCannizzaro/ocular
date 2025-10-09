"""
Service for managing .env file updates from the settings interface.
"""

import os
import re
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime


class EnvService:
    """Service for reading and writing .env file."""

    def __init__(self, env_path: Optional[str] = None):
        """
        Initialize EnvService.

        Args:
            env_path: Path to .env file. If None, uses project root .env
        """
        if env_path:
            self.env_path = Path(env_path)
        else:
            # Default to project root .env
            project_root = Path(__file__).parent.parent.parent
            self.env_path = project_root / ".env"

    def read_env(self) -> Dict[str, str]:
        """
        Read all variables from .env file.

        Returns:
            Dictionary of environment variables
        """
        env_vars = {}

        if not self.env_path.exists():
            return env_vars

        with open(self.env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue

                # Parse KEY=VALUE format
                match = re.match(r'^([A-Z_][A-Z0-9_]*)\s*=\s*(.*)$', line)
                if match:
                    key, value = match.groups()
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    env_vars[key] = value

        return env_vars

    def write_env(self, updates: Dict[str, str], backup: bool = True) -> bool:
        """
        Update .env file with new values.

        Args:
            updates: Dictionary of KEY=VALUE pairs to update
            backup: Whether to create backup before writing

        Returns:
            True if successful, False otherwise
        """
        try:
            # Read existing .env file
            existing_content = []
            existing_keys = set()

            if self.env_path.exists():
                # Create backup if requested
                if backup:
                    self._create_backup()

                with open(self.env_path, 'r', encoding='utf-8') as f:
                    existing_content = f.readlines()

                # Track which keys already exist
                for line in existing_content:
                    match = re.match(r'^([A-Z_][A-Z0-9_]*)\s*=', line.strip())
                    if match:
                        existing_keys.add(match.group(1))

            # Update existing lines and track which updates were applied
            updated_content = []
            applied_updates = set()

            for line in existing_content:
                stripped = line.strip()

                # Preserve comments and empty lines
                if not stripped or stripped.startswith('#'):
                    updated_content.append(line)
                    continue

                # Check if this line contains a key we want to update
                match = re.match(r'^([A-Z_][A-Z0-9_]*)\s*=', stripped)
                if match:
                    key = match.group(1)
                    if key in updates:
                        # Update this line
                        value = updates[key]
                        # Quote value if it contains spaces
                        if ' ' in value or not value:
                            value = f'"{value}"'
                        updated_content.append(f'{key}={value}\n')
                        applied_updates.add(key)
                    else:
                        # Keep existing line
                        updated_content.append(line)
                else:
                    # Keep malformed lines
                    updated_content.append(line)

            # Add new keys that weren't in the file
            new_keys = set(updates.keys()) - applied_updates
            if new_keys:
                # Add section header if adding new keys
                if updated_content and not updated_content[-1].endswith('\n\n'):
                    updated_content.append('\n')
                updated_content.append(f'# Updated via settings UI - {datetime.now().isoformat()}\n')

                for key in sorted(new_keys):
                    value = updates[key]
                    # Quote value if it contains spaces
                    if ' ' in value or not value:
                        value = f'"{value}"'
                    updated_content.append(f'{key}={value}\n')

            # Write updated content
            with open(self.env_path, 'w', encoding='utf-8') as f:
                f.writelines(updated_content)

            return True

        except Exception as e:
            print(f"Error writing .env file: {e}")
            return False

    def update_env_var(self, key: str, value: str, backup: bool = True) -> bool:
        """
        Update a single environment variable.

        Args:
            key: Environment variable name (must be uppercase with underscores)
            value: New value for the variable
            backup: Whether to create backup before writing

        Returns:
            True if successful, False otherwise
        """
        # Validate key format
        if not re.match(r'^[A-Z_][A-Z0-9_]*$', key):
            raise ValueError(f"Invalid environment variable name: {key}")

        return self.write_env({key: value}, backup=backup)

    def delete_env_var(self, key: str, backup: bool = True) -> bool:
        """
        Remove an environment variable from .env file.

        Args:
            key: Environment variable name to remove
            backup: Whether to create backup before writing

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.env_path.exists():
                return False

            # Create backup if requested
            if backup:
                self._create_backup()

            # Read and filter content
            with open(self.env_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Remove lines with the specified key
            filtered_lines = []
            for line in lines:
                stripped = line.strip()
                match = re.match(r'^([A-Z_][A-Z0-9_]*)\s*=', stripped)
                if match and match.group(1) == key:
                    # Skip this line (delete it)
                    continue
                filtered_lines.append(line)

            # Write filtered content
            with open(self.env_path, 'w', encoding='utf-8') as f:
                f.writelines(filtered_lines)

            return True

        except Exception as e:
            print(f"Error deleting env variable: {e}")
            return False

    def _create_backup(self) -> None:
        """Create timestamped backup of .env file."""
        if not self.env_path.exists():
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.env_path.parent / f'.env.backup.{timestamp}'

        try:
            with open(self.env_path, 'r', encoding='utf-8') as src:
                with open(backup_path, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
            print(f"Created backup: {backup_path}")
        except Exception as e:
            print(f"Warning: Failed to create backup: {e}")

    def get_backup_files(self) -> List[Path]:
        """
        Get list of .env backup files.

        Returns:
            List of backup file paths, sorted by timestamp (newest first)
        """
        if not self.env_path.parent.exists():
            return []

        backups = list(self.env_path.parent.glob('.env.backup.*'))
        return sorted(backups, reverse=True)

    def restore_backup(self, backup_path: Path) -> bool:
        """
        Restore .env from a backup file.

        Args:
            backup_path: Path to backup file

        Returns:
            True if successful, False otherwise
        """
        try:
            if not backup_path.exists():
                return False

            # Create backup of current .env before restoring
            self._create_backup()

            # Copy backup to .env
            with open(backup_path, 'r', encoding='utf-8') as src:
                with open(self.env_path, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())

            return True

        except Exception as e:
            print(f"Error restoring backup: {e}")
            return False

    def validate_env_value(self, key: str, value: str) -> tuple[bool, Optional[str]]:
        """
        Validate environment variable value.

        Args:
            key: Environment variable name
            value: Value to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # URL validation for specific keys
        url_keys = [
            'DATABASE_URL', 'REDIS_URL', 'RUNPOD_BUCKET_ENDPOINT_URL',
            'VASTAI_ENDPOINT_URL', 'DEEPINFRA_BASE_URL',
            'AZURE_DOC_INTEL_ENDPOINT'
        ]

        if key in url_keys:
            if value and not re.match(r'^[a-z]+://', value):
                return False, f"{key} must be a valid URL (e.g., https://...)"

        # Email validation for email keys
        if 'EMAIL' in key:
            if value and not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value):
                return False, f"{key} must be a valid email address"

        # Port number validation
        if 'PORT' in key:
            try:
                port = int(value)
                if not (1 <= port <= 65535):
                    return False, f"{key} must be between 1 and 65535"
            except ValueError:
                return False, f"{key} must be a valid port number"

        # Environment validation
        if key == 'ENVIRONMENT':
            valid_envs = ['development', 'staging', 'production', 'testing']
            if value.lower() not in valid_envs:
                return False, f"ENVIRONMENT must be one of: {', '.join(valid_envs)}"

        # Log level validation
        if key == 'LOG_LEVEL':
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if value.upper() not in valid_levels:
                return False, f"LOG_LEVEL must be one of: {', '.join(valid_levels)}"

        return True, None


# Singleton instance
_env_service = None


def get_env_service(env_path: Optional[str] = None) -> EnvService:
    """Get or create EnvService singleton instance."""
    global _env_service
    if _env_service is None:
        _env_service = EnvService(env_path)
    return _env_service
