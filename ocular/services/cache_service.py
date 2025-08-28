"""
Caching service for Ocular OCR system.
"""

import asyncio
import json
import time
import hashlib
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from ..core.interfaces import CacheProvider
from ..core.models import OCRResult
from ..providers.settings import OcularSettings

logger = logging.getLogger(__name__)


class InMemoryCache:
    """Simple in-memory cache implementation."""
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached value."""
        async with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                return None
            
            # Check TTL
            if entry["expires_at"] and time.time() > entry["expires_at"]:
                del self._cache[key]
                return None
            
            return entry["data"]
    
    async def set(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Set cached value."""
        async with self._lock:
            expires_at = time.time() + ttl if ttl else None
            self._cache[key] = {
                "data": data,
                "created_at": time.time(),
                "expires_at": expires_at
            }
    
    async def delete(self, key: str) -> None:
        """Delete cached value."""
        async with self._lock:
            self._cache.pop(key, None)
    
    async def clear(self) -> None:
        """Clear all cached values."""
        async with self._lock:
            self._cache.clear()
    
    async def cleanup_expired(self) -> int:
        """Clean up expired entries and return count."""
        expired_keys = []
        current_time = time.time()
        
        async with self._lock:
            for key, entry in self._cache.items():
                if entry["expires_at"] and current_time > entry["expires_at"]:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
        
        return len(expired_keys)
    
    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total_entries = len(self._cache)
            expired_count = 0
            current_time = time.time()
            
            for entry in self._cache.values():
                if entry["expires_at"] and current_time > entry["expires_at"]:
                    expired_count += 1
            
            return {
                "total_entries": total_entries,
                "active_entries": total_entries - expired_count,
                "expired_entries": expired_count
            }


class FileCache:
    """File-based cache implementation."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        # Use hash to avoid filesystem issues with long keys
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached value from file."""
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        try:
            async with self._lock:
                with open(cache_path, 'r') as f:
                    entry = json.load(f)
                
                # Check TTL
                if entry["expires_at"] and time.time() > entry["expires_at"]:
                    cache_path.unlink(missing_ok=True)
                    return None
                
                return entry["data"]
        except Exception as e:
            logger.warning(f"Error reading cache file {cache_path}: {e}")
            cache_path.unlink(missing_ok=True)
            return None
    
    async def set(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Set cached value to file."""
        cache_path = self._get_cache_path(key)
        
        expires_at = time.time() + ttl if ttl else None
        entry = {
            "data": data,
            "created_at": time.time(),
            "expires_at": expires_at
        }
        
        try:
            async with self._lock:
                with open(cache_path, 'w') as f:
                    json.dump(entry, f)
        except Exception as e:
            logger.error(f"Error writing cache file {cache_path}: {e}")
    
    async def delete(self, key: str) -> None:
        """Delete cached value file."""
        cache_path = self._get_cache_path(key)
        cache_path.unlink(missing_ok=True)
    
    async def clear(self) -> None:
        """Clear all cached files."""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Error clearing cache directory: {e}")
    
    async def cleanup_expired(self) -> int:
        """Clean up expired cache files."""
        expired_count = 0
        current_time = time.time()
        
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r') as f:
                        entry = json.load(f)
                    
                    if entry.get("expires_at") and current_time > entry["expires_at"]:
                        cache_file.unlink()
                        expired_count += 1
                        
                except Exception:
                    # Remove corrupted cache files
                    cache_file.unlink(missing_ok=True)
                    expired_count += 1
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
        
        return expired_count
    
    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_files = 0
        total_size = 0
        expired_count = 0
        current_time = time.time()
        
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                total_files += 1
                total_size += cache_file.stat().st_size
                
                try:
                    with open(cache_file, 'r') as f:
                        entry = json.load(f)
                    
                    if entry.get("expires_at") and current_time > entry["expires_at"]:
                        expired_count += 1
                except Exception:
                    expired_count += 1
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
        
        return {
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "active_files": total_files - expired_count,
            "expired_files": expired_count
        }


class CacheService(CacheProvider):
    """Main caching service for OCR results."""
    
    def __init__(self, settings: OcularSettings):
        self.settings = settings
        self.default_ttl = settings.processing.cache_ttl_seconds
        
        # Choose cache implementation
        if hasattr(settings.files, 'cache_dir') and settings.files.cache_dir:
            self._cache = FileCache(settings.files.cache_dir)
            logger.info("Using file-based cache")
        else:
            self._cache = InMemoryCache()
            logger.info("Using in-memory cache")
        
        # Start background cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background task for cleaning up expired entries."""
        async def cleanup_worker():
            while True:
                try:
                    await asyncio.sleep(3600)  # Run every hour
                    expired_count = await self._cache.cleanup_expired()
                    if expired_count > 0:
                        logger.info(f"Cleaned up {expired_count} expired cache entries")
                except Exception as e:
                    logger.error(f"Error in cache cleanup task: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_worker())
    
    async def get_cached_result(self, key: str) -> Optional[OCRResult]:
        """Get cached OCR result."""
        try:
            cached_data = await self._cache.get(key)
            
            if cached_data is None:
                return None
            
            # Reconstruct OCRResult from cached data
            return OCRResult(**cached_data)
            
        except Exception as e:
            logger.warning(f"Error retrieving cached result: {e}")
            return None
    
    async def cache_result(
        self, 
        key: str, 
        result: OCRResult, 
        ttl: Optional[int] = None
    ) -> None:
        """Cache OCR result."""
        try:
            # Convert OCRResult to dict for caching
            result_data = result.dict()
            
            cache_ttl = ttl or self.default_ttl
            await self._cache.set(key, result_data, cache_ttl)
            
            logger.debug(f"Cached result with key: {key[:16]}...")
            
        except Exception as e:
            logger.error(f"Error caching result: {e}")
    
    async def invalidate_cache(self, key: str) -> None:
        """Invalidate cached result."""
        try:
            await self._cache.delete(key)
            logger.debug(f"Invalidated cache key: {key[:16]}...")
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
    
    async def clear_cache(self) -> None:
        """Clear all cached results."""
        try:
            await self._cache.clear()
            logger.info("Cleared all cached results")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def generate_cache_key(self, **kwargs) -> str:
        """Generate cache key from parameters."""
        # Create deterministic key from parameters
        key_parts = []
        
        for k, v in sorted(kwargs.items()):
            if isinstance(v, Path):
                # Include file path and modification time
                key_parts.append(f"{k}:{v}")
                try:
                    key_parts.append(f"mtime:{v.stat().st_mtime}")
                except:
                    pass
            elif isinstance(v, list):
                key_parts.append(f"{k}:{sorted(str(x) for x in v)}")
            else:
                key_parts.append(f"{k}:{v}")
        
        key_string = "|".join(key_parts)
        
        # Hash to create manageable key length
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            stats = await self._cache.stats()
            stats["cache_type"] = type(self._cache).__name__
            stats["default_ttl"] = self.default_ttl
            return stats
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}
    
    async def cleanup_expired_entries(self) -> int:
        """Manually trigger cleanup of expired entries."""
        try:
            return await self._cache.cleanup_expired()
        except Exception as e:
            logger.error(f"Error during manual cleanup: {e}")
            return 0
    
    async def shutdown(self) -> None:
        """Shutdown cache service."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Final cleanup
        await self.cleanup_expired_entries()
        logger.info("Cache service shutdown completed")