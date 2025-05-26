import json
import redis.asyncio as redis
from typing import Any, Optional, Union
import logging
from config import settings

logger = logging.getLogger(__name__)

class CacheService:
    """Redis-based caching service for high performance"""
    
    def __init__(self):
        self.redis_pool = None
        self._connection = None
    
    async def connect(self):
        """Initialize Redis connection pool"""
        try:
            self.redis_pool = redis.ConnectionPool.from_url(
                settings.redis_url,
                max_connections=settings.redis_max_connections,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )
            self._connection = redis.Redis(connection_pool=self.redis_pool)
            
            # Test connection
            await self._connection.ping()
            logger.info("✅ Redis cache service connected")
            
        except Exception as e:
            logger.warning(f"⚠️ Redis cache service unavailable: {str(e)}")
            self._connection = None
    
    async def disconnect(self):
        """Close Redis connections"""
        if self._connection:
            await self._connection.close()
        if self.redis_pool:
            await self.redis_pool.disconnect()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self._connection:
            return None
        
        try:
            value = await self._connection.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {str(e)}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set value in cache with TTL"""
        if not self._connection:
            return False
        
        try:
            serialized_value = json.dumps(value, default=str)
            await self._connection.setex(key, ttl, serialized_value)
            return True
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self._connection:
            return False
        
        try:
            await self._connection.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error for key {key}: {str(e)}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self._connection:
            return False
        
        try:
            return await self._connection.exists(key) > 0
        except Exception as e:
            logger.warning(f"Cache exists error for key {key}: {str(e)}")
            return False
    
    async def publish(self, channel: str, message: Any) -> bool:
        """Publish message to Redis channel"""
        if not self._connection:
            return False
        
        try:
            serialized_message = json.dumps(message, default=str)
            await self._connection.publish(channel, serialized_message)
            return True
        except Exception as e:
            logger.warning(f"Cache publish error for channel {channel}: {str(e)}")
            return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern"""
        if not self._connection:
            return 0
        
        try:
            keys = await self._connection.keys(pattern)
            if keys:
                return await self._connection.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Cache invalidate pattern error for {pattern}: {str(e)}")
            return 0
