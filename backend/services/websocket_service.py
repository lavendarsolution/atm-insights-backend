import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Set

from fastapi import WebSocket, WebSocketDisconnect
from services.cache_service import CacheService

logger = logging.getLogger(__name__)


class ConnectionManager:
    """WebSocket connection manager for status changes only"""

    def __init__(self, cache_service: CacheService):
        # Store connections by type
        self.dashboard_connections: Set[WebSocket] = set()
        self.atm_detail_connections: Dict[str, Set[WebSocket]] = (
            {}
        )  # atm_id -> connections
        self.alerts_connections: Set[WebSocket] = set()  # alerts page connections
        self.cache_service = cache_service
        self.redis_subscriber = None
        self.subscriber_task = None

    async def start_redis_subscriber(self):
        """Start Redis subscriber for real-time updates"""
        try:
            # Create Redis connection for subscribing
            import redis.asyncio as redis
            from config import settings

            self.redis_subscriber = redis.Redis.from_url(settings.redis_url)

            # Subscribe to channels (removed dashboard_updates)
            pubsub = self.redis_subscriber.pubsub()
            await pubsub.subscribe(
                "telemetry_updates",  # For ATM detail pages
                "atm_status_changes",  # For status changes only
                "alerts_updates",  # For alerts page real-time updates
            )

            # Start background task to handle messages
            self.subscriber_task = asyncio.create_task(
                self._handle_redis_messages(pubsub)
            )

            logger.info(
                "✅ Redis subscriber started for WebSocket updates (status changes only)"
            )

        except Exception as e:
            logger.error(f"Failed to start Redis subscriber: {str(e)}")

    async def stop_redis_subscriber(self):
        """Stop Redis subscriber"""
        if self.subscriber_task:
            self.subscriber_task.cancel()
        if self.redis_subscriber:
            await self.redis_subscriber.close()

    async def _handle_redis_messages(self, pubsub):
        """Handle messages from Redis pub/sub"""
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    channel = message["channel"].decode()
                    try:
                        data = json.loads(message["data"].decode())
                        await self._route_message(channel, data)
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Invalid JSON in Redis message: {message['data']}"
                        )

        except asyncio.CancelledError:
            logger.info("Redis subscriber task cancelled")
        except Exception as e:
            logger.error(f"Error in Redis message handler: {str(e)}")

    async def _route_message(self, channel: str, data: dict):
        """Route messages to appropriate WebSocket connections"""
        try:
            if channel == "atm_status_changes":
                await self._broadcast_atm_status_change(data)
            elif channel == "telemetry_updates":
                await self._broadcast_telemetry_update(data)
            elif channel == "alerts_updates":
                await self._broadcast_alerts_update(data)

        except Exception as e:
            logger.error(f"Error routing message from channel {channel}: {str(e)}")

    # Dashboard connections (for status changes only)
    async def connect_dashboard(self, websocket: WebSocket):
        """Connect dashboard WebSocket"""
        await websocket.accept()
        self.dashboard_connections.add(websocket)
        logger.info(
            f"Dashboard connected. Total connections: {len(self.dashboard_connections)}"
        )

        # Send connection confirmation only
        try:
            await websocket.send_json(
                {
                    "type": "connection_established",
                    "message": "Connected to status change updates",
                    "timestamp": datetime.now().isoformat(),
                }
            )
        except Exception as e:
            logger.error(f"Error sending connection confirmation: {str(e)}")

    def disconnect_dashboard(self, websocket: WebSocket):
        """Disconnect dashboard WebSocket"""
        self.dashboard_connections.discard(websocket)
        logger.info(
            f"Dashboard disconnected. Remaining connections: {len(self.dashboard_connections)}"
        )

    async def _broadcast_to_dashboard(self, data: dict):
        """Broadcast status changes to all dashboard connections"""
        if not self.dashboard_connections:
            return

        message = {
            "type": "atm_status_change",
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }

        disconnected = set()
        for connection in self.dashboard_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.add(connection)

        # Remove disconnected connections
        for conn in disconnected:
            self.dashboard_connections.discard(conn)

    # ATM Detail connections
    async def connect_atm_detail(self, websocket: WebSocket, atm_id: str):
        """Connect ATM detail WebSocket"""
        await websocket.accept()

        if atm_id not in self.atm_detail_connections:
            self.atm_detail_connections[atm_id] = set()

        self.atm_detail_connections[atm_id].add(websocket)
        logger.info(
            f"ATM detail connected for {atm_id}. Total connections: {len(self.atm_detail_connections[atm_id])}"
        )

        # Send initial ATM data
        try:
            initial_data = await self._get_atm_initial_data(atm_id)
            await websocket.send_json(
                {
                    "type": "atm_initial",
                    "atm_id": atm_id,
                    "data": initial_data,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        except Exception as e:
            logger.error(f"Error sending initial ATM data for {atm_id}: {str(e)}")

    def disconnect_atm_detail(self, websocket: WebSocket, atm_id: str):
        """Disconnect ATM detail WebSocket"""
        if atm_id in self.atm_detail_connections:
            self.atm_detail_connections[atm_id].discard(websocket)
            if not self.atm_detail_connections[atm_id]:
                del self.atm_detail_connections[atm_id]

        logger.info(f"ATM detail disconnected for {atm_id}")

    # Alerts connections
    async def connect_alerts(self, websocket: WebSocket):
        """Connect alerts WebSocket"""
        await websocket.accept()
        self.alerts_connections.add(websocket)
        logger.info(
            f"Alerts connected. Total connections: {len(self.alerts_connections)}"
        )

        # Send connection confirmation
        try:
            await websocket.send_json(
                {
                    "type": "alerts_initial",
                    "message": "Connected to alerts stream",
                    "timestamp": datetime.now().isoformat(),
                }
            )
        except Exception as e:
            logger.error(f"Error sending alerts connection confirmation: {str(e)}")

    def disconnect_alerts(self, websocket: WebSocket):
        """Disconnect alerts WebSocket"""
        self.alerts_connections.discard(websocket)
        logger.info(
            f"Alerts disconnected. Remaining connections: {len(self.alerts_connections)}"
        )

    async def _broadcast_alerts_update(self, data: dict):
        """Broadcast alerts update to all alerts connections"""
        if not self.alerts_connections:
            return

        message = {
            "type": data.get("type", "alert_update"),
            "data": data.get("data"),
            "timestamp": datetime.now().isoformat(),
        }

        disconnected = set()
        for connection in self.alerts_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.add(connection)

        # Remove disconnected connections
        for conn in disconnected:
            self.alerts_connections.discard(conn)

    async def _broadcast_telemetry_update(self, data: dict):
        """Broadcast telemetry update to ATM detail connections only"""
        atm_id = data.get("atm_id")
        if not atm_id:
            return

        # Broadcast to specific ATM detail page only
        if atm_id in self.atm_detail_connections:
            message = {
                "type": "telemetry_update",
                "atm_id": atm_id,
                "data": data,
                "timestamp": datetime.now().isoformat(),
            }

            disconnected = set()
            for connection in self.atm_detail_connections[atm_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.add(connection)

            # Remove disconnected connections
            for conn in disconnected:
                self.atm_detail_connections[atm_id].discard(conn)

    async def _broadcast_atm_status_change(self, data: dict):
        """Broadcast ATM status change to dashboard connections only"""
        atm_id = data.get("atm_id")
        old_status = data.get("old_status")
        new_status = data.get("new_status")

        # Broadcast to dashboard for status changes
        await self._broadcast_to_dashboard(
            {
                "type": "atm_status_change",
                "atm_id": atm_id,
                "old_status": old_status,
                "new_status": new_status,
                "timestamp": data.get("timestamp"),
            }
        )

        # Also broadcast to specific ATM detail page if connected
        if atm_id in self.atm_detail_connections:
            message = {
                "type": "atm_status_change",
                "atm_id": atm_id,
                "data": data,
                "timestamp": datetime.now().isoformat(),
            }

            disconnected = set()
            for connection in self.atm_detail_connections[atm_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.add(connection)

            # Remove disconnected connections
            for conn in disconnected:
                self.atm_detail_connections[atm_id].discard(conn)

    async def _get_atm_initial_data(self, atm_id: str) -> dict:
        """Get initial ATM data including recent telemetry"""
        try:
            # Get latest telemetry
            latest_telemetry = await self.cache_service.get(
                f"latest_telemetry:{atm_id}"
            )

            # Get recent telemetry history (last 100 records)
            telemetry_history = await self.cache_service.get(
                f"telemetry_history:{atm_id}"
            )

            return {
                "latest_telemetry": latest_telemetry,
                "telemetry_history": telemetry_history or [],
                "last_updated": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting ATM initial data for {atm_id}: {str(e)}")
            return {}

    async def broadcast_custom_message(
        self, message_type: str, data: dict, target: str = "dashboard"
    ):
        """Broadcast custom message to specific target"""
        message = {
            "type": message_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }

        if target == "dashboard":
            await self._broadcast_to_dashboard(data)
        elif target.startswith("atm:"):
            atm_id = target.split(":", 1)[1]
            if atm_id in self.atm_detail_connections:
                disconnected = set()
                for connection in self.atm_detail_connections[atm_id]:
                    try:
                        await connection.send_json(message)
                    except Exception:
                        disconnected.add(connection)

                for conn in disconnected:
                    self.atm_detail_connections[atm_id].discard(conn)


# Global connection manager instance
connection_manager: ConnectionManager = None


def get_connection_manager() -> ConnectionManager:
    """Get global connection manager instance"""
    return connection_manager


def set_connection_manager(manager: ConnectionManager):
    """Set global connection manager instance"""
    global connection_manager
    connection_manager = manager
