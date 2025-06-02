import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from services.websocket_service import get_connection_manager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    """WebSocket endpoint for dashboard real-time updates"""
    connection_manager = get_connection_manager()
    if not connection_manager:
        await websocket.close(code=1011, reason="WebSocket service not available")
        return

    try:
        await connection_manager.connect_dashboard(websocket)

        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for any client messages (ping/pong, etc.)
                data = await websocket.receive_text()
                logger.debug(f"Dashboard WebSocket received: {data}")

                # Handle client messages if needed
                if data == "ping":
                    await websocket.send_text("pong")

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Dashboard WebSocket error: {str(e)}")
                break

    except WebSocketDisconnect:
        logger.info("Dashboard WebSocket disconnected")
    except Exception as e:
        logger.error(f"Dashboard WebSocket connection error: {str(e)}")
    finally:
        connection_manager.disconnect_dashboard(websocket)


@router.websocket("/ws/atm/{atm_id}")
async def atm_detail_websocket(websocket: WebSocket, atm_id: str):
    """WebSocket endpoint for ATM detail page real-time updates"""
    connection_manager = get_connection_manager()
    if not connection_manager:
        await websocket.close(code=1011, reason="WebSocket service not available")
        return

    try:
        await connection_manager.connect_atm_detail(websocket, atm_id)
        logger.info(f"ATM detail WebSocket connected for {atm_id}")

        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for any client messages
                data = await websocket.receive_text()
                logger.debug(f"ATM detail WebSocket received for {atm_id}: {data}")

                # Handle client messages if needed
                if data == "ping":
                    await websocket.send_text("pong")
                elif data == "request_history":
                    # Client requesting fresh telemetry history
                    await websocket.send_json(
                        {
                            "type": "history_requested",
                            "atm_id": atm_id,
                            "message": "History refresh requested",
                        }
                    )

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"ATM detail WebSocket error for {atm_id}: {str(e)}")
                break

    except WebSocketDisconnect:
        logger.info(f"ATM detail WebSocket disconnected for {atm_id}")
    except Exception as e:
        logger.error(f"ATM detail WebSocket connection error for {atm_id}: {str(e)}")
    finally:
        connection_manager.disconnect_atm_detail(websocket, atm_id)


@router.websocket("/ws/alerts")
async def alerts_websocket(websocket: WebSocket):
    """WebSocket endpoint for alerts page real-time updates"""
    connection_manager = get_connection_manager()
    if not connection_manager:
        await websocket.close(code=1011, reason="WebSocket service not available")
        return

    try:
        await connection_manager.connect_alerts(websocket)

        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for any client messages (ping/pong, etc.)
                data = await websocket.receive_text()
                logger.debug(f"Alerts WebSocket received: {data}")

                # Handle client messages if needed
                if data == "ping":
                    await websocket.send_text("pong")

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Alerts WebSocket error: {str(e)}")
                break

    except WebSocketDisconnect:
        logger.info("Alerts WebSocket disconnected")
    except Exception as e:
        logger.error(f"Alerts WebSocket connection error: {str(e)}")
    finally:
        connection_manager.disconnect_alerts(websocket)


@router.get("/ws/status")
async def websocket_status():
    """Get WebSocket service status"""
    connection_manager = get_connection_manager()
    if not connection_manager:
        return {"status": "unavailable", "connections": 0}

    dashboard_connections = len(connection_manager.dashboard_connections)
    atm_connections = sum(
        len(conns) for conns in connection_manager.atm_detail_connections.values()
    )
    alerts_connections = len(connection_manager.alerts_connections)

    return {
        "status": "active",
        "dashboard_connections": dashboard_connections,
        "atm_detail_connections": atm_connections,
        "alerts_connections": alerts_connections,
        "total_connections": dashboard_connections
        + atm_connections
        + alerts_connections,
        "monitored_atms": list(connection_manager.atm_detail_connections.keys()),
    }
