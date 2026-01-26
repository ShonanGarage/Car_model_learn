import asyncio
import base64
import time
from enum import Enum
from typing import Any, Dict, Optional, Set

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.container import Container


class WsCommand(str, Enum):
    MOVE_FORWARD = "MOVE_FORWARD"
    MOVE_BACKWARD = "MOVE_BACKWARD"
    STOP = "STOP"
    STEER_LEFT = "STEER_LEFT"
    STEER_RIGHT = "STEER_RIGHT"
    RESET_STEER = "RESET_STEER"
    QUIT = "QUIT"


class WebSocketServer:
    def __init__(self, update_hz: int = 10):
        self.container = Container()
        self.container.drive_service.set_course_id(self.container.settings.course_id)
        self.update_hz = max(1, update_hz)
        self._connections: Set[WebSocket] = set()
        self._broadcast_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def ws_endpoint(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.add(websocket)
        if self._broadcast_task is None:
            self._broadcast_task = asyncio.create_task(self._broadcast_loop())

        try:
            while True:
                message = await websocket.receive_json()
                await self._handle_command(message)
        except WebSocketDisconnect:
            pass
        finally:
            self._connections.discard(websocket)
            if not self._connections:
                await self._shutdown_background()

    async def _handle_command(self, message: Dict[str, Any]) -> None:
        action = message.get("action")
        if not action:
            return

        try:
            cmd = WsCommand(action)
        except ValueError:
            return

        step = message.get("step", 100)

        async with self._lock:
            if cmd == WsCommand.MOVE_FORWARD:
                self.container.drive_service.move_forward()
            elif cmd == WsCommand.MOVE_BACKWARD:
                self.container.drive_service.move_backward()
            elif cmd == WsCommand.STOP:
                self.container.drive_service.stop()
            elif cmd == WsCommand.STEER_LEFT:
                self.container.drive_service.steer_left(step=step)
            elif cmd == WsCommand.STEER_RIGHT:
                self.container.drive_service.steer_right(step=step)
            elif cmd == WsCommand.RESET_STEER:
                self.container.drive_service.reset_steer()
            elif cmd == WsCommand.QUIT:
                await self._close_all()

    async def _broadcast_loop(self) -> None:
        interval = 1.0 / self.update_hz
        try:
            while self._connections:
                async with self._lock:
                    await asyncio.to_thread(self.container.drive_service.update)
                    payload = self._build_telemetry()

                await self._broadcast(payload)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

    def _build_telemetry(self) -> Dict[str, Any]:
        image_b64 = self._encode_frame(self.container.drive_service.frame)
        return {
            "type": "telemetry",
            "timestamp": time.time(),
            "state": self.container.drive_service.state.name,
            "distances": self.container.drive_service.distances,
            "steer_us": self.container.drive_service.current_steer_us,
            "throttle_us": self.container.drive_service.control.throttle.value,
            "image_jpeg_b64": image_b64,
        }

    def _encode_frame(self, frame: Optional[object]) -> Optional[str]:
        if frame is None:
            return None
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.container.settings.camera.jpeg_quality]
        ok, buf = cv2.imencode(".jpg", frame, encode_param)
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")

    async def _broadcast(self, payload: Dict[str, Any]) -> None:
        if not self._connections:
            return
        dead: Set[WebSocket] = set()
        for ws in self._connections:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self._connections.discard(ws)

    async def _close_all(self) -> None:
        for ws in list(self._connections):
            try:
                await ws.close()
            except Exception:
                pass
        self._connections.clear()

    async def _shutdown_background(self) -> None:
        if self._broadcast_task:
            self._broadcast_task.cancel()
            self._broadcast_task = None
        self.container.drive_service.stop()
        self.container.camera_gateway.release()
        self.container.camera_view.stop()
        self.container.data_repository.stop()


def create_app(update_hz: int = 10) -> FastAPI:
    server = WebSocketServer(update_hz=update_hz)
    app = FastAPI()
    app.add_api_websocket_route("/ws", server.ws_endpoint)

    @app.on_event("shutdown")
    async def _on_shutdown() -> None:
        await server._shutdown_background()

    return app


def main() -> None:
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
