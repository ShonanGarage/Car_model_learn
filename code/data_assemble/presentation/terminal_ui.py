import sys
import threading
import queue
import socket
from typing import List, Set, Tuple

try:
    import keyboard
except ImportError as exc:
    raise ImportError(
        "Missing dependency 'keyboard'. Install it to enable key up/down handling."
    ) from exc

class TerminalUI:
    def __init__(self, udp_port: int = 5005):
        self._command_queue = queue.Queue()
        self._event_queue = queue.Queue()
        self._pressed: Set[str] = set()
        self._stop_event = threading.Event()

        print("Control: W (Forward), S (Backward), X/Space (Stop), A (Left), D (Right), C (Center), Q (Quit)")
        # 1. Listen for keyboard input (supports key up/down)
        self._kb_hook = keyboard.hook(self._handle_key_event)
        
        # 2. Listen for UDP packets (Reference from user's script)
        self._udp_port = udp_port
        self._udp_thread = threading.Thread(target=self._listen_udp, daemon=True)
        self._udp_thread.start()

    def _handle_key_event(self, event) -> None:
        if self._stop_event.is_set():
            return
        key = (event.name or "").lower()
        if key == "space":
            key = " "
        if not key:
            return
        if event.event_type == "down":
            if key in self._pressed:
                return
            self._pressed.add(key)
        elif event.event_type == "up":
            self._pressed.discard(key)
        self._event_queue.put((key, event.event_type))

    def _listen_udp(self):
        """UDP server to receive remote control packets (Reference logic)"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)
        try:
            sock.bind(("0.0.0.0", self._udp_port))
            while not self._stop_event.is_set():
                try:
                    data, _ = sock.recvfrom(1024)
                    cmd = data.decode("utf-8").strip().lower()
                    if cmd:
                        self._command_queue.put(cmd)
                except socket.timeout:
                    continue
        except Exception as e:
            print(f"UDP Listener Error: {e}")
        finally:
            sock.close()

    def get_commands(self) -> List[str]:
        """Returns all queued UDP commands in order, draining the queue."""
        cmds: List[str] = []
        try:
            while not self._command_queue.empty():
                new_cmd = self._command_queue.get_nowait()
                cmds.append(new_cmd)
        except queue.Empty:
            pass
        return cmds

    def get_key_events(self) -> List[Tuple[str, str]]:
        """Returns all queued key events (key, event_type), draining the queue."""
        events: List[Tuple[str, str]] = []
        try:
            while not self._event_queue.empty():
                events.append(self._event_queue.get_nowait())
        except queue.Empty:
            pass
        return events

    def get_pressed_keys(self) -> Set[str]:
        """Returns a snapshot of currently pressed keys."""
        return set(self._pressed)

    def display_status(self, state: str, distances: list, steer_us: int):
        # Move cursor up or print status line
        dist_str = " / ".join([f"{d:.2f}m" if d > 0 else "ERR" for d in distances])
        sys.stdout.write(f"\rStatus: {state} | Steer: {steer_us}us | Sonars: {dist_str}    ")
        sys.stdout.flush()

    def stop(self):
        self._stop_event.set()
        keyboard.unhook(self._kb_hook)
