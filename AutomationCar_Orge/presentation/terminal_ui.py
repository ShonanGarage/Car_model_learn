import sys
import threading
import queue
import termios
import tty
import socket
from typing import Optional

class TerminalUI:
    def __init__(self, udp_port: int = 5005):
        self._input_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._old_settings = termios.tcgetattr(sys.stdin)
        
        # 1. Listen for terminal keyboard input
        self._kb_thread = threading.Thread(target=self._listen_keyboard, daemon=True)
        self._kb_thread.start()
        
        # 2. Listen for UDP packets (Reference from user's script)
        self._udp_port = udp_port
        self._udp_thread = threading.Thread(target=self._listen_udp, daemon=True)
        self._udp_thread.start()

    def _listen_keyboard(self):
        print("Control: W (Forward), S (Backward), X/Space (Stop), A (Left), D (Right), C (Center), Q (Quit)")
        try:
            tty.setcbreak(sys.stdin.fileno())
            while not self._stop_event.is_set():
                char = sys.stdin.read(1).lower()
                if not char:
                    break
                # Convert Space to ' ' and handle other keys
                self._input_queue.put(char)
        except Exception:
            pass
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

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
                        self._input_queue.put(cmd)
                except socket.timeout:
                    continue
        except Exception as e:
            print(f"UDP Listener Error: {e}")
        finally:
            sock.close()

    def get_command(self) -> Optional[str]:
        """Returns the most recent command from the queue, draining it.
        This prevents input lag if the loop slows down.
        """
        cmd = None
        try:
            while not self._input_queue.empty():
                new_cmd = self._input_queue.get_nowait()
                if new_cmd == 'q': # prioritize quit
                    return 'q'
                cmd = new_cmd
            return cmd
        except queue.Empty:
            return cmd

    def display_status(self, state: str, distances: list, steer_us: int):
        # Move cursor up or print status line
        dist_str = " / ".join([f"{d:.2f}m" if d > 0 else "ERR" for d in distances])
        sys.stdout.write(f"\rStatus: {state} | Steer: {steer_us}us | Sonars: {dist_str}    ")
        sys.stdout.flush()

    def stop(self):
        self._stop_event.set()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
