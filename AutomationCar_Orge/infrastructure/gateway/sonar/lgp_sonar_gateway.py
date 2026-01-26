import time
import lgpio
import threading
from typing import List
from internal.interface.gateway.sonar_gateway_interface import SonarGatewayInterface
from app.config.settings import Settings

class LgpSonarGateway(SonarGatewayInterface):
    def __init__(self, settings: Settings, handle: int):
        self.settings = settings
        self.handle = handle
        self.trig_echo_pairs = [
            (s.trig_gpio, s.echo_gpio) for s in settings.sonar.values()
        ]
        
        for trig, echo in self.trig_echo_pairs:
            self._claim_output(trig)
            self._claim_input(echo)

        # Background thread for continuous reading
        self._last_distances = [-1.0] * len(self.trig_echo_pairs)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _claim_output(self, gpio: int):
        lgpio.gpio_claim_output(self.handle, gpio)

    def _claim_input(self, gpio: int):
        lgpio.gpio_claim_input(self.handle, gpio)

    def read_distances_m(self) -> List[float]:
        """Returns the latest distances from the background thread instantly."""
        return list(self._last_distances)

    def _worker(self):
        while not self._stop_event.is_set():
            new_distances = []
            for trig, echo in self.trig_echo_pairs:
                dist = self._read_single_distance_m(trig, echo)
                new_distances.append(dist)
                time.sleep(self.settings.sonar_inter_gap_s)
            self._last_distances = new_distances
            time.sleep(0.01) # Small rest between full sweeps

    def _read_single_distance_m(self, trig: int, echo: int) -> float:
        try:
            lgpio.gpio_write(self.handle, trig, 0)
            time.sleep(0.00001)
            lgpio.gpio_write(self.handle, trig, 1)
            time.sleep(0.00001)
            lgpio.gpio_write(self.handle, trig, 0)

            t0 = time.time()
            while lgpio.gpio_read(self.handle, echo) == 0:
                if time.time() - t0 > self.settings.sonar_timeout_s:
                    return -1.0

            start = time.time()
            while lgpio.gpio_read(self.handle, echo) == 1:
                if time.time() - start > self.settings.sonar_timeout_s:
                    return -1.0

            end = time.time()
            pulse = end - start
            dist_m = (pulse * 343.0) / 2.0

            if dist_m < 0.02 or dist_m > 5.0:
                return -1.0
            return dist_m
        except Exception:
            return -1.0

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)
