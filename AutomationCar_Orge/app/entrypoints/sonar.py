import argparse
import sys
import time
from pathlib import Path

try:
    import lgpio
except Exception as exc:  # pragma: no cover - runtime environment guard
    print(f"lgpio import failed: {exc}")
    print("Install lgpio or run on hardware that supports it.")
    sys.exit(1)

from app.config.settings import load_settings


def _read_distance_m(handle: int, trig: int, echo: int, timeout_s: float) -> float:
    lgpio.gpio_write(handle, trig, 0)
    time.sleep(0.00001)
    lgpio.gpio_write(handle, trig, 1)
    time.sleep(0.00001)
    lgpio.gpio_write(handle, trig, 0)

    t0 = time.perf_counter()
    while lgpio.gpio_read(handle, echo) == 0:
        if time.perf_counter() - t0 > timeout_s:
            return -1.0

    start = time.perf_counter()
    while lgpio.gpio_read(handle, echo) == 1:
        if time.perf_counter() - start > timeout_s:
            return -1.0

    end = time.perf_counter()
    pulse = end - start
    dist_m = (pulse * 343.0) / 2.0

    if dist_m < 0.02 or dist_m > 5.0:
        return -1.0
    return dist_m


def _format_distance(dist_m: float) -> str:
    if dist_m <= 0:
        return "ERR"
    return f"{dist_m * 100.0:.1f}cm"


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple sonar test based on config.yaml")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument("--once", action="store_true", help="Measure once and exit")
    args = parser.parse_args()

    settings = load_settings(args.config)
    if not settings.sonar:
        print("No sonar settings found.")
        return

    handle = lgpio.gpiochip_open(0)
    if handle < 0:
        raise RuntimeError(f"gpiochip_open(0) failed: {handle}")

    try:
        for module in settings.sonar.values():
            lgpio.gpio_claim_output(handle, module.trig_gpio)
            lgpio.gpio_claim_input(handle, module.echo_gpio)
            lgpio.gpio_write(handle, module.trig_gpio, 0)

        print("Sonar test started. Ctrl+C to stop.")
        print("Sensors:", ", ".join(settings.sonar.keys()))

        while True:
            results = {}
            for name, module in settings.sonar.items():
                dist = _read_distance_m(
                    handle=handle,
                    trig=module.trig_gpio,
                    echo=module.echo_gpio,
                    timeout_s=settings.sonar_timeout_s,
                )
                results[name] = dist
                time.sleep(settings.sonar_inter_gap_s)

            line = " | ".join(f"{name}: {_format_distance(dist)}" for name, dist in results.items())
            print(line)

            if args.once:
                break
            time.sleep(settings.sonar_sweep_sleep_s)
    except KeyboardInterrupt:
        pass
    finally:
        lgpio.gpiochip_close(handle)
        print("Sonar test stopped.")


if __name__ == "__main__":
    main()
