#!/usr/bin/env python3
"""
check_pico_buttons.py
========================
Directly poll and print PICO controller button states in real time to
verify the SDK is detecting button presses at all (A, B, X, Y, triggers,
grips, menu buttons).

Usage
-----
    conda run -n xr python check_pico_buttons.py --seconds 20
"""

import argparse
import time

import xrobotoolkit_sdk as xrt


def safe_call(fn_name):
    fn = getattr(xrt, fn_name, None)
    if fn is None:
        return f"N/A({fn_name})"
    try:
        return fn()
    except Exception as e:
        return f"ERR({e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--hz", type=float, default=10.0)
    args = ap.parse_args()

    print("=" * 70)
    print("PICO Controller Button State Checker")
    print("=" * 70)
    xrt.init()
    time.sleep(0.5)

    print("Available xrt functions related to buttons:")
    button_fns = [f for f in dir(xrt) if "button" in f.lower() or f in (
        "get_A_button", "get_B_button", "get_X_button", "get_Y_button")]
    for f in sorted(button_fns):
        print(f"  {f}")
    print()

    print("Press buttons on the PICO controllers now. Printing every "
          f"{1.0/args.hz:.2f}s for {args.seconds}s...\n")

    dt = 1.0 / args.hz
    start_t = time.time()
    last_state = None

    while time.time() - start_t < args.seconds:
        a = safe_call("get_A_button")
        b = safe_call("get_B_button")
        x = safe_call("get_X_button")
        y = safe_call("get_Y_button")
        lt = safe_call("get_left_trigger")
        rt = safe_call("get_right_trigger")
        lg = safe_call("get_left_grip")
        rg = safe_call("get_right_grip")
        lm = safe_call("get_left_menu_button")
        rm = safe_call("get_right_menu_button")

        state = (a, b, x, y, lm, rm)
        elapsed = time.time() - start_t

        # Always print if any button is truthy, else print every 2s as heartbeat
        any_pressed = any(bool(v) is True for v in (a, b, x, y, lm, rm))
        if any_pressed or state != last_state or int(elapsed) % 2 == 0:
            print(f"[{elapsed:6.2f}s] A={a} B={b} X={x} Y={y} "
                  f"L_menu={lm} R_menu={rm} "
                  f"L_trig={lt} R_trig={rt} L_grip={lg} R_grip={rg}")

        last_state = state
        time.sleep(dt)

    print("\nDone.")
    if hasattr(xrt, "close"):
        xrt.close()


if __name__ == "__main__":
    main()
