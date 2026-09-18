#!/usr/bin/env python3
"""Run a shell command on the G1's onboard computer over SSH (password auth).

Usage:
    .venv_sim/bin/python sim2real/g1_ssh.py "hostname; uname -a"
    .venv_sim/bin/python sim2real/g1_ssh.py --file probe.sh

Host/credentials can be overridden with G1_HOST / G1_USER / G1_PASS env vars.
"""
import argparse
import os
import sys

import paramiko

HOST = os.environ.get("G1_HOST", "192.168.8.122")
USER = os.environ.get("G1_USER", "unitree")
PASS = os.environ.get("G1_PASS", "123")


def run(command: str, timeout: float = 120.0) -> int:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST, username=USER, password=PASS, timeout=15,
        allow_agent=False, look_for_keys=False,
    )
    try:
        stdin, stdout, stderr = client.exec_command(
            command, timeout=timeout, get_pty=False,
        )
        stdin.close()
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        rc = stdout.channel.recv_exit_status()
        if out:
            sys.stdout.write(out)
        if err:
            sys.stdout.write("\n--- stderr ---\n" + err)
        return rc
    finally:
        client.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", help="command string to run remotely")
    ap.add_argument("--file", help="read the command/script from a local file")
    args = ap.parse_args()

    if args.file:
        cmd = open(args.file).read()
    elif args.command:
        cmd = args.command
    else:
        ap.error("provide a command or --file")
    return run(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
