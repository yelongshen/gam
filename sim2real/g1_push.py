#!/usr/bin/env python3
"""Upload a local file to the G1 and optionally run a remote command.

Usage:
    .venv_sim/bin/python sim2real/g1_push.py LOCAL REMOTE [--run "cmd"]
"""
import argparse
import os

import paramiko

HOST = os.environ.get("G1_HOST", "192.168.8.122")
USER = os.environ.get("G1_USER", "unitree")
PASS = os.environ.get("G1_PASS", "123")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("local")
    ap.add_argument("remote")
    ap.add_argument("--run", help="command to execute after upload")
    args = ap.parse_args()

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASS,
              allow_agent=False, look_for_keys=False, timeout=15)
    sftp = c.open_sftp()
    sftp.put(args.local, args.remote)
    print(f"[push] {args.local} -> {USER}@{HOST}:{args.remote} "
          f"({sftp.stat(args.remote).st_size} bytes)")
    sftp.close()

    if args.run:
        _in, out, err = c.exec_command(args.run, timeout=180)
        _in.close()
        o = out.read().decode("utf-8", "replace")
        e = err.read().decode("utf-8", "replace")
        if o:
            print(o, end="")
        if e:
            print("--- stderr ---\n" + e[-1500:], end="")
    c.close()


if __name__ == "__main__":
    raise SystemExit(main())
