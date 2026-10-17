#!/usr/bin/env python3
"""Hardware check: send set_pixels through the Arduino Router and print the RPC reply.

Usage: python3 hw_check.py [r g b brightness count]   (default: 0 255 0 50 8)
"""

import socket
import sys

import msgpack

params = [int(a) for a in sys.argv[1:6]] or [0, 255, 0, 50, 8]
if len(params) != 5:
    sys.exit("usage: hw_check.py r g b brightness count")

with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
    s.settimeout(3)
    s.connect("/var/run/arduino-router.sock")
    s.sendall(msgpack.packb([0, 1, "set_pixels", params], use_bin_type=True))
    print("set_pixels", params, "-> RPC response:", msgpack.unpackb(s.recv(1024), raw=False))
