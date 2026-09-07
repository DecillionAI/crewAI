"""The Caspar client wire protocol, as spoken over a WebSocket.

Frames match `drivers/network/framing.rs`:

    request  : u32be(len) | 0x03 | lp(signature) | lp(userId) | lp(path)
                          | lp(packetId) | payload
    response :             0x02 | lp(packetId)  | u32be(resCode) | payload
    update   :             0x01 | lp(key)       | payload

    lp(x) = u32be(len(x)) || x

The asymmetry is deliberate and matches the node: a client wraps every outgoing
WebSocket message in the same 4-byte length prefix the TCP transport uses (the
node strips it), while the node's own frames arrive without one, since a
WebSocket message is already delimited.

The bridge is an anonymous client — it signs nothing — so `signature` and
`userId` are empty on every request. Its authority is the bearer token inside
the payload, which the node checks.
"""

from __future__ import annotations

import struct
from typing import Tuple

TAG_UPDATE = 0x01
TAG_RESPONSE = 0x02
TAG_REQUEST = 0x03

#: Single byte a client sends back to acknowledge a delivered response frame.
#: The node holds the next response until it arrives.
ACK_FRAME = struct.pack(">I", 1) + bytes([TAG_UPDATE])

MAX_FRAME_BYTES = 20_000_000


def _lp(data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + data


def _lp_str(value: str) -> bytes:
    return _lp(value.encode("utf-8"))


def encode_request(path: str, packet_id: str, payload: bytes) -> bytes:
    """Encode one anonymous request, length-prefixed and ready to send."""
    body = (
        bytes([TAG_REQUEST])
        + _lp_str("")  # signature: the bridge holds no key
        + _lp_str("")  # userId: it is not a Caspar user
        + _lp_str(path)
        + _lp_str(packet_id)
        + payload
    )
    return struct.pack(">I", len(body)) + body


def _read_lp(data: bytes, pos: int) -> Tuple[bytes, int]:
    if pos + 4 > len(data):
        raise ValueError("truncated length-prefixed field")
    size = struct.unpack(">I", data[pos : pos + 4])[0]
    end = pos + 4 + size
    if size > MAX_FRAME_BYTES or end > len(data):
        raise ValueError("length-prefixed field runs past the frame")
    return data[pos + 4 : end], end


def decode_frame(frame: bytes) -> dict:
    """Decode one inbound frame from the node.

    Returns `{"kind": "response", "packetId", "code", "payload"}` or
    `{"kind": "update", "key", "payload"}`. An unrecognised tag is reported
    rather than guessed at, so a protocol change surfaces as an error instead
    of as silently dropped traffic.
    """
    if not frame:
        raise ValueError("empty frame")
    # Tolerate an outer length prefix in case a transport adds one.
    if len(frame) > 4 and frame[0] == 0x00:
        declared = struct.unpack(">I", frame[:4])[0]
        if declared == len(frame) - 4:
            frame = frame[4:]
    tag = frame[0]
    if tag == TAG_RESPONSE:
        packet_id, pos = _read_lp(frame, 1)
        if pos + 4 > len(frame):
            raise ValueError("response frame has no result code")
        code = struct.unpack(">I", frame[pos : pos + 4])[0]
        return {
            "kind": "response",
            "packetId": packet_id.decode("utf-8", "replace"),
            "code": code,
            "payload": frame[pos + 4 :],
        }
    if tag == TAG_UPDATE:
        key, pos = _read_lp(frame, 1)
        return {
            "kind": "update",
            "key": key.decode("utf-8", "replace"),
            "payload": frame[pos:],
        }
    raise ValueError(f"unsupported frame tag: 0x{tag:02x}")
