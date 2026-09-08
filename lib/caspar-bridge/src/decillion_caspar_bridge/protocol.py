"""The Caspar client wire protocol, as spoken over a WebSocket.

    request  : u32be(len) | lp(signature) | lp(userId) | lp(path)
                          | lp(packetId) | payload
    response :              0x02 | lp(packetId) | u32be(resCode) | payload
    update   :              0x01 | lp(key)      | payload

    lp(x) = u32be(len(x)) || x

The shape is asymmetric in two ways, and both matter:

* **Only requests carry the 4-byte length prefix.** A client wraps every
  outgoing message in the same prefix the TCP transport uses and the node
  strips it (`Ws::handle_connection`); the node's own frames arrive without
  one, because a WebSocket message is already delimited.

* **Only the node's frames carry a TAG byte.** A response is `0x02`, an update
  `0x01` — but a request has none. The node strips the length prefix and hands
  the remainder straight to `decode_request_body`, which reads
  `lp(signature)` from byte zero. A tag byte here shifts every field by one and
  the node cannot parse the frame at all: it rejects it with "lp field too
  large: 50331648", which is `0x03000000` — the tag and the first three bytes
  of the signature's length, read as one number. Nothing answers, and a bridge
  that sent one sat in a connect / 30-second-timeout / reconnect loop forever
  with an empty error, because that is what `asyncio.wait_for` raises.

  `framing.rs` does define tagged request frames — for FEDERATION, between
  nodes. The client transport is the untagged one. Do not "restore" the tag.

The bridge is an anonymous client — it signs nothing — so `signature` and
`userId` are empty on every request. Its authority is the bearer token inside
the payload, which the node checks.
"""

from __future__ import annotations

import struct
from typing import Tuple

#: Tags the NODE puts on the frames it sends. A request carries none.
TAG_UPDATE = 0x01
TAG_RESPONSE = 0x02

#: Single byte a client sends back to acknowledge a delivered response frame.
#: The node holds the next response until it arrives.
ACK_FRAME = struct.pack(">I", 1) + bytes([TAG_UPDATE])

MAX_FRAME_BYTES = 20_000_000


def _lp(data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + data


def _lp_str(value: str) -> bytes:
    return _lp(value.encode("utf-8"))


def encode_request(path: str, packet_id: str, payload: bytes) -> bytes:
    """Encode one anonymous request, length-prefixed and ready to send.

    No tag byte: the node's client transport does not expect one on a request.
    See the module docstring for what happens when there is one.
    """
    body = (
        _lp_str("")  # signature: the bridge holds no key
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
