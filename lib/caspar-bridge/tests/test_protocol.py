"""The wire protocol must match the node's framing exactly, byte for byte."""

import struct

import pytest

from decillion_caspar_bridge.protocol import (
    ACK_FRAME,
    TAG_REQUEST,
    decode_frame,
    encode_request,
)


def test_request_is_length_prefixed_and_anonymous():
    frame = encode_request("/gateway/subscribe", "pkt1", b'{"token":"t"}')
    declared = struct.unpack(">I", frame[:4])[0]
    assert declared == len(frame) - 4
    body = frame[4:]
    assert body[0] == TAG_REQUEST
    # signature and userId are both empty: the bridge holds no key, and its
    # authority is the token inside the payload.
    assert struct.unpack(">I", body[1:5])[0] == 0
    assert struct.unpack(">I", body[5:9])[0] == 0


def test_decodes_a_response_frame():
    payload = b'{"ok":true}'
    body = (
        bytes([0x02])
        + struct.pack(">I", 4)
        + b"pkt1"
        + struct.pack(">I", 0)
        + payload
    )
    frame = decode_frame(body)
    assert frame == {
        "kind": "response",
        "packetId": "pkt1",
        "code": 0,
        "payload": payload,
    }


def test_decodes_an_update_frame():
    payload = b'{"runId":"r1"}'
    body = bytes([0x01]) + struct.pack(">I", 11) + b"crew/prompt" + payload
    frame = decode_frame(body)
    assert frame["kind"] == "update"
    assert frame["key"] == "crew/prompt"
    assert frame["payload"] == payload


def test_ack_is_a_single_tagged_byte():
    assert ACK_FRAME == struct.pack(">I", 1) + bytes([0x01])


def test_unknown_tag_is_reported_not_guessed():
    with pytest.raises(ValueError):
        decode_frame(bytes([0x09]) + b"whatever")


def test_truncated_field_is_rejected():
    with pytest.raises(ValueError):
        decode_frame(bytes([0x01]) + struct.pack(">I", 999) + b"short")
