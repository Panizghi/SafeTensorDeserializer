package com.example;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;

public class SafeTensorHeader {
    public long headerLength;
    public JsonNode metadata;

    public static SafeTensorHeader fromByteArray(byte[] buffer) throws IOException {
        if (buffer.length < 8) {
            throw new IllegalArgumentException("Buffer too small to contain valid safetensor header.");
        }

        long headerLength = bytesToLong(buffer, 0);
        if (headerLength > 100_000_000) {
            throw new IllegalArgumentException("Header too large.");
        }

        int metadataStart = 8;
        int metadataEnd = metadataStart + (int) headerLength;
        if (metadataEnd > buffer.length) {
            throw new IllegalArgumentException("Buffer too small to contain full header.");
        }

        String headerJson = new String(buffer, metadataStart, (int) headerLength);
        ObjectMapper mapper = new ObjectMapper();
        JsonNode metadata = mapper.readTree(headerJson);

        SafeTensorHeader header = new SafeTensorHeader();
        header.headerLength = headerLength;
        header.metadata = metadata;
        return header;
    }

    private static long bytesToLong(byte[] bytes, int offset) {
        return ((long) bytes[offset] & 0xff) |
               ((long) bytes[offset + 1] & 0xff) << 8 |
               ((long) bytes[offset + 2] & 0xff) << 16 |
               ((long) bytes[offset + 3] & 0xff) << 24 |
               ((long) bytes[offset + 4] & 0xff) << 32 |
               ((long) bytes[offset + 5] & 0xff) << 40 |
               ((long) bytes[offset + 6] & 0xff) << 48 |
               ((long) bytes[offset + 7] & 0xff) << 56;
    }
}
