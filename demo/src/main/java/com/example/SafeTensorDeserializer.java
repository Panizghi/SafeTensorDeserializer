package com.example;

import com.fasterxml.jackson.databind.JsonNode;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class SafeTensorDeserializer {

    public static void main(String[] args) throws IOException {
        // Change this to the path of your safetensor file
        String filePath = "/Users/paniz/Documents/GitHub/jimmy/melissa/anserini/collections/safetensors/vectors.safetensors";

        // Read file
        byte[] buffer = readFile(filePath);

        // Parse header
        SafeTensorHeader header = SafeTensorHeader.fromByteArray(buffer);

        // Extract tensor info
        List<TensorInfo> tensorInfos = extractTensorInfo(header.metadata);

        // Extract tensor data
        for (TensorInfo tensorInfo : tensorInfos) {
            switch (tensorInfo.dtype) {
                case "F32":
                    float[] floatData = extractFloatTensor(buffer, tensorInfo.dataOffsets);
                    System.out.println("Tensor data (float): ");
                    for (float value : floatData) {
                        System.out.print(value + " ");
                    }
                    System.out.println();
                    break;
                case "I32":
                    int[] intData = extractIntTensor(buffer, tensorInfo.dataOffsets);
                    System.out.println("Tensor data (int): ");
                    for (int value : intData) {
                        System.out.print(value + " ");
                    }
                    System.out.println();
                    break;
                case "I64":
                    long[] longData = extractLongTensor(buffer, tensorInfo.dataOffsets);
                    System.out.println("Tensor data (long): ");
                    for (long value : longData) {
                        System.out.print(value + " ");
                    }
                    System.out.println();
                    break;
                case "F64":
                    double[] doubleData = extractDoubleTensor(buffer, tensorInfo.dataOffsets);
                    System.out.println("Tensor data (double): ");
                    for (double value : doubleData) {
                        System.out.print(value + " ");
                    }
                    System.out.println();
                    break;
                // Handle other data types similarly
                default:
                    System.out.println("Unsupported data type: " + tensorInfo.dtype);
                    break;
            }
        }
    }

    public static byte[] readFile(String filePath) throws IOException {
        return Files.readAllBytes(Path.of(filePath));
    }

    public static List<TensorInfo> extractTensorInfo(JsonNode metadata) {
        List<TensorInfo> tensorInfos = new ArrayList<>();
        Iterator<Map.Entry<String, JsonNode>> fields = metadata.fields();

        while (fields.hasNext()) {
            Map.Entry<String, JsonNode> field = fields.next();
            if (!field.getKey().equals("__metadata__")) {
                JsonNode tensorNode = field.getValue();
                TensorInfo tensorInfo = new TensorInfo();
                tensorInfo.dtype = tensorNode.get("dtype").asText();
                tensorNode.get("shape").forEach(shapeNode -> tensorInfo.shape.add(shapeNode.asInt()));
                tensorInfo.dataOffsets[0] = tensorNode.get("data_offsets").get(0).asInt();
                tensorInfo.dataOffsets[1] = tensorNode.get("data_offsets").get(1).asInt();
                tensorInfos.add(tensorInfo);
            }
        }
        return tensorInfos;
    }

    public static float[] extractFloatTensor(byte[] buffer, int[] dataOffsets) {
        int start = dataOffsets[0];
        int end = dataOffsets[1];
        int length = (end - start) / 4; // 4 bytes per float
        float[] tensorData = new float[length];

        for (int i = 0; i < length; i++) {
            tensorData[i] = Float.intBitsToFloat(
                ((buffer[start + i * 4] & 0xff)) |
                ((buffer[start + i * 4 + 1] & 0xff) << 8) |
                ((buffer[start + i * 4 + 2] & 0xff) << 16) |
                ((buffer[start + i * 4 + 3] & 0xff) << 24)
            );
        }
        return tensorData;
    }

    public static int[] extractIntTensor(byte[] buffer, int[] dataOffsets) {
        int start = dataOffsets[0];
        int end = dataOffsets[1];
        int length = (end - start) / 4; // 4 bytes per int
        int[] tensorData = new int[length];

        for (int i = 0; i < length; i++) {
            tensorData[i] = 
                ((buffer[start + i * 4] & 0xff)) |
                ((buffer[start + i * 4 + 1] & 0xff) << 8) |
                ((buffer[start + i * 4 + 2] & 0xff) << 16) |
                ((buffer[start + i * 4 + 3] & 0xff) << 24);
        }
        return tensorData;
    }

    public static long[] extractLongTensor(byte[] buffer, int[] dataOffsets) {
        int start = dataOffsets[0];
        int end = dataOffsets[1];
        int length = (end - start) / 8; // 8 bytes per long
        long[] tensorData = new long[length];

        for (int i = 0; i < length; i++) {
            tensorData[i] = 
                ((long) buffer[start + i * 8] & 0xff) |
                ((long) (buffer[start + i * 8 + 1] & 0xff) << 8) |
                ((long) (buffer[start + i * 8 + 2] & 0xff) << 16) |
                ((long) (buffer[start + i * 8 + 3] & 0xff) << 24) |
                ((long) (buffer[start + i * 8 + 4] & 0xff) << 32) |
                ((long) (buffer[start + i * 8 + 5] & 0xff) << 40) |
                ((long) (buffer[start + i * 8 + 6] & 0xff) << 48) |
                ((long) (buffer[start + i * 8 + 7] & 0xff) << 56);
        }
        return tensorData;
    }

    public static double[] extractDoubleTensor(byte[] buffer, int[] dataOffsets) {
        int start = dataOffsets[0];
        int end = dataOffsets[1];
        int length = (end - start) / 8; // 8 bytes per double
        double[] tensorData = new double[length];

        for (int i = 0; i < length; i++) {
            long bits = 
                ((long) buffer[start + i * 8] & 0xff) |
                ((long) (buffer[start + i * 8 + 1] & 0xff) << 8) |
                ((long) (buffer[start + i * 8 + 2] & 0xff) << 16) |
                ((long) (buffer[start + i * 8 + 3] & 0xff) << 24) |
                ((long) (buffer[start + i * 8 + 4] & 0xff) << 32) |
                ((long) (buffer[start + i * 8 + 5] & 0xff) << 40) |
                ((long) (buffer[start + i * 8 + 6] & 0xff) << 48) |
                ((long) (buffer[start + i * 8 + 7] & 0xff) << 56);
            tensorData[i] = Double.longBitsToDouble(bits);
        }
        return tensorData;
    }
}
