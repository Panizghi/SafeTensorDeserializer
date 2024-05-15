package com.example;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
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
        String outputFilePath = "output.json";

        // Read file
        byte[] buffer = readFile(filePath);

        // Parse header
        SafeTensorHeader header = SafeTensorHeader.fromByteArray(buffer);

        // Extract tensor info
        List<TensorInfo> tensorInfos = extractTensorInfo(header.metadata);

        // Extract tensor data
        List<Map<String, Object>> tensorDataList = new ArrayList<>();
        for (TensorInfo tensorInfo : tensorInfos) {
            if ("F32".equals(tensorInfo.dtype)) {
                float[][] vectorsData = extractFloatTensor(buffer, tensorInfo);
                for (float[] vector : vectorsData) {
                    tensorDataList.add(Map.of(
                        "dtype", tensorInfo.dtype,
                        "shape", tensorInfo.shape,
                        "vector", vector
                    ));
                }
            } else if ("I32".equals(tensorInfo.dtype)) {
                int[][] vectorsData = extractIntTensor(buffer, tensorInfo);
                for (int[] vector : vectorsData) {
                    tensorDataList.add(Map.of(
                        "dtype", tensorInfo.dtype,
                        "shape", tensorInfo.shape,
                        "vector", vector
                    ));
                }
            } else if ("I64".equals(tensorInfo.dtype)) {
                long[][] vectorsData = extractLongTensor(buffer, tensorInfo);
                for (long[] vector : vectorsData) {
                    tensorDataList.add(Map.of(
                        "dtype", tensorInfo.dtype,
                        "shape", tensorInfo.shape,
                        "vector", vector
                    ));
                }
            } else if ("F64".equals(tensorInfo.dtype)) {
                double[][] vectorsData = extractDoubleTensor(buffer, tensorInfo);
                for (double[] vector : vectorsData) {
                    tensorDataList.add(Map.of(
                        "dtype", tensorInfo.dtype,
                        "shape", tensorInfo.shape,
                        "vector", vector
                    ));
                }
            } else {
                System.out.println("Unsupported data type: " + tensorInfo.dtype);
            }
        }

        // Save the deserialized data to a JSON file
        saveToJson(tensorDataList, outputFilePath);
        System.out.println("Saved deserialized data to " + outputFilePath);
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

    public static float[][] extractFloatTensor(byte[] buffer, TensorInfo tensorInfo) {
        int start = tensorInfo.dataOffsets[0];
        int end = tensorInfo.dataOffsets[1];
        int length = (end - start) / 4; // 4 bytes per float

        // Assuming the shape is [number_of_vectors, vector_size]
        int numberOfVectors = tensorInfo.shape.get(0);
        int vectorSize = tensorInfo.shape.get(1);

        float[][] tensorData = new float[numberOfVectors][vectorSize];

        for (int i = 0; i < numberOfVectors; i++) {
            for (int j = 0; j < vectorSize; j++) {
                int index = start + 4 * (i * vectorSize + j);
                tensorData[i][j] = Float.intBitsToFloat(
                    ((buffer[index] & 0xff)) |
                    ((buffer[index + 1] & 0xff) << 8) |
                    ((buffer[index + 2] & 0xff) << 16) |
                    ((buffer[index + 3] & 0xff) << 24)
                );
            }
        }

        return tensorData;
    }

    public static int[][] extractIntTensor(byte[] buffer, TensorInfo tensorInfo) {
        int start = tensorInfo.dataOffsets[0];
        int end = tensorInfo.dataOffsets[1];
        int length = (end - start) / 4; // 4 bytes per int

        // Assuming the shape is [number_of_vectors, vector_size]
        int numberOfVectors = tensorInfo.shape.get(0);
        int vectorSize = tensorInfo.shape.get(1);

        int[][] tensorData = new int[numberOfVectors][vectorSize];

        for (int i = 0; i < numberOfVectors; i++) {
            for (int j = 0; j < vectorSize; j++) {
                int index = start + 4 * (i * vectorSize + j);
                tensorData[i][j] = 
                    ((buffer[index] & 0xff)) |
                    ((buffer[index + 1] & 0xff) << 8) |
                    ((buffer[index + 2] & 0xff) << 16) |
                    ((buffer[index + 3] & 0xff) << 24);
            }
        }

        return tensorData;
    }

    public static long[][] extractLongTensor(byte[] buffer, TensorInfo tensorInfo) {
        int start = tensorInfo.dataOffsets[0];
        int end = tensorInfo.dataOffsets[1];
        int length = (end - start) / 8; // 8 bytes per long

        // Assuming the shape is [number_of_vectors, vector_size]
        int numberOfVectors = tensorInfo.shape.get(0);
        int vectorSize = tensorInfo.shape.get(1);

        long[][] tensorData = new long[numberOfVectors][vectorSize];

        for (int i = 0; i < numberOfVectors; i++) {
            for (int j = 0; j < vectorSize; j++) {
                int index = start + 8 * (i * vectorSize + j);
                tensorData[i][j] = 
                    ((long) buffer[index] & 0xff) |
                    ((long) (buffer[index + 1] & 0xff) << 8) |
                    ((long) (buffer[index + 2] & 0xff) << 16) |
                    ((long) (buffer[index + 3] & 0xff) << 24) |
                    ((long) (buffer[index + 4] & 0xff) << 32) |
                    ((long) (buffer[index + 5] & 0xff) << 40) |
                    ((long) (buffer[index + 6] & 0xff) << 48) |
                    ((long) (buffer[index + 7] & 0xff) << 56);
            }
        }

        return tensorData;
    }

    public static double[][] extractDoubleTensor(byte[] buffer, TensorInfo tensorInfo) {
        int start = tensorInfo.dataOffsets[0];
        int end = tensorInfo.dataOffsets[1];
        int length = (end - start) / 8; // 8 bytes per double

        // Assuming the shape is [number_of_vectors, vector_size]
        int numberOfVectors = tensorInfo.shape.get(0);
        int vectorSize = tensorInfo.shape.get(1);

        double[][] tensorData = new double[numberOfVectors][vectorSize];

        for (int i = 0; i < numberOfVectors; i++) {
            for (int j = 0; j < vectorSize; j++) {
                int index = start + 8 * (i * vectorSize + j);
                long bits = 
                    ((long) buffer[index] & 0xff) |
                    ((long) (buffer[index + 1] & 0xff) << 8) |
                    ((long) (buffer[index + 2] & 0xff) << 16) |
                    ((long) (buffer[index + 3] & 0xff) << 24) |
                    ((long) (buffer[index + 4] & 0xff) << 32) |
                    ((long) (buffer[index + 5] & 0xff) << 40) |
                    ((long) (buffer[index + 6] & 0xff) << 48) |
                    ((long) (buffer[index + 7] & 0xff) << 56);
                tensorData[i][j] = Double.longBitsToDouble(bits);
            }
        }

        return tensorData;
    }

    public static void saveToJson(List<Map<String, Object>> data, String outputFilePath) throws IOException {
        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.writeValue(new File(outputFilePath), data);
    }
}
