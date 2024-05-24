package com.example;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

public class SafeTensorsDeserializer {

    public static void main(String[] args) {
        String vectorsFilePath = "python/output/vectors.safetensors";
        String outputJsonFilePath = "data.json";

        try {
            // Read and deserialize the SafeTensors file
            byte[] vectorsData = Files.readAllBytes(Paths.get(vectorsFilePath));

            // Deserialize vectors
            Map<String, Object> vectorsHeader = parseHeader(vectorsData);
            double[][] vectors = extractVectors(vectorsData, vectorsHeader);

            // Prepare the output data structure
            Map<String, Object> outputData = new HashMap<>();
            outputData.put("vectors", vectors);

            // Serialize the output data to JSON and save to file
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.writeValue(new File(outputJsonFilePath), outputData);

            System.out.println("Deserialized data saved to " + outputJsonFilePath);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static Map<String, Object> parseHeader(byte[] data) throws IOException {
        ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        long headerSize = buffer.getLong();
        byte[] headerBytes = new byte[(int) headerSize];
        buffer.get(headerBytes);
        String headerJson = new String(headerBytes, StandardCharsets.UTF_8).trim();
        System.out.println("Header JSON: " + headerJson);
        ObjectMapper objectMapper = new ObjectMapper();
        return objectMapper.readValue(headerJson, Map.class);
    }

    private static double[][] extractVectors(byte[] data, Map<String, Object> header) {
        Map<String, Object> vectorsInfo = (Map<String, Object>) header.get("vectors");
        String dtype = (String) vectorsInfo.get("dtype");
        List<Integer> shapeList = (List<Integer>) vectorsInfo.get("shape");
        int rows = shapeList.get(0);
        int cols = shapeList.get(1);
        List<Number> dataOffsets = (List<Number>) vectorsInfo.get("data_offsets");
        long begin = dataOffsets.get(0).longValue();
        long end = dataOffsets.get(1).longValue();

        System.out.println("Vectors shape: " + rows + "x" + cols);
        System.out.println("Data offsets: " + begin + " to " + end);
        System.out.println("Data type: " + dtype);

        ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        // Correctly position the buffer to start reading after the header
        buffer.position((int) (begin + buffer.getLong(0) + 8));

        double[][] vectors = new double[rows][cols];
        if (dtype.equals("F64")) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    vectors[i][j] = buffer.getDouble();
                }
            }
        } else {
            throw new UnsupportedOperationException("Unsupported data type: " + dtype);
        }

        // Log the first few rows and columns to verify the content
        System.out.println("First few vectors:");
        for (int i = 0; i < Math.min(5, rows); i++) {
            for (int j = 0; j < Math.min(10, cols); j++) {
                System.out.print(vectors[i][j] + " ");
            }
            System.out.println();
        }

        return vectors;
    }
}
