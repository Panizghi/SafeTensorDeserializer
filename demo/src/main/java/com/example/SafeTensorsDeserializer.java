package com.example;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

public class SafeTensorsDeserializer {

    public static void main(String[] args) {
        String vectorsFilePath = "python/output/vectors.safetensors";
        String docidsFilePath = "python/output/docids.safetensors";
        String outputJsonFilePath = "data.json";

        try {
            // Read and deserialize the SafeTensors files
            byte[] vectorsData = Files.readAllBytes(Paths.get(vectorsFilePath));
            byte[] docidsData = Files.readAllBytes(Paths.get(docidsFilePath));

            // Deserialize vectors
            Map<String, Object> vectorsHeader = parseHeader(vectorsData);
            double[][] vectors = extractVectors(vectorsData, vectorsHeader);

            // Deserialize docids
            Map<String, Object> docidsHeader = parseHeader(docidsData);
            String[] docids = extractDocids(docidsData, docidsHeader);

            // Prepare the output data structure
            Map<String, Object> outputData = new HashMap<>();
            outputData.put("vectors", vectors);
            outputData.put("docids", docids);

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
        String headerJson = new String(headerBytes).trim();
        ObjectMapper objectMapper = new ObjectMapper();
        return objectMapper.readValue(headerJson, Map.class);
    }

    private static double[][] extractVectors(byte[] data, Map<String, Object> header) {
        Map<String, Object> vectorsInfo = (Map<String, Object>) header.get("vectors");
        List<Integer> shapeList = (List<Integer>) vectorsInfo.get("shape");
        int rows = shapeList.get(0);
        int cols = shapeList.get(1);
        List<Number> dataOffsets = (List<Number>) vectorsInfo.get("data_offsets");
        long begin = dataOffsets.get(0).longValue();
        long end = dataOffsets.get(1).longValue();

        System.out.println("Vectors shape: " + rows + "x" + cols);
        System.out.println("Data offsets: " + begin + " to " + end);

        ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        buffer.position((int) begin);

        double[][] vectors = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                vectors[i][j] = buffer.getDouble();
            }
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

    private static String[] extractDocids(byte[] data, Map<String, Object> header) {
        Map<String, Object> docidsInfo = (Map<String, Object>) header.get("docids");
        List<Integer> shapeList = (List<Integer>) docidsInfo.get("shape");
        int length = shapeList.get(0);
        List<Number> dataOffsets = (List<Number>) docidsInfo.get("data_offsets");
        long begin = dataOffsets.get(0).longValue();
        long end = dataOffsets.get(1).longValue();

        System.out.println("Docids shape: " + length);
        System.out.println("Data offsets: " + begin + " to " + end);

        ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        buffer.position((int) begin);
        String[] docids = new String[length];
        byte[] stringBytes = new byte[4];
        for (int i = 0; i < length; i++) {
            buffer.get(stringBytes);
            docids[i] = new String(stringBytes).trim();
        }

        // Log the first few docids to verify the content
        System.out.println("First few docids:");
        for (int i = 0; i < Math.min(10, docids.length); i++) {
            System.out.print(docids[i] + " ");
        }
        System.out.println();

        return docids;
    }
}
