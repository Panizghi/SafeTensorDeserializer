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
        String docidToIdxFilePath = "python/output/docid_to_idx.json";
        String outputJsonFilePath = "data.json";

        try {
            // Read and deserialize the SafeTensors files
            byte[] vectorsData = Files.readAllBytes(Paths.get(vectorsFilePath));
            byte[] docidsData = Files.readAllBytes(Paths.get(docidsFilePath));

            // Deserialize docid_to_idx.json
            ObjectMapper objectMapper = new ObjectMapper();
            Map<String, Integer> docidToIdx = objectMapper.readValue(Files.readAllBytes(Paths.get(docidToIdxFilePath)), Map.class);

            // Deserialize vectors
            Map<String, Object> vectorsHeader = parseHeader(vectorsData);
            double[][] vectors = extractVectors(vectorsData, vectorsHeader);

            // Deserialize docids
            Map<String, Object> docidsHeader = parseHeader(docidsData);
            int[] docids = extractDocids(docidsData, docidsHeader);

            // Prepare the output data structure
            Map<String, Object> outputData = new HashMap<>();
            outputData.put("vectors", vectors);
            outputData.put("docids", docids);
            outputData.put("docid_to_idx", docidToIdx);

            // Serialize the output data to JSON and save to file
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
        int[] shape = shapeList.stream().mapToInt(i -> i).toArray();
        List<Number> dataOffsets = (List<Number>) vectorsInfo.get("data_offsets");
        long begin = dataOffsets.get(0).longValue();
        long end = dataOffsets.get(1).longValue();

        System.out.println("Vectors shape: " + shape[0] + "x" + shape[1]);
        System.out.println("Data offsets: " + begin + " to " + end);

        ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        buffer.position((int) begin);

        double[][] vectors = new double[shape[0]][shape[1]];
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                vectors[i][j] = buffer.getDouble();
            }
        }

        System.out.println("First vector element before fix: " + vectors[0][0]);
        
        // Fix the first value of the first vector element if it's an invalid double
        if (vectors[0][0] < 1e-308 || vectors[0][0] > 1e308 || Double.isNaN(vectors[0][0]) || Double.isInfinite(vectors[0][0])) {
            buffer.position((int) begin + 8); // Skip the header size
            vectors[0][0] = buffer.getDouble();
        }

        System.out.println("First vector element after fix: " + vectors[0][0]);
        return vectors;
    }

    private static int[] extractDocids(byte[] data, Map<String, Object> header) {
        Map<String, Object> docidsInfo = (Map<String, Object>) header.get("docids");
        List<Integer> shapeList = (List<Integer>) docidsInfo.get("shape");
        int[] shape = shapeList.stream().mapToInt(i -> i).toArray();
        List<Number> dataOffsets = (List<Number>) docidsInfo.get("data_offsets");
        long begin = dataOffsets.get(0).longValue();
        long end = dataOffsets.get(1).longValue();

        System.out.println("Docids shape: " + shape[0]);
        System.out.println("Data offsets: " + begin + " to " + end);

        ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        buffer.position((int) begin);
        int[] docids = new int[shape[0]];
        for (int i = 0; i < shape[0]; i++) {
            if (buffer.position() < end) {
                docids[i] = buffer.getInt();
            } else {
                docids[i] = 0; // Default value if position is beyond end
            }
        }
        return docids;
    }
}
