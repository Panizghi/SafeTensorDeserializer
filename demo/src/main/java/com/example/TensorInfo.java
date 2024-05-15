package com.example;

import java.util.ArrayList;
import java.util.List;

public class TensorInfo {
    public String dtype;
    public List<Integer> shape = new ArrayList<>();
    public int[] dataOffsets = new int[2];
}
