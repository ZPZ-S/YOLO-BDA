graph TD
    A["Input Features"] --> B["Shared Convolution Layers"]
    B --> C1["Task Decomposition - Classification"]
    B --> C2["Task Decomposition - Regression"]
    C1 --> D1["Classification Probability Generation"]
    C2 --> D2["Dynamic Convolution Alignment - Regression"]
    D1 --> E["Concatenate Outputs"]
    D2 --> E
    E --> F1["Training: Output Feature Maps"]
    E --> F2["Inference: Decode Bounding Boxes and Probabilities"]



