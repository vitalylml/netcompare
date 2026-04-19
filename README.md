# Netcompare

Netcompare is an experimental viewer for neural network, deep learning, and machine learning models, built to compare two models side by side. It highlights structural differences using intuitive, human-readable color categories:

- **Matched nodes:** Light Gray
- **Modified nodes:** Light Yellow
- **Added nodes:** Light Green
- **Removed nodes:** Light Red

Netcompare requires that both models use the same framework (e.g., ONNX vs ONNX, Keras vs Keras) and that one model is a modification or evolution of the other. Comparing unrelated architectures is not supported.
