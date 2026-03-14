/// ModelService for loading and running inference with TensorFlow Lite model.
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:typed_data';

class ModelService {
  late Interpreter interpreter;

  Future<void> loadModel(String modelPath) async {
    interpreter = await Interpreter.fromAsset(modelPath);
  }

  List<double> runInference(Uint8List input) {
    var output = List.filled(2, 0.0).reshape([1, 2]);
    interpreter.run(input, output);
    return output[0];
  }
}
