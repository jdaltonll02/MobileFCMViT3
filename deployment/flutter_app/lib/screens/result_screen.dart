/// ResultScreen for displaying prediction probabilities and Grad-CAM heatmap.
import 'package:flutter/material.dart';

class ResultScreen extends StatelessWidget {
  final List<double> probabilities;
  final String gradcamPath;

  const ResultScreen({required this.probabilities, required this.gradcamPath, Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Results')),
      body: Column(
        children: [
          Text('Prediction Probabilities:'),
          Text('Benign: ${probabilities[0].toStringAsFixed(2)}'),
          Text('Malignant: ${probabilities[1].toStringAsFixed(2)}'),
          SizedBox(height: 20),
          Text('Grad-CAM Heatmap:'),
          Image.asset(gradcamPath),
        ],
      ),
    );
  }
}
