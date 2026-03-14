/// Widget for displaying Grad-CAM heatmap.
import 'package:flutter/material.dart';

class GradCAMWidget extends StatelessWidget {
  final String gradcamPath;

  const GradCAMWidget({required this.gradcamPath, Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Image.asset(gradcamPath);
  }
}
