/// Utility functions for image handling.
import 'dart:typed_data';
import 'package:image/image.dart' as img;

class ImageUtils {
  static Uint8List imageToBytes(img.Image image) {
    return Uint8List.fromList(img.encodePng(image));
  }
}
