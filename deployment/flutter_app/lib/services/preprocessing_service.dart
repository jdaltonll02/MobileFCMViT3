/// PreprocessingService for image preprocessing before inference.
import 'dart:typed_data';
import 'package:image/image.dart' as img;

class PreprocessingService {
  Uint8List preprocess(Uint8List imageBytes) {
    img.Image image = img.decodeImage(imageBytes)!;
    img.Image resized = img.copyResize(image, width: 224, height: 224);
    // Add normalization, CLAHE, denoise, etc. as needed
    return Uint8List.fromList(img.encodePng(resized));
  }
}
