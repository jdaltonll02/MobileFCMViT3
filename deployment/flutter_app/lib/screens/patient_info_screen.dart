/// PatientInfoScreen for displaying and editing patient information.
import 'package:flutter/material.dart';

class PatientInfoScreen extends StatelessWidget {
  final Map<String, String> patientInfo;

  const PatientInfoScreen({required this.patientInfo, Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Patient Information')),
      body: ListView(
        children: patientInfo.entries.map((entry) => ListTile(
          title: Text(entry.key),
          subtitle: Text(entry.value),
        )).toList(),
      ),
    );
  }
}
