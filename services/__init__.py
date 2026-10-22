"""
services package
-----------------
This package holds the BUSINESS LOGIC layer of the application.

In Clean Architecture, "services" sit between the routes (controllers)
and the models/database layer. They orchestrate *what* the application
does, independent of *how* it is exposed (Flask routes) or *how* data
is stored (database layer).

Planned modules (NOT implemented yet, per project scope):
    - detection_service.py : Will orchestrate calls to the multimodal
                              deepfake detection pipeline (audio, video,
                              image analysis) once that logic is added.
    - report_service.py    : Will handle generation of result reports.

Currently this package only contains placeholder scaffolding so that
routes in app.py have a clean, documented place to import business
logic from in the next development phase.
"""
