"""
models package
--------------
This package will hold the DATA / ENTITY definitions used across the
application — for example, a `DetectionResult` data class or an
ORM-mapped `User` / `Report` model, depending on the database choice
made later in the project.

Per current project scope, NO database ORM models, schemas, or
AI model definitions have been implemented here. This package only
establishes the folder boundary and import path (`from models import ...`)
for future use, in line with Clean Architecture's separation of the
data/entity layer from business logic (services/) and delivery
mechanisms (app.py, templates/).
"""
