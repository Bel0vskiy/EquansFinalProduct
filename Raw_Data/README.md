# Data Pipeline (Until CSV Output)

This pipeline converts Revit models to IFC and extracts rooms with assigned devices (lamps and sockets).
The final result is a single CSV table with room geometry and device coordinates.

HOW TO USE (for the next team):

1) Convert Revit (.rvt) to IFC  
Put Revit files into:
rvt_files/ProjectName/file.rvt
I advice you to rename two rvt files with ARCH and MEP to proper parsing the information.
(To determine what is it use online autodesk viewer)

Run:
```bash
python convert_menu.py
```
Result:
ifc_files/ProjectName/file.ifc

2) Prepare IFC files  
Each project folder must contain two IFC files:
- architecture IFC (filename contains "arch")
- MEP IFC (filename contains "mep")

Example:
ifc_files/
  ProjectName/
    arch.ifc
    mep.ifc

3) Extract rooms and devices (FINAL STEP)  
Open run_extract.py and set:
ifc_folder = "ifc_files"

Run:
python run_extract.py

FINAL OUTPUT:
- room_list_all.json
- room_list_all.csv

The CSV file contains room geometry, device type, and device coordinates.
This CSV is the handover point for analytics, ML, or visualization.
