# main.py

import sys
import traceback
from PySide6.QtWidgets import QApplication, QMessageBox

"""
Main file to run latest version of UI
"""

# Add GNN root to path to allow importing Model
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the main window class
from viewer.main_window import AdvancedRoomViewer

def main():
    """Main application entry point"""
    # Create the Qt Application
    app = QApplication(sys.argv)

    # Set application properties (optional but good practice)
    app.setApplicationName("Advanced 3D Room Viewer")
    app.setApplicationVersion("1.0")

    # Create and show the main window
    # You could pass a default data path here if desired
    # default_path = "./Data/rotterdam"
    window = AdvancedRoomViewer() # default_data_path=default_path
    window.show()

    # Start the Qt event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Basic top-level error handling
        print(f"CRITICAL ERROR: {e}")
        # Try to show a message box if possible
        try:
            critical_app = QApplication.instance()
            if critical_app is None:
                critical_app = QApplication([sys.executable] + sys.argv)
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Icon.Critical)
            msg_box.setWindowTitle("Fatal Error")
            msg_box.setText(f"Application failed to start or encountered a critical error:\n{e}")
            msg_box.setDetailedText(traceback.format_exc())
            msg_box.exec()
        except Exception as msg_e:
            print(f"Could not display graphical error message: {msg_e}")
        traceback.print_exc()
        sys.exit(1)