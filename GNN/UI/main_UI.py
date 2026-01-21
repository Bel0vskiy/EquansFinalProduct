import sys
import traceback
from PySide6.QtWidgets import QApplication, QMessageBox

from viewer.main_window import AdvancedRoomViewer

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Advanced 3D Room Viewer")
    app.setApplicationVersion("1.0")

    window = AdvancedRoomViewer()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
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