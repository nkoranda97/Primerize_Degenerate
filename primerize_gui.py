from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QPushButton,
    QLabel,
    QLineEdit,
    QGroupBox,
    QMessageBox,
)
from PySide6.QtCore import Qt
from primerize.primerize_1d import Primerize_1D
import sys


class PrimerizeGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = Primerize_1D()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Primerize Designer")
        self.setGeometry(100, 100, 1000, 600)  # Reduced height since we removed console

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create sequence input section
        sequence_group = QGroupBox("Sequence Input")
        sequence_layout = QVBoxLayout()
        self.sequence_text = QTextEdit()
        self.sequence_text.setPlaceholderText("Enter your sequence here...")
        self.sequence_text.setMinimumHeight(100)
        design_button = QPushButton("Design Primers")
        design_button.clicked.connect(self.run_design)
        sequence_layout.addWidget(self.sequence_text)
        sequence_layout.addWidget(design_button)
        sequence_group.setLayout(sequence_layout)

        # Create parameters section
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout()
        self.params = {
            "Minimum Annealing Temperature": (
                "Minimum annealing temperature (°C)",
                QLineEdit(),
                "60",
                "MIN_TM",
            ),
            "Minimum Primer Length": (
                "Minimum primer length",
                QLineEdit(),
                "15",
                "MIN_LENGTH",
            ),
            "Maximum Primer Length": (
                "Maximum primer length",
                QLineEdit(),
                "60",
                "MAX_LENGTH",
            ),
        }

        for display_name, (tooltip, widget, default, _) in self.params.items():
            param_layout = QHBoxLayout()
            label = QLabel(display_name)
            label.setToolTip(tooltip)
            widget.setToolTip(tooltip)
            widget.setPlaceholderText(default)
            param_layout.addWidget(label)
            param_layout.addWidget(widget)
            params_layout.addLayout(param_layout)
        params_group.setLayout(params_layout)

        # Create results section
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)

        # Add all sections to main layout
        top_layout = QHBoxLayout()
        left_column = QVBoxLayout()
        right_column = QVBoxLayout()

        left_column.addWidget(sequence_group)
        left_column.addWidget(params_group)
        right_column.addWidget(results_group)

        top_layout.addLayout(left_column)
        top_layout.addLayout(right_column)

        main_layout.addLayout(top_layout)

    def show_error(self, message, suggestions=None):
        error_box = QMessageBox(self)
        error_box.setIcon(QMessageBox.Icon.Warning)
        error_box.setWindowTitle("Design Failed")

        if suggestions:
            error_box.setText(f"{message}\n\nSuggestions:")
            error_box.setInformativeText(suggestions)
        else:
            error_box.setText(message)

        error_box.exec()

    def run_design(self):
        # Clear previous results
        self.results_text.clear()

        # Set parameters
        for display_name, (_, widget, default, param_name) in self.params.items():
            value = widget.text() or default  # Use default if field is empty
            if value:  # Only set if value is provided
                try:
                    value = float(value) if param_name == "MIN_TM" else int(value)
                    self.worker.set(param_name, value)
                except ValueError:
                    self.show_error(f"Invalid value for {display_name}")
                    return

        # Get sequence and run design
        sequence = self.sequence_text.toPlainText().strip()
        if not sequence:
            self.show_error("Please enter a sequence")
            return

        try:
            job = self.worker.design(sequence=sequence)

            # Display results
            if len(job.primer_set) > 0:
                results = []
                for i, primer in enumerate(job.primer_set, start=1):
                    results.append(f">Primer_{i}")
                    results.append(f"{primer}")
                self.results_text.setText("\n".join(results))
            else:
                suggestions = (
                    "In order, try the following:\n"
                    "• Try increasing the Maximum Primer Length parameter\n"
                    "• Try decreasing the Minimum Primer Length parameter\n"
                    "• Try adjusting the Minimum Annealing Temperature parameter\n"
                    "• Check if your sequence is valid"
                )
                self.show_error(
                    "No solutions found under given constraints", suggestions
                )

        except Exception as e:
            self.show_error(str(e))


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    gui = PrimerizeGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
