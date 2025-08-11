import tkinter as tk
from tkinter import ttk
from test_results_tab import TestResultsTab
from ground_truths_tab import GroundTruthsTab


class TestResultsVisualizer:
    def __init__(self):
        # Create main window
        self.root = tk.Tk()
        self.root.title("Test Results Visualizer")
        self.root.geometry("800x600")
        
        # Create notebook (tab container)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Initialize tabs
        self.test_results_tab = TestResultsTab(self.notebook)
        self.ground_truths_tab = GroundTruthsTab(self.notebook)
    
    def run(self):
        """Start the application"""
        self.root.mainloop()


if __name__ == "__main__":
    app = TestResultsVisualizer()
    app.run()