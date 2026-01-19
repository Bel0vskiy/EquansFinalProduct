import tkinter as tk
import subprocess
import sys

def launch_table_ui():
     subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/app.py"])

def launch_gnn(file):
     subprocess.Popen([sys.executable, "gnn command", f'{file}'])

root = tk.Tk()
root.title("Streamlit App")

btn = tk.Button(root, text="Launch", command=launch_table_ui)
btn.pack()
root.mainloop()