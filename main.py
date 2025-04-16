import tkinter as tk

from gui.main_window import MainWindow


def main():
    root = tk.Tk()
    root.geometry("1920x1080")
    # root.attributes('-fullscreen', True)
    root.state("zoomed")
    root.title("Laser Microscope Control")
    app = MainWindow(master=root)
    app.mainloop()


if __name__ == "__main__":
    main()
