import logging
import os
import tkinter as tk
from datetime import datetime

from gui.main_window import MainWindow


def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S.log"))

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),  # also show logs in console
        ],
    )


def main():
    setup_logging()
    root = tk.Tk()
    root.geometry("1920x1080")
    # root.attributes('-fullscreen', True)
    root.state("zoomed")
    root.title("Laser Microscope Control")
    root.iconbitmap("ikonka.ico")
    app = MainWindow(master=root)
    app.mainloop()


if __name__ == "__main__":
    main()
