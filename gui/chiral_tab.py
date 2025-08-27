import logging
import os
import threading
import tkinter as tk
from tkinter import (
    Button,
    Canvas,
    Entry,
    Frame,
    Label,
    OptionMenu,
    Scrollbar,
    StringVar,
    filedialog,
    messagebox,
)

import numpy as np  # only used to coerce numpy arrays to PIL for display
from PIL import Image, ImageTk

import utils.consts as consts
from controls.chiral_control import ChiralControl
from utils.utils import thread_execute

logger = logging.getLogger(__name__)


class ChiralTab:
    # Fixed canvas size and per-source-pixel screen size.
    # With PIXEL_SCALE=8 and CANVAS_SIZE=640, an 80x80 image fits exactly;
    # a 2x2 becomes a small 16x16 block centered on gray.
    CANVAS_SIZE = 640
    PIXEL_SCALE = 8
    CANVAS_BG = "#4a4a4a"  # mid gray
    OVERLAY_COLOR = "#ff4d4d"  # red X for selected pixel

    def __init__(self, parent, control: ChiralControl):
        self.control = control
        self.frame = Frame(parent)
        self.frame.pack(fill=tk.Y)

        # Title
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        Label(cur_frame, text="CHIRAL TAB", font=consts.subsystem_name_font).pack(side=tk.LEFT)

        # Controls row
        controls_row = Frame(self.frame)
        controls_row.pack(fill=tk.X, pady=6)

        self.btn_load = Button(controls_row, text="Load pattern…", command=self._on_load_clicked)
        self.btn_load.pack(side=tk.LEFT, padx=4)

        # Pixel selection controls
        px_row = Frame(self.frame)
        px_row.pack(fill=tk.X, pady=(0, 6))

        Label(px_row, text="X:").pack(side=tk.LEFT, padx=(0, 4))
        self.selected_x = StringVar(value="0")
        self.entry_x = Entry(px_row, width=6, textvariable=self.selected_x)
        self.entry_x.pack(side=tk.LEFT, padx=(0, 8))
        self.entry_x.bind("<Return>", lambda e: self.render_pattern())

        Label(px_row, text="Y:").pack(side=tk.LEFT, padx=(0, 4))
        self.selected_y = StringVar(value="0")
        self.entry_y = Entry(px_row, width=6, textvariable=self.selected_y)
        self.entry_y.pack(side=tk.LEFT, padx=(0, 12))
        self.entry_y.bind("<Return>", lambda e: self.render_pattern())

        # Placeholder parameters (do nothing yet)
        # Feel free to rename later
        Label(px_row, text="Param A:").pack(side=tk.LEFT, padx=(0, 4))
        self.param_a = StringVar(value="")
        Entry(px_row, width=10, textvariable=self.param_a).pack(side=tk.LEFT, padx=(0, 8))

        Label(px_row, text="Param B:").pack(side=tk.LEFT, padx=(0, 4))
        self.param_b = StringVar(value="")
        Entry(px_row, width=10, textvariable=self.param_b).pack(side=tk.LEFT, padx=(0, 8))

        Label(px_row, text="Param C:").pack(side=tk.LEFT, padx=(0, 4))
        self.param_c = StringVar(value="")
        Entry(px_row, width=10, textvariable=self.param_c).pack(side=tk.LEFT, padx=(0, 8))

        # Script selection row
        script_row = Frame(self.frame)
        script_row.pack(fill=tk.X, pady=(0, 6))

        self.btn_select_script = Button(script_row, text="Select script…", command=self._on_select_script)
        self.btn_select_script.pack(side=tk.LEFT, padx=(0, 8))

        self.script_label_var = StringVar(value="(no script)")
        Label(script_row, textvariable=self.script_label_var).pack(side=tk.LEFT)

        # Start melting action
        actions_row = Frame(self.frame)
        actions_row.pack(fill=tk.X, pady=(0, 8))

        self.btn_start = Button(actions_row, text="START_MELTING", command=self._on_start_melting)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 8))

        # Canvas for image preview
        canvas_row = Frame(self.frame)
        canvas_row.pack(fill=tk.BOTH, expand=True)

        self.canvas = Canvas(
            canvas_row,
            width=self.CANVAS_SIZE,
            height=self.CANVAS_SIZE,
            bg=self.CANVAS_BG,
            highlightthickness=0,
        )
        self.canvas.pack(padx=6, pady=6)

        # Keep a reference to the PhotoImage to avoid garbage collection
        self._canvas_image_tk = None

        # Store rendering context (scale/offset/size) for overlay
        self._render_ctx = {
            "scale": self.PIXEL_SCALE,
            "off_x": 0,
            "off_y": 0,
            "img_w": 0,
            "img_h": 0,
        }

        # Initial placeholder
        self._draw_placeholder()

    # -------------------------
    # UI Callbacks
    # -------------------------

    def _on_load_clicked(self):
        """
        Ask the control to load the pattern image (control handles all logic/UI for that),
        then re-render canvas from control.pattern_image.
        """
        try:
            self.control.load_pattern_image()
            # Refresh preview
            self.render_pattern()
        except Exception as e:
            logger.exception("Failed to load pattern image.")
            messagebox.showerror("Load error", f"Could not load pattern image:\n{e}")

    def _on_select_script(self):
        """
        Pick a melting script file and store the path in control.melting_script_path.
        """
        try:
            path = filedialog.askopenfilename(
                title="Select melting script",
                filetypes=[("Microscope scripts", "*.scrpt"), ("All files", "*.*")],
                initialdir=os.getcwd(),
                parent=getattr(self.control.master, "root", None),
            )
            if not path:
                return
            # store in control
            self.control.melting_script_path = path
            self.script_label_var.set(os.path.basename(path))
            logger.info("Selected melting script: %s", path)
        except Exception as e:
            logger.exception("Failed to select script.")
            messagebox.showerror("Script selection error", f"Could not select script:\n{e}")

    def _on_start_melting(self):
        """
        Call control.start_melting(...) with parameters gathered from GUI.
        Runs in a background thread via thread_execute.
        """
        # Parse X/Y as ints; report validation errors
        try:
            x = int(self.selected_x.get())
            y = int(self.selected_y.get())
        except ValueError:
            messagebox.showerror("Invalid coordinates", "X and Y must be integers.")
            return

        # Optional: warn if no script chosen
        if not getattr(self.control, "melting_script_path", None):
            if not messagebox.askyesno(
                "No script selected",
                "No melting script selected. Continue anyway?",
            ):
                return

        # Collect placeholder params (strings for now)
        pA = self.param_a.get()
        pB = self.param_b.get()
        pC = self.param_c.get()

        # Kick off control logic in a thread (non-blocking UI)
        self.control.start_melting(x, y, pA, pB, pC)

    # -------------------------
    # Rendering
    # -------------------------

    def render_pattern(self):
        """
        Render the control.pattern_image on the fixed-size canvas, using a fixed PIXEL_SCALE.
        The image is scaled with NEAREST (hard edges) and centered on a gray background.
        Also draws an 'X' overlay over the selected pixel, if in bounds.
        """
        self.canvas.delete("all")
        pil_img = self._coerce_to_pil(getattr(self.control, "pattern_image", None))

        if pil_img is None:
            self._draw_placeholder()
            return

        # Ensure 3-channel or L; keep as-is otherwise
        if pil_img.mode not in ("L", "RGB"):
            try:
                pil_img = pil_img.convert("RGB")
            except Exception:
                pil_img = pil_img.convert("L")

        w, h = pil_img.size
        if w <= 0 or h <= 0:
            self._draw_placeholder("Invalid image size")
            return

        # Scale each source pixel to a fixed on-screen size
        scaled_w = w * self.PIXEL_SCALE
        scaled_h = h * self.PIXEL_SCALE

        # Safety clamp: if something larger than expected appears, fit it but keep hard edges
        max_size = self.CANVAS_SIZE
        scale_used = self.PIXEL_SCALE
        if scaled_w > max_size or scaled_h > max_size:
            scale = min(max_size / scaled_w, max_size / scaled_h)
            int_scale = max(1, int(round(scale * self.PIXEL_SCALE)))
            scaled_w = w * int_scale
            scaled_h = h * int_scale
            scale_used = int_scale

        scaled = pil_img.resize((scaled_w, scaled_h), resample=Image.NEAREST)

        # Compose onto gray background and center it
        bg = Image.new("RGB", (self.CANVAS_SIZE, self.CANVAS_SIZE), self._hex_to_rgb(self.CANVAS_BG))
        off_x = (self.CANVAS_SIZE - scaled_w) // 2
        off_y = (self.CANVAS_SIZE - scaled_h) // 2
        bg.paste(scaled, (off_x, off_y))

        # Draw on canvas
        self._canvas_image_tk = ImageTk.PhotoImage(bg)
        self.canvas.create_image(0, 0, anchor="nw", image=self._canvas_image_tk)

        # Save render context for overlays
        self._render_ctx.update(
            {
                "scale": scale_used,
                "off_x": off_x,
                "off_y": off_y,
                "img_w": w,
                "img_h": h,
            }
        )

        # Overlay the selected pixel (big X)
        self._draw_selected_pixel_overlay()

    def _draw_selected_pixel_overlay(self):
        """
        Draw a red 'X' over the selected pixel, aligned to scaled pixel grid and centering.
        """
        ctx = self._render_ctx
        scale = ctx["scale"]
        off_x = ctx["off_x"]
        off_y = ctx["off_y"]
        img_w = ctx["img_w"]
        img_h = ctx["img_h"]

        # Parse selection
        try:
            sx = int(self.selected_x.get())
            sy = int(self.selected_y.get())
        except ValueError:
            return  # nothing to draw

        # In-bounds?
        if not (0 <= sx < img_w and 0 <= sy < img_h):
            return

        # Compute pixel box on canvas
        left = off_x + sx * scale
        top = off_y + sy * scale
        right = left + scale
        bottom = top + scale

        # Draw an "X" across that pixel block
        # Slight padding to keep the X inside the block
        pad = max(1, scale // 10)
        self.canvas.create_line(
            left + pad,
            top + pad,
            right - pad,
            bottom - pad,
            fill=self.OVERLAY_COLOR,
            width=max(2, scale // 6),
        )
        self.canvas.create_line(
            left + pad,
            bottom - pad,
            right - pad,
            top + pad,
            fill=self.OVERLAY_COLOR,
            width=max(2, scale // 6),
        )

    def _draw_placeholder(self, msg: str | None = None):
        """Draw a simple 'no image' placeholder."""
        self.canvas.delete("all")
        self.canvas.configure(bg=self.CANVAS_BG)
        text = "No pattern loaded" if msg is None else msg
        self.canvas.create_text(
            self.CANVAS_SIZE // 2,
            self.CANVAS_SIZE // 2,
            text=text,
            fill="#bbbbbb",
            font=("Segoe UI", 12),
        )

    # -------------------------
    # Helpers
    # -------------------------

    @staticmethod
    def _hex_to_rgb(hx: str) -> tuple[int, int, int]:
        hx = hx.lstrip("#")
        return tuple(int(hx[i : i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def _coerce_to_pil(obj) -> Image.Image | None:
        """Accepts PIL.Image.Image or numpy arrays. Returns PIL image or None."""
        if obj is None:
            return None
        if isinstance(obj, Image.Image):
            return obj

        try:
            import numpy as _np  # local import in case numpy is optional elsewhere

            if isinstance(obj, _np.ndarray):
                arr = obj
                if arr.ndim == 2:
                    return Image.fromarray(arr.astype(np.uint8), mode="L")
                if arr.ndim == 3:
                    # If 3 channels, assume already RGB-like
                    ch = arr.shape[2]
                    if ch == 3:
                        return Image.fromarray(arr.astype(np.uint8), mode="RGB")
                    if ch == 4:
                        return Image.fromarray(arr[:, :, :3].astype(np.uint8), mode="RGB")
        except Exception as e:
            logger.debug(f"Failed to coerce to PIL: {e}")

        # Fallback: give up gracefully
        return None
