"""
MACONFLIC - Village Polygon Annotator
======================================
Interactive tool for drawing irregular polygons around settlement
landscape regions in satellite screenshots.

HOW TO USE:
    python scripts/annotate_village_polygons.py --village kodanur
    python scripts/annotate_village_polygons.py --village pudur
    python scripts/annotate_village_polygons.py --village panapalli
    python scripts/annotate_village_polygons.py --village kandivazhi
    python scripts/annotate_village_polygons.py --village sembukarai

CONTROLS (in the annotation window):
    Left-click          -- Add a vertex to the polygon
    Right-click         -- Remove the last vertex (undo)
    Scroll wheel        -- Zoom in / out around cursor
    Middle-click drag   -- Pan the image
    ENTER / SPACE       -- Close the polygon and save
    P                   -- Preview the cutout
    R                   -- Reset (clear all vertices)
    Q / ESC             -- Quit without saving
    F                   -- Fit entire image to window

OUTPUT:
    data/village_polygon_cutouts/annotations/<village>_polygon.json

ANNOTATION FORMAT:
    {
        "village": "kodanur",
        "source_image": "kodanur.png",
        "image_size": [width, height],
        "points": [[x1,y1], [x2,y2], ...],
        "annotated_at": "ISO-8601 timestamp",
        "annotation_method": "manual_polygon"
    }
"""

import os
import sys
import json
import math
import argparse
from datetime import datetime

try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError:
    print("[ERROR] tkinter is not available. Install tk support for Python.")
    sys.exit(1)

try:
    from PIL import Image, ImageTk, ImageDraw
except ImportError:
    print("[ERROR] Pillow is not installed. Run: pip install Pillow")
    sys.exit(1)

# ---- Paths ---------------------------------------------------------------
REPO_ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_DIR      = os.path.join(REPO_ROOT, "data", "Seven Villages")
ANNOTATION_DIR = os.path.join(REPO_ROOT, "data", "village_polygon_cutouts", "annotations")
os.makedirs(ANNOTATION_DIR, exist_ok=True)

VILLAGE_MAP = {
    "kodanur":    "kodanur.png",
    "pudur":      "pudur.png",
    "panapalli":  "panapalli.png",
    "kandivazhi": "kandivazhi.png",
    "sembukarai": "sembukarai.png",
    # add when screenshots arrive:
    # "dhoomanur":   "dhoomanur.png",
    # "thuvaipathy":  "thuvaipathy.png",
}

DISPLAY_MAX = 1200   # max display dimension (px)
DOT_RADIUS  = 5
LINE_COLOR  = "#00FF00"
DOT_COLOR   = "#FFFF00"
CLOSE_COLOR = "#FF4444"


class AnnotatorApp:
    def __init__(self, root: tk.Tk, village: str, img_path: str, ann_path: str):
        self.root     = root
        self.village  = village
        self.img_path = img_path
        self.ann_path = ann_path

        self.orig_img  = Image.open(img_path).convert("RGB")
        self.orig_w, self.orig_h = self.orig_img.size

        # Determine initial scale so image fits in display
        scale_x = DISPLAY_MAX / self.orig_w
        scale_y = DISPLAY_MAX / self.orig_h
        self.scale  = min(1.0, scale_x, scale_y)
        self.offset = [0, 0]   # canvas offset for pan

        # Polygon points in ORIGINAL image coordinates
        self.points = []

        # Pan tracking
        self._pan_start = None
        self._pan_offset_start = None

        self._build_ui()
        self._load_existing_annotation()
        self._redraw()

    # ---- UI ----------------------------------------------------------------
    def _build_ui(self):
        self.root.title(f"MACONFLIC Annotator -- {self.village}")
        self.root.resizable(True, True)

        # Toolbar
        tb = tk.Frame(self.root, bg="#1a1a2e", pady=4)
        tb.pack(side=tk.TOP, fill=tk.X)

        btn_cfg = dict(bg="#16213e", fg="white", relief=tk.FLAT, padx=8, pady=4, font=("Consolas", 10))
        tk.Label(tb, text=f"  {self.village.upper()}  |  Left-click=add vertex  |  Right-click=undo  |  Scroll=zoom  |  Middle-drag=pan  |  ENTER=save  |  R=reset  |  P=preview  |  Q=quit",
                 bg="#1a1a2e", fg="#94a3b8", font=("Consolas", 9)).pack(side=tk.LEFT, padx=8)

        self.status_var = tk.StringVar(value="0 vertices")
        tk.Label(tb, textvariable=self.status_var, bg="#1a1a2e", fg="#22c55e",
                 font=("Consolas", 10, "bold")).pack(side=tk.RIGHT, padx=12)

        # Canvas
        disp = int(self.orig_w * self.scale)
        disp_h = int(self.orig_h * self.scale)
        self.canvas = tk.Canvas(self.root, width=min(disp, DISPLAY_MAX),
                                height=min(disp_h, DISPLAY_MAX),
                                bg="black", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bindings
        self.canvas.bind("<Button-1>",        self._on_left_click)
        self.canvas.bind("<Button-3>",        self._on_right_click)
        self.canvas.bind("<Button-2>",        self._on_pan_start)
        self.canvas.bind("<B2-Motion>",       self._on_pan_drag)
        self.canvas.bind("<MouseWheel>",      self._on_scroll)      # Windows
        self.canvas.bind("<Button-4>",        self._on_scroll)      # Linux scroll up
        self.canvas.bind("<Button-5>",        self._on_scroll)      # Linux scroll down
        self.root.bind("<Return>",            lambda e: self._save())
        self.root.bind("<space>",             lambda e: self._save())
        self.root.bind("r",                   lambda e: self._reset())
        self.root.bind("R",                   lambda e: self._reset())
        self.root.bind("p",                   lambda e: self._preview())
        self.root.bind("P",                   lambda e: self._preview())
        self.root.bind("f",                   lambda e: self._fit())
        self.root.bind("F",                   lambda e: self._fit())
        self.root.bind("q",                   lambda e: self._quit())
        self.root.bind("<Escape>",            lambda e: self._quit())

    # ---- Coordinate helpers ------------------------------------------------
    def _canvas_to_orig(self, cx, cy):
        """Convert canvas pixel -> original image pixel."""
        ox = (cx - self.offset[0]) / self.scale
        oy = (cy - self.offset[1]) / self.scale
        return ox, oy

    def _orig_to_canvas(self, ox, oy):
        """Convert original image pixel -> canvas pixel."""
        cx = ox * self.scale + self.offset[0]
        cy = oy * self.scale + self.offset[1]
        return cx, cy

    # ---- Drawing -----------------------------------------------------------
    def _redraw(self):
        self.canvas.delete("all")

        # Scaled image
        dw = int(self.orig_w * self.scale)
        dh = int(self.orig_h * self.scale)
        dw, dh = max(dw, 1), max(dh, 1)
        disp_img = self.orig_img.resize((dw, dh), Image.Resampling.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(disp_img)
        self.canvas.create_image(self.offset[0], self.offset[1],
                                 anchor=tk.NW, image=self._tk_img)

        if len(self.points) < 1:
            self.status_var.set("0 vertices -- left-click to add")
            return

        # Draw lines between points
        canvas_pts = [self._orig_to_canvas(px, py) for px, py in self.points]
        if len(canvas_pts) >= 2:
            for i in range(len(canvas_pts) - 1):
                self.canvas.create_line(canvas_pts[i][0], canvas_pts[i][1],
                                        canvas_pts[i+1][0], canvas_pts[i+1][1],
                                        fill=LINE_COLOR, width=2)

        # Closing line preview
        if len(canvas_pts) >= 3:
            last = canvas_pts[-1]
            first = canvas_pts[0]
            self.canvas.create_line(last[0], last[1], first[0], first[1],
                                    fill=CLOSE_COLOR, width=1, dash=(4, 4))

        # Draw dots
        for i, (cx, cy) in enumerate(canvas_pts):
            color = CLOSE_COLOR if i == 0 else DOT_COLOR
            r = DOT_RADIUS
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                    fill=color, outline="white", width=1)

        n = len(self.points)
        self.status_var.set(f"{n} vertices  |  ENTER to save  |  R to reset")

    # ---- Event handlers ----------------------------------------------------
    def _on_left_click(self, event):
        ox, oy = self._canvas_to_orig(event.x, event.y)
        ox = max(0.0, min(float(self.orig_w - 1), ox))
        oy = max(0.0, min(float(self.orig_h - 1), oy))
        self.points.append([round(ox, 1), round(oy, 1)])
        self._redraw()

    def _on_right_click(self, event):
        if self.points:
            self.points.pop()
            self._redraw()

    def _on_pan_start(self, event):
        self._pan_start = (event.x, event.y)
        self._pan_offset_start = list(self.offset)

    def _on_pan_drag(self, event):
        if self._pan_start:
            dx = event.x - self._pan_start[0]
            dy = event.y - self._pan_start[1]
            self.offset = [self._pan_offset_start[0] + dx,
                           self._pan_offset_start[1] + dy]
            self._redraw()

    def _on_scroll(self, event):
        # Determine zoom direction
        if hasattr(event, "delta"):
            direction = 1 if event.delta > 0 else -1
        elif event.num == 4:
            direction = 1
        else:
            direction = -1

        factor = 1.15 if direction > 0 else (1.0 / 1.15)
        new_scale = self.scale * factor
        new_scale = max(0.05, min(10.0, new_scale))

        # Zoom toward cursor
        mx, my = event.x, event.y
        # orig point under cursor stays fixed
        ox = (mx - self.offset[0]) / self.scale
        oy = (my - self.offset[1]) / self.scale
        self.scale = new_scale
        self.offset[0] = mx - ox * self.scale
        self.offset[1] = my - oy * self.scale
        self._redraw()

    # ---- Actions -----------------------------------------------------------
    def _reset(self):
        if messagebox.askyesno("Reset", "Clear all vertices?"):
            self.points = []
            self._redraw()

    def _fit(self):
        """Fit the full image to the canvas."""
        cw = self.canvas.winfo_width() or DISPLAY_MAX
        ch = self.canvas.winfo_height() or DISPLAY_MAX
        sx = cw / self.orig_w
        sy = ch / self.orig_h
        self.scale = min(sx, sy)
        self.offset = [0, 0]
        self._redraw()

    def _preview(self):
        if len(self.points) < 3:
            messagebox.showinfo("Preview", "Need at least 3 vertices for a preview.")
            return
        # Generate preview image
        cutout = _make_cutout(self.orig_img, self.points)
        # Composite over dark checkerboard background
        checker = _make_checker(cutout.width, cutout.height, 20)
        checker.paste(cutout, (0, 0), cutout)
        # Resize for display
        pw = min(800, cutout.width)
        ph = int(cutout.height * (pw / cutout.width))
        preview_disp = checker.resize((pw, ph), Image.Resampling.LANCZOS)

        win = tk.Toplevel(self.root)
        win.title("Cutout Preview")
        tk_img = ImageTk.PhotoImage(preview_disp)
        lbl = tk.Label(win, image=tk_img, bg="black")
        lbl.image = tk_img
        lbl.pack()

    def _save(self):
        if len(self.points) < 3:
            messagebox.showwarning("Save", "Need at least 3 vertices to save a polygon.")
            return
        data = {
            "village": self.village,
            "source_image": os.path.basename(self.img_path),
            "image_size": [self.orig_w, self.orig_h],
            "points": self.points,
            "annotated_at": datetime.now().isoformat(timespec="seconds"),
            "annotation_method": "manual_polygon"
        }
        with open(self.ann_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[OK] Annotation saved: {self.ann_path}")
        messagebox.showinfo("Saved", f"Polygon saved with {len(self.points)} vertices.\n{self.ann_path}")

    def _quit(self):
        if self.points:
            if not messagebox.askyesno("Quit", "Quit without saving?"):
                return
        self.root.destroy()

    def _load_existing_annotation(self):
        if os.path.exists(self.ann_path):
            try:
                with open(self.ann_path) as f:
                    data = json.load(f)
                self.points = data.get("points", [])
                print(f"[OK] Loaded existing annotation: {len(self.points)} vertices")
            except Exception as e:
                print(f"[WARN] Could not load existing annotation: {e}")


# ---- Cutout helpers -------------------------------------------------------

def _make_cutout(orig_img: Image.Image, points: list) -> Image.Image:
    """Return RGBA image with polygon region visible, rest transparent."""
    mask = Image.new("L", orig_img.size, 0)
    draw = ImageDraw.Draw(mask)
    flat_pts = [(int(x), int(y)) for x, y in points]
    draw.polygon(flat_pts, fill=255)
    rgba = orig_img.convert("RGBA")
    rgba.putalpha(mask)
    return rgba


def _make_checker(w: int, h: int, cell: int = 20) -> Image.Image:
    """Generate a dark checkerboard background for transparency preview."""
    img = Image.new("RGB", (w, h), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            if ((x // cell) + (y // cell)) % 2 == 0:
                draw.rectangle([x, y, x + cell - 1, y + cell - 1], fill=(50, 50, 50))
    return img


# ---- Entry point ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MACONFLIC Village Polygon Annotator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--village", required=True,
                        choices=list(VILLAGE_MAP.keys()),
                        help="Village key to annotate.")
    args = parser.parse_args()

    village    = args.village
    img_file   = VILLAGE_MAP[village]
    img_path   = os.path.join(INPUT_DIR, img_file)
    ann_path   = os.path.join(ANNOTATION_DIR, f"{village}_polygon.json")

    if not os.path.exists(img_path):
        print(f"[ERROR] Source image not found: {img_path}")
        sys.exit(1)

    root = tk.Tk()
    app  = AnnotatorApp(root, village, img_path, ann_path)
    root.mainloop()


if __name__ == "__main__":
    main()
