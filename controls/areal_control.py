import logging
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageOps

import utils.image_utils as imgut
from utils.consts import CamConsts
from utils.utils import get_save_file_dialog

if TYPE_CHECKING:
    from gui.main_window import MainWindow

logger = logging.getLogger(__name__)

MAX_MINIMAP_SIZE_MM = 25  # a little bit more than M30 limit size (was it 20?)
AREAL_MAP_SCRIPT = "areal_map"


class ArealControl:
    MINIMAP_SIZE = 400

    def __init__(self, master: "MainWindow"):
        self.master = master
        self.minimap = None
        self.overlay = None
        self.last_place = (-1, -1)  # position of stage in mm
        self.scale_factor = self.MINIMAP_SIZE / MAX_MINIMAP_SIZE_MM  # pixels in mm
        self.current_obj_for_mapping = ""
        self.amap_start_mm = (0.0, 0.0)
        self.amap_end_mm = (0.0, 0.0)
        self.amap_images = []
        self.map_name = "areal_map"

    def reset_minimap(self):
        self.current_obj_for_mapping = self.master.camera_panel.get_objective()
        self.minimap = np.zeros((self.MINIMAP_SIZE, self.MINIMAP_SIZE, 3), np.uint8)
        self.overlay = np.zeros(
            (self.MINIMAP_SIZE, self.MINIMAP_SIZE, 4), np.uint8
        )  # reset also overlay for safety

    def get_new_img_if_new_place(self) -> Image.Image | None:
        stage_status = self.master.stage_controller.get_status()
        cplace = (stage_status["x_pos"], stage_status["y_pos"])
        # TODO: maybe its better to only check the state and not position here?
        if stage_status.get("state", "") == "IDLE":
            self.last_place = cplace
            return self.master.camera_controller.get_image()
        else:
            return None

    def update_minimap(self):
        new_image = self.get_new_img_if_new_place()  # this also updates last_place

        # current position
        obj = self.master.camera_panel.get_objective()
        px_x = self.mm2px_minimap(self.last_place[0])
        px_y = self.mm2px_minimap(self.last_place[1])

        # what camera sees in mm
        view_w_mm = imgut.scale_px2mm(CamConsts.REAL_WIDTH, obj)
        view_h_mm = imgut.scale_px2mm(CamConsts.REAL_HEIGHT, obj)

        # how large it should be on the minimap (size in px)
        mmap_w_px = self.mm2px_minimap(view_w_mm)
        mmap_h_px = self.mm2px_minimap(view_h_mm)

        # update by new images
        if new_image is not None:
            rescale_factor = mmap_w_px / CamConsts.REAL_WIDTH  # this is how much we should shirnk the image
            img_scaled = ImageOps.scale(new_image, rescale_factor, resample=Image.Resampling.LANCZOS)
            img_arr_scaled = np.asarray(img_scaled)
            logger.debug(f"placing new map at = {(px_x, px_y)}")
            self.minimap = imgut.add_image(self.minimap, img_arr_scaled, (px_x, px_y))

        # add drawings: view
        self.overlay = np.zeros((self.MINIMAP_SIZE, self.MINIMAP_SIZE, 4), np.uint8)
        self.overlay = cv2.rectangle(
            self.overlay,
            (px_x, px_y),
            (px_x + mmap_w_px, px_y + mmap_h_px),
            (200, 45, 45, 222),
            thickness=1,
        )

        # add drawings: amap
        if self.amap_end_mm != self.amap_start_mm:
            amap_start_px = tuple(self.mm2px_minimap(val) for val in self.amap_start_mm)
            amap_end_px = tuple(self.mm2px_minimap(val) for val in self.amap_end_mm)
            self.overlay = cv2.rectangle(
                self.overlay,
                amap_start_px,
                amap_end_px,
                (45, 45, 220, 222),
                thickness=1,
            )

    def mm2px_minimap(self, mm_dist: float) -> int:
        return int(mm_dist * self.scale_factor)

    def px2mm_minimap(self, px_dist: int) -> float:
        return px_dist / self.scale_factor

    def save_minimap(self):
        if path := get_save_file_dialog(filetypes=[("PNG Image", "*.png")]):
            path += ".png" if not path.endswith(".png") else ""
            cv2.imwrite(path, self.minimap)
            logger.info(f"Saved minimap at {path}")
        else:
            logger.info(f"Saved minimap cancelled")

    def make_areal_map(self, map_name: str):
        self.map_name = map_name
        self.current_obj_for_mapping = self.master.camera_panel.get_objective()

        amap_step_w_mm = imgut.scale_px2mm(CamConsts.REAL_WIDTH, self.current_obj_for_mapping)
        amap_step_h_mm = imgut.scale_px2mm(CamConsts.REAL_HEIGHT, self.current_obj_for_mapping)

        amap_size_w_mm = self.amap_end_mm[0] - self.amap_start_mm[0]
        amap_size_h_mm = self.amap_end_mm[1] - self.amap_start_mm[1]

        amap_size_w_px = imgut.scale_mm2px(amap_size_w_mm, self.current_obj_for_mapping)
        amap_size_h_px = imgut.scale_mm2px(amap_size_h_mm, self.current_obj_for_mapping)

        # steps required in each direction
        imgs_w_req = (
            amap_size_w_px // CamConsts.REAL_WIDTH + 1 if (amap_size_w_px % CamConsts.REAL_WIDTH > 0) else 0
        )
        imgs_h_req = (
            amap_size_h_px // CamConsts.REAL_HEIGHT + 1
            if (amap_size_h_px % CamConsts.REAL_HEIGHT > 0)
            else 0
        )

        variables = {
            "amap_step_w_mm": amap_step_w_mm,
            "amap_step_h_mm": amap_step_h_mm,
            "imgs_w_req": imgs_w_req,
            "imgs_h_req": imgs_h_req,
            "start_x_mm": self.amap_start_mm[0],
            "start_y_mm": self.amap_start_mm[1],
            "amap_size_w_px": amap_size_w_px,
            "amap_size_h_px": amap_size_h_px,
        }
        self.amap_images = []

        # run script with vars
        msg = "Loading on-start variables for areal mapping:"
        self.master.automation_controller.update_variables(variables, optional_msg=msg)
        self.master.automation_controller.execute_script_file(AREAL_MAP_SCRIPT)

    def add_image_to_areal_map(self, image: Image, posx: int, posy: int):
        self.amap_images.append(
            [posx * CamConsts.REAL_WIDTH, posy * CamConsts.REAL_HEIGHT, np.array(image)]
        )

    def compose_images(self, cut_x, cut_y):
        combine = np.zeros((10, 10, 3))  # starting canvas
        for img_data in self.amap_images:
            x, y, img = img_data
            combine = imgut.add_image(combine, img, (x, y))
        combine = combine[:cut_y, :cut_x, :]
        fpath = self.master.image_control._format_name(self.map_name)
        try:
            cv2.imwrite(fpath, combine)
            logger.info(f"Saved map at {fpath}")
        except Exception as e:
            logger.error(f"Something went wrong")
            logger.error(e)

    def move_on_minimap(self, x: int, y: int):
        x_mm = self.px2mm_minimap(x)
        y_mm = self.px2mm_minimap(y)
        self.master.stage_controller.set_postion(x_mm, y_mm)
