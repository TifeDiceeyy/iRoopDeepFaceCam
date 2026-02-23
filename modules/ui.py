import threading
import modules.server
import os
import webbrowser
import customtkinter as ctk
from typing import Callable, Tuple,  List
import cv2
from PIL import Image, ImageOps
from cv2_enumerate_cameras import enumerate_cameras
import modules.globals
import modules.metadata
from modules.face_analyser import get_one_face, get_one_face_left, get_one_face_right,get_many_faces
from modules.capturer import get_video_frame, get_video_frame_total
from modules.processors.frame.core import get_frame_processors_modules
from modules.processors.frame.face_swapper import estimate_head_rotation

from modules.utilities import is_image, is_video, resolve_relative_path, has_image_extension
import numpy as np
import time

try:
    import pyvirtualcam
    PYVIRTUALCAM_AVAILABLE = True
except ImportError:
    PYVIRTUALCAM_AVAILABLE = False

# Global reference to the worker so the dropdown can find it
worker = None 

class LiveSwapWorker:
    def __init__(self):
        self.stopped = False
        self.latest_frame = None
        self.lock = threading.Lock()
        self.camera = None
        self.frame_processors = []
        self.source_images = []
        
        # Resolution Variables
        self.camera_index = 0
        self.change_res_flag = False
        self.new_width = 640
        self.new_height = 480

        # Auto-Rotation Variables
        self.rotation_check_counter = 0

        # Virtual Camera
        self._vcam = None

    def start(self, camera_index):
        self.camera_index = camera_index
        
        # 1. Load Processors (From your code)
        self.frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
        
        # 2. Reset Tracking (From your code)
        if modules.globals.face_tracking:
            for processor in self.frame_processors:
                if hasattr(processor, 'reset_face_tracking'):
                    processor.reset_face_tracking()
                    
        # 3. Load Source Images (From your code)
        if modules.globals.source_path:
            source_image = cv2.imread(modules.globals.source_path)
            faces = get_many_faces(source_image)
            if faces:
                self.source_images = sorted(faces, key=lambda face: face.bbox[0])[:20]
        
        # 4. Extract Embeddings (From your code)
        for frame_processor in self.frame_processors:
             if hasattr(frame_processor, 'extract_face_embedding') and self.source_images:
                source_embeddings = []
                for face in self.source_images:
                     source_embeddings.append(frame_processor.extract_face_embedding(face))
                modules.globals.source_face_left_embedding = source_embeddings

        threading.Thread(target=self.process_loop, daemon=True).start()

    def set_resolution(self, width, height):
        self.new_width = width
        self.new_height = height
        self.change_res_flag = True

    def process_loop(self):
        # Initial Setup
        self.camera = cv2.VideoCapture(self.camera_index)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, modules.ui.PREVIEW_DEFAULT_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, modules.ui.PREVIEW_DEFAULT_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, 60)

        while not self.stopped:
            # --- HANDLE RESOLUTION CHANGE ---
            if self.change_res_flag:
                if self.camera.isOpened(): self.camera.release()
                self.camera = cv2.VideoCapture(self.camera_index)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.new_width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.new_height)
                self.camera.set(cv2.CAP_PROP_FPS, 60)
                self.change_res_flag = False

            if self.camera.isOpened(): self.camera.grab() # Latency hack

            ret, current_frame = self.camera.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Flip Logic
            if modules.globals.flip_x:
                current_frame = cv2.flip(current_frame, 1)
            if modules.globals.flip_y:
                current_frame = cv2.flip(current_frame, 0)

            # --- YOUR AUTO ROTATION LOGIC (Moved to Thread) ---
            if modules.globals.auto_rotate_value and self.rotation_check_counter % modules.globals.rotation_check_interval == 0:
                rotation_check_faces = get_many_faces(current_frame)
                new_rot = 0
                found_rotation = False

                if rotation_check_faces:
                    main_face = max(rotation_check_faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                    angle = estimate_head_rotation(main_face)
                    if angle > 60: new_rot = -90 
                    elif angle < -60: new_rot = 90
                    found_rotation = True
                
                if not found_rotation:
                    frame_180 = cv2.rotate(current_frame, cv2.ROTATE_180)
                    faces_180 = get_many_faces(frame_180)
                    if faces_180:
                        new_rot = 180
                        found_rotation = True

                if found_rotation:
                    if modules.globals.face_rot_range != new_rot:
                        modules.globals.face_rot_range = new_rot
                        # Note: We update the logic variable here. 
                        # Visual dropdown updates happen in the UI thread loop to avoid crashes.

            self.rotation_check_counter += 1
            # --------------------------------------------------

            # --- FACE SWAP PROCESSING ---
            processed_frame = current_frame.copy()
            if self.source_images:
                try:
                    for processor in self.frame_processors:
                        processed_frame = processor.process_frame(self.source_images, processed_frame)
                except Exception:
                    pass

            # Update Latest Frame
            with self.lock:
                self.latest_frame = processed_frame

            # Virtual Camera Output
            if PYVIRTUALCAM_AVAILABLE and modules.globals.virtual_camera_out:
                try:
                    h, w = processed_frame.shape[:2]
                    if self._vcam is None or self._vcam.width != w or self._vcam.height != h:
                        if self._vcam is not None:
                            self._vcam.close()
                            self._vcam = None
                        self._vcam = pyvirtualcam.Camera(width=w, height=h, fps=30, delay=0)
                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    self._vcam.send(rgb_frame)
                    self._vcam.sleep_until_next_frame()
                except Exception:
                    pass
            else:
                if self._vcam is not None:
                    try:
                        self._vcam.close()
                    except Exception:
                        pass
                    self._vcam = None

        if self._vcam is not None:
            try:
                self._vcam.close()
            except Exception:
                pass
            self._vcam = None
        if self.camera:
            self.camera.release()

    def get_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def stop(self):
        self.stopped = True
        if self._vcam is not None:
            try:
                self._vcam.close()
            except Exception:
                pass
            self._vcam = None
        
global camera
camera = None

ROOT = None
ROOT_HEIGHT = 1100
ROOT_WIDTH = 600

PREVIEW = None
PREVIEW_IMAGE = None
PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH  = 1300
PREVIEW_DEFAULT_WIDTH  = 640
PREVIEW_DEFAULT_HEIGHT = 360
BLUR_SIZE=1

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

preview_label = None
preview_label_cam = None
preview_slider = None
source_label = None
target_label = None
status_label = None

img_ft, vid_ft = modules.globals.file_types


def init(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global ROOT, PREVIEW, PREVIEW_IMAGE

    modules.server.start()
    
    ROOT = create_root(start, destroy)
    PREVIEW = create_preview(ROOT)
    # PREVIEW_IMAGE = create_preview_image(ROOT) # <--- DELETE or COMMENT OUT THIS LINE

    return ROOT

def refresh_preview(*args):
    """
    Refreshes the preview window if it is open and NOT in Live Camera mode.
    """
    # 1. Check if Preview Window is open
    if PREVIEW and PREVIEW.state() == 'normal':
        # 2. Check if Live Worker is running (we don't want to interfere with Live Cam)
        if worker is not None and not worker.stopped:
            return

        # 3. Get current frame from slider (default to 0 for images)
        current_frame = 0
        if preview_slider:
            try:
                current_frame = int(preview_slider.get())
            except Exception:
                pass
        
        # 4. Trigger the update
        update_preview(current_frame)

def create_root(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    
    global source_label, target_label, status_label
    global preview_size_var, mouth_mask_var,mask_size_var
    global mask_down_size_var,mask_feather_ratio_var
    global flip_faces_value,fps_label,stickyface_var
    global detect_face_right_value,filter_var
    global pseudo_threshold_var,both_faces_var
    global face_tracking_value,many_faces_var,pseudo_face_var,rot_range_var,face_index_var
    global target_face1_value,target_face2_value,target_face3_value,target_face4_value,target_face5_value
    global target_face6_value,target_face7_value,target_face8_value,target_face9_value,target_face10_value
    global pseudo_face_switch, stickiness_dropdown, pseudo_threshold_dropdown, clear_tracking_button
    global embedding_weight_size_var,position_size_var,old_embedding_size_var,new_embedding_size_var
    global weight_distribution_size_var,embedding_weight_size_dropdown,weight_distribution_size_dropdown
    global position_size_dropdown, old_embedding_size_dropdown,new_embedding_size_dropdown,camera_var
    global enhancer_switch 
    global auto_rotate_var

    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode('system')
    ctk.set_default_color_theme(resolve_relative_path('ui.json'))

    root = ctk.CTk()

    # --- DYNAMIC STARTUP SIZE ---
    screen_height = root.winfo_screenheight()
    
    # Check if screen is large enough for full 900px height (plus buffer for taskbar)
    # If yes, start at 900. If no, start at 600 (or max available).
    if screen_height >= (ROOT_HEIGHT + 100):
        initial_height = ROOT_HEIGHT
    else:
        initial_height = 600

    root.geometry(f"{ROOT_WIDTH}x{initial_height}")
    # ----------------------------

    # Lower minimum height to allow window resizing on small screens (scrollbar will activate)
    root.minsize(ROOT_WIDTH, 600)
    root.title(f'{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}')
    root.configure()
    root.protocol('WM_DELETE_WINDOW', lambda: destroy())

    # --- SCROLLBAR IMPLEMENTATION ---
    # 1. Create a Scrollable Frame that fills the window
    main_scroll_frame = ctk.CTkScrollableFrame(root, fg_color="transparent")
    main_scroll_frame.pack(fill="both", expand=True)

    # 2. Create a Container Frame inside the scroll view.
    # We fix the height to ROOT_HEIGHT (900) so the 'place' layout works exactly as designed.
    ui_container = ctk.CTkFrame(main_scroll_frame, height=ROOT_HEIGHT, fg_color="transparent")
    ui_container.pack(fill="x", expand=True)
    ui_container.pack_propagate(False) # Prevent frame from shrinking
    # --------------------------------

    modules.globals.face_index_var = ctk.StringVar(value="0")
    y_start = 0.0082
    y_increment = 0.041

    # NOTE: All widgets are now parented to 'ui_container' instead of 'root'

    info_label = ctk.CTkLabel(ui_container, text='Webcam takes 30 seconds to start on first face detection', justify='center')
    info_label.place(relx=0, rely=0, relwidth=1)
    fps_label = ctk.CTkLabel(ui_container, text='FPS:  ', justify='center',font=("Arial", 12))
    fps_label.place(relx=0, rely=0.04, relwidth=1)
    
    def update_det_freq(value):
        modules.globals.detection_frequency = int(value)

    det_freq_label = ctk.CTkLabel(ui_container, text='Detect Freq:', font=("Arial", 12), anchor="e")
    det_freq_label.place(relx=0.36, rely=0.07, relwidth=0.2)

    det_freq_var = ctk.StringVar(value=str(modules.globals.detection_frequency))
    det_freq_dropdown = ctk.CTkOptionMenu(ui_container, 
                                          values=[str(i) for i in range(1, 11)],
                                          variable=det_freq_var,
                                          command=update_det_freq,
                                          width=50,
                                          height=22,
                                          font=("Arial", 11))
    det_freq_dropdown.place(relx=0.45, rely=0.10,relwidth=0.12)
    
    # Image preview area
    source_label = ctk.CTkLabel(ui_container, text=None)
    source_label.place(relx=0.03, rely=y_start + 0.40*y_increment, relwidth=0.40, relheight=0.15)

    target_label = ctk.CTkLabel(ui_container, text=None)
    target_label.place(relx=0.58, rely=y_start + 0.40*y_increment, relwidth=0.40, relheight=0.15)

    y_align = 3.35

    # Buttons for selecting source and target
    select_face_button = ctk.CTkButton(ui_container, text='Select a face/s\n(left to right 10 faces max)', cursor='hand2', command=lambda: select_source_path())
    select_face_button.place(relx=0.05, rely=y_start + 3.35*y_increment, relwidth=0.36, relheight=0.06)

    # Initialize and create rotation range dropdown
    filter_var = ctk.StringVar(value="Normal")
    filter_dropdown = ctk.CTkOptionMenu(ui_container, values=["Normal", "White Ink", "Black Ink", "Pencil"],
                                        variable=filter_var,
                                        font=("Arial", 12),
                                        command=fliter)
    filter_dropdown.place(relx=0.42, rely=y_start + 3.35*y_increment, relwidth=0.17)


    select_target_button = ctk.CTkButton(ui_container, text='Select a target\n( Image / Video )', cursor='hand2', command=lambda: select_target_path())
    select_target_button.place(relx=0.60, rely=y_start + 3.35*y_increment, relwidth=0.36, relheight=0.06)


    ##### Face Rotation range Frame

    # Outline frame for face rotation range and dropdown
    face_rot_frame = ctk.CTkFrame(ui_container, fg_color="transparent", border_width=1, border_color="grey")
    face_rot_frame.place(relx=0.02, rely=y_start + 5.0*y_increment, relwidth=0.96, relheight=0.041)

    # Create shared StringVar in modules.globals
    if not hasattr(modules.globals, 'face_rot_range'):
        modules.globals.face_rot_range = ctk.StringVar(value="0")

    # Function to handle rotation range changes
    def update_rotation_range(size):
        modules.globals.face_rot_range= int(size)
        modules.globals.rot_range_dropdown_preview.set(size)

    # Create rotation range label
    face_rot_label = ctk.CTkLabel(face_rot_frame, text="Face Rot Range (+90)-(-90) ->", font=("Arial", 16))
    face_rot_label.place(relx=0.02, rely=0.5, relheight=0.5, anchor="w")

    # Initialize and create rotation range dropdown
    modules.globals.rot_range_var = ctk.StringVar(value="0")
    rot_range_dropdown = ctk.CTkOptionMenu(face_rot_frame, values=["0", "90", "180", "-90"],
                                        variable=modules.globals.rot_range_var,
                                        command=update_rotation_range)
    rot_range_dropdown.place(relx=0.98, rely=0.5, relwidth=0.2, anchor="e")

    # Store the switch in modules.globals for access from create_preview
    modules.globals.rot_range_dropdown_preview = rot_range_dropdown

    # --- 2. FREQUENCY DROPDOWN ---
    def update_freq(value):
        modules.globals.rotation_check_interval = int(value)

    freq_label = ctk.CTkLabel(face_rot_frame, text="Freq:", font=("Arial", 12))
    freq_label.place(relx=0.62, rely=0.5, anchor="e")

    freq_var = ctk.StringVar(value=str(modules.globals.rotation_check_interval))
    freq_dropdown = ctk.CTkOptionMenu(face_rot_frame, values=["1", "2", "3", "5", "10", "15", "30", "60"],
                                      variable=freq_var,
                                      command=update_freq,
                                      width=60)
    freq_dropdown.place(relx=0.75, rely=0.5, anchor="e")
    # -----------------------------

    # --- NEW AUTO ROTATE SWITCH ---
    
    # Shared BooleanVar for Auto Rotate
    if not hasattr(modules.globals, 'auto_rotate_var'):
        modules.globals.auto_rotate_var = ctk.BooleanVar(value=modules.globals.auto_rotate_value)

    def toggle_auto_rotate():
        is_auto = modules.globals.auto_rotate_var.get()
        modules.globals.auto_rotate_value = is_auto
        
        # Sync Main Switch
        if modules.globals.auto_rotate_switch_main:
            modules.globals.auto_rotate_switch_main.select() if is_auto else modules.globals.auto_rotate_switch_main.deselect()
            
        # Sync Preview Switch
        if modules.globals.auto_rotate_switch_preview:
            modules.globals.auto_rotate_switch_preview.select() if is_auto else modules.globals.auto_rotate_switch_preview.deselect()

    auto_rotate_switch = ctk.CTkSwitch(face_rot_frame, text='Auto', 
                                      variable=modules.globals.auto_rotate_var, 
                                      command=toggle_auto_rotate, width=50)
    # Place it to the left of the dropdown (0.98 - 0.2 = 0.78 area)
    auto_rotate_switch.place(relx=0.55, rely=0.5, anchor="e")
    
    modules.globals.auto_rotate_switch_main = auto_rotate_switch



    # Left column of switches
    both_faces_var = ctk.BooleanVar(value=modules.globals.both_faces)
    both_faces_switch = ctk.CTkSwitch(ui_container, text='Use 1st 2 Faces', variable=both_faces_var, cursor='hand2',
                                    command=lambda: both_faces(modules.globals, 'both_faces', both_faces_var.get()))
    both_faces_switch.place(relx=0.03, rely=y_start + 6.4*y_increment, relwidth=0.8)

    def update_mask_target(value):
        modules.globals.mask_target_option = value

    # # Label "Mask:"
    # mask_target_label = ctk.CTkLabel(ui_container, text='Mask:', font=("Arial", 12))
    # mask_target_label.place(relx=0.35, rely=y_start + 6.4*y_increment, relwidth=0.05)

    # # Dropdown [Both, Left, Right]
    # mask_target_var = ctk.StringVar(value="Both")
    # mask_target_dropdown = ctk.CTkOptionMenu(ui_container, 
    #                                       values=["Both", "Left", "Right"],
    #                                       variable=mask_target_var,
    #                                       command=update_mask_target,
    #                                       width=60,
    #                                       height=22,
    #                                       font=("Arial", 11))
    # mask_target_dropdown.place(relx=0.42, rely=y_start + 6.4*y_increment + 0.002, relwidth=0.10)

    flip_faces_value = ctk.BooleanVar(value=modules.globals.flip_faces)
    flip_faces_switch = ctk.CTkSwitch(ui_container, text='Flip First Two Source Faces', variable=flip_faces_value, cursor='hand2',
                                    command=lambda: flip_faces('flip_faces', flip_faces_value.get()))
    flip_faces_switch.place(relx=0.03, rely=y_start + 7.1*y_increment, relwidth=0.8)

    detect_face_right_value = ctk.BooleanVar(value=modules.globals.detect_face_right)
    detect_face_right_switch = ctk.CTkSwitch(ui_container, text='Detect Target Faces From Right', variable=detect_face_right_value, cursor='hand2',
                                    command=lambda: detect_faces_right('detect_face_right', detect_face_right_value.get()))
    detect_face_right_switch.place(relx=0.03, rely=y_start + 7.8*y_increment, relwidth=0.8)

    many_faces_var = ctk.BooleanVar(value=modules.globals.many_faces)
    many_faces_switch = ctk.CTkSwitch(ui_container, text='Use All Source Faces (10 Max)', variable=many_faces_var, cursor='hand2',
                                    command=lambda: many_faces('many_faces', many_faces_var.get()))
    many_faces_switch.place(relx=0.03, rely=y_start + 8.5*y_increment, relwidth=0.8)

    show_target_face_box_var = ctk.BooleanVar(value=modules.globals.show_target_face_box)
    show_target_face_box_switch = ctk.CTkSwitch(ui_container, text='Show InsightFace Landmarks', variable=show_target_face_box_var, cursor='hand2',
                                    command=lambda: setattr(modules.globals, 'show_target_face_box', show_target_face_box_var.get()))
    show_target_face_box_switch.place(relx=0.03, rely=y_start + 9.2*y_increment, relwidth=0.8)

    show_mouth_mask_var = ctk.BooleanVar(value=modules.globals.show_mouth_mask_box)
    show_mouth_mask_switch = ctk.CTkSwitch(ui_container, text='Show Mouth Mask Box', variable=show_mouth_mask_var, cursor='hand2',
                                    command=lambda: setattr(modules.globals, 'show_mouth_mask_box', show_mouth_mask_var.get()))
    show_mouth_mask_switch.place(relx=0.03, rely=y_start + 9.9*y_increment, relwidth=0.8)

   # Create a shared BooleanVar in modules.globals
    if not hasattr(modules.globals, 'flipX_var'):
        modules.globals.flipX_var = ctk.BooleanVar(value=modules.globals.flip_x)

    # Right column of switches
    def toggle_flipX():
        is_flipX = modules.globals.flipX_var.get()
        modules.globals.flip_x = is_flipX
        if hasattr(modules.globals, 'flipX_switch_preview'):
            modules.globals.flipX_switch_preview.select() if is_flipX else modules.globals.flipX_switch_preview.deselect()

    live_flip_x_var = ctk.BooleanVar(value=modules.globals.flip_x)


    live_flip_x_vswitch = ctk.CTkSwitch(ui_container, text='Flip X',variable=modules.globals.flipX_var, cursor='hand2',
                                      command=toggle_flipX)
    live_flip_x_vswitch.place(relx=0.55, rely=y_start + 6.4*y_increment, relwidth=0.2)


    # Store the switch in modules.globals for access from create_root
    modules.globals.flipX_switch_preview = live_flip_x_vswitch

   # Create a shared BooleanVar in modules.globals
    if not hasattr(modules.globals, 'flipY_var'):
        modules.globals.flipY_var = ctk.BooleanVar(value=modules.globals.flip_y)

    # Right column of switches
    def toggle_flipY():
        is_flipY = modules.globals.flipY_var.get()
        modules.globals.flip_y = is_flipY
        if hasattr(modules.globals, 'flipX_switch_preview'):
            modules.globals.flipY_switch_preview.select() if is_flipY else modules.globals.flipY_switch_preview.deselect()


    live_flip_y_var = ctk.BooleanVar(value=modules.globals.flip_y)
    live_flip_y_switch = ctk.CTkSwitch(ui_container, text='Flip Y',variable=modules.globals.flipY_var, cursor='hand2',
                                      command=toggle_flipY)
    live_flip_y_switch.place(relx=0.80, rely=y_start + 6.4*y_increment, relwidth=0.2)

    # Store the switch in modules.globals for access from create_root
    modules.globals.flipY_switch_preview = live_flip_y_switch

    keep_fps_var = ctk.BooleanVar(value=modules.globals.keep_fps)
    keep_fps_switch = ctk.CTkSwitch(ui_container, text='Keep fps', variable=keep_fps_var, cursor='hand2',
                                    command=lambda: setattr(modules.globals, 'keep_fps', keep_fps_var.get()))
    keep_fps_switch.place(relx=0.55, rely=y_start + 7.1*y_increment, relwidth=0.4)

    keep_audio_var = ctk.BooleanVar(value=modules.globals.keep_audio)
    keep_audio_switch = ctk.CTkSwitch(ui_container, text='Keep Audio', variable=keep_audio_var, cursor='hand2',
                                    command=lambda: setattr(modules.globals, 'keep_audio', keep_audio_var.get()))
    keep_audio_switch.place(relx=0.55, rely=y_start + 7.8*y_increment, relwidth=0.4)

    keep_frames_var = ctk.BooleanVar(value=modules.globals.keep_frames)
    keep_frames_switch = ctk.CTkSwitch(ui_container, text='Keep Frames', variable=keep_frames_var, cursor='hand2',
                                    command=lambda: setattr(modules.globals, 'keep_frames', keep_frames_var.get()))
    keep_frames_switch.place(relx=0.55, rely=y_start + 8.5*y_increment, relwidth=0.4)

    nsfw_filter_var = ctk.BooleanVar(value=modules.globals.nsfw_filter)
    nsfw_filter_switch = ctk.CTkSwitch(ui_container, text='NSFW Filter', variable=nsfw_filter_var, cursor='hand2',
                                    command=lambda: setattr(modules.globals, 'nsfw_filter', nsfw_filter_var.get()))
    nsfw_filter_switch.place(relx=0.55, rely=y_start + 9.2*y_increment, relwidth=0.4)

    enhancer_value = ctk.BooleanVar(value=modules.globals.fp_ui['face_enhancer'])
    enhancer_switch = ctk.CTkSwitch(ui_container, text='Face Enhancer', variable=enhancer_value, cursor='hand2',
                                    command=lambda: update_tumbler('face_enhancer', enhancer_value.get()))
    enhancer_switch.place(relx=0.55, rely=y_start + 9.9*y_increment, relwidth=0.20)

    enhancer_fidelity_label = ctk.CTkLabel(ui_container, text='Fidelity:', font=("Arial", 12), anchor='e')
    enhancer_fidelity_label.place(relx=0.75, rely=y_start + 9.9*y_increment, relwidth=0.11)

    enhancer_fidelity_var = ctk.StringVar(value=str(modules.globals.enhancer_fidelity))
    def update_enhancer_fidelity(value):
        modules.globals.enhancer_fidelity = float(value)
        refresh_preview()
    enhancer_fidelity_dropdown = ctk.CTkOptionMenu(
        ui_container,
        values=["0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"],
        variable=enhancer_fidelity_var,
        command=update_enhancer_fidelity,
        height=24, font=("Arial", 12))
    enhancer_fidelity_dropdown.place(relx=0.86, rely=y_start + 9.9*y_increment, relwidth=0.12)


    ##### Mouth Mask Frame

    # Outline frame for mouth mask and dropdown
    outline_frame = ctk.CTkFrame(ui_container, fg_color="transparent", border_width=1, border_color="grey")
    outline_frame.place(relx=0.02, rely=y_start + 10.9*y_increment, relwidth=0.96, relheight=0.041)

   # Create a shared BooleanVar in modules.globals
    if not hasattr(modules.globals, 'mouth_mask_var'):
        modules.globals.mouth_mask_var = ctk.BooleanVar(value=modules.globals.mouth_mask)

    # Mouth mask switch
    def toggle_mouthmask():
        is_mouthmask = modules.globals.mouth_mask_var.get()
        modules.globals.mouth_mask = is_mouthmask
        if hasattr(modules.globals, 'mouth_mask_switch_preview'):
            modules.globals.mouth_mask_switch_preview.select() if is_mouthmask else modules.globals.mouth_mask_switch_preview.deselect()

    mouth_mask_switch = ctk.CTkSwitch(outline_frame, text='Mouth Mask (M) | Feather, Padding, Top ->', 
                                      variable=modules.globals.mouth_mask_var, cursor='hand2',
                                      command=toggle_mouthmask)
    mouth_mask_switch.place(relx=0.02, rely=0.5, relwidth=0.6, relheight=0.5, anchor="w")

    # Store the switch in modules.globals for access from create_preview
    modules.globals.mouth_mask_switch_preview = mouth_mask_switch
    
    # Size dropdown (rightmost)
    mask_size_var = ctk.StringVar(value="1")
    mask_size_dropdown = ctk.CTkOptionMenu(outline_frame, values=["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","25","30","35","40","45","50"],
                                            variable=mask_size_var,
                                            command=mask_size)
    mask_size_dropdown.place(relx=0.98, rely=0.5, relwidth=0.1, anchor="e")

    # Down size dropdown
    mask_down_size_var = ctk.StringVar(value="1.05")
    mask_down_size_dropdown = ctk.CTkOptionMenu(outline_frame, values=["0.80","0.85","0.90","0.95","1.00","1.05","1.10","1.15","1.20","1.25","1.50","1.75","2.00","2.25","2.50","2.75","3.00"],
                                            variable=mask_down_size_var,
                                            command=mask_down_size)
    mask_down_size_dropdown.place(relx=0.87, rely=0.5, relwidth=0.12, anchor="e")

    # Feather ratio dropdown
    mask_feather_ratio_var = ctk.StringVar(value="30")
    mask_feather_ratio_size_dropdown = ctk.CTkOptionMenu(outline_frame, values=["8","9","10","15","20","25","30","35","40"],
                                            variable=mask_feather_ratio_var,
                                            command=mask_feather_ratio_size)
    mask_feather_ratio_size_dropdown.place(relx=0.76, rely=0.5, relwidth=0.1,  anchor="e")


    ##### Face Tracking Frame

    # Outline frame for face tracking
    outline_face_track_frame = ctk.CTkFrame(ui_container, fg_color="transparent", border_width=1, border_color="grey")
    outline_face_track_frame.place(relx=0.02, rely=y_start + 11.9*y_increment, relwidth=0.96, relheight=0.196)
     # Face Tracking switch
    face_tracking_value = ctk.BooleanVar(value=modules.globals.face_tracking)
    face_tracking_switch = ctk.CTkSwitch(outline_face_track_frame, text='Auto Face Track', variable=face_tracking_value, cursor='hand2',
                                    command=lambda: face_tracking('face_tracking', face_tracking_value.get()))
    face_tracking_switch.place(relx=0.02, rely=0.1, relwidth=0.4)

    # Pseudo Face switch
    pseudo_face_var = ctk.BooleanVar(value=modules.globals.use_pseudo_face)
    pseudo_face_switch = ctk.CTkSwitch(outline_face_track_frame, text='Pseudo Face\n(fake face\nfor occlusions)', variable=pseudo_face_var, cursor='hand2',
                                    command=lambda: setattr(modules.globals, 'use_pseudo_face', pseudo_face_var.get()))
    pseudo_face_switch.place(relx=0.02, rely=0.3, relwidth=0.4)


    # Red box frame
    red_box_frame = ctk.CTkFrame(outline_face_track_frame, fg_color="transparent", border_width=1, border_color="#800000")
    red_box_frame.place(relx=0.32, rely=0.02, relwidth=0.31, relheight=0.65)

   
    tempFontSiz=10
    # Face Cosine Similarity label
    similarity_label = ctk.CTkLabel(red_box_frame, text="Similarity * Position",font=("Arial", 14) )
    similarity_label.place(relx=0.05, rely=0.01, relwidth=0.85 )
    # Target Face 1 label and value
    target_face1_label = ctk.CTkLabel(red_box_frame, text="Face 1:", font=("Arial", tempFontSiz))
    target_face1_label.place(relx=0.02, rely=0.18, relwidth=0.3)

    target_face1_value = ctk.CTkLabel(red_box_frame, text="0.00", anchor="w", font=("Arial", tempFontSiz))
    target_face1_value.place(relx=0.30, rely=0.18, relwidth=0.2)
    # Target Face 2 label and value
    target_face2_label = ctk.CTkLabel(red_box_frame, text="Face 2:", font=("Arial", tempFontSiz))
    target_face2_label.place(relx=0.02, rely=0.30, relwidth=0.3)

    target_face2_value = ctk.CTkLabel(red_box_frame, text="0.00", anchor="w", font=("Arial", tempFontSiz))
    target_face2_value.place(relx=0.30, rely=0.30, relwidth=0.2)

    target_face3_label = ctk.CTkLabel(red_box_frame, text="Face 3:", font=("Arial", tempFontSiz))
    target_face3_label.place(relx=0.02, rely=0.42, relwidth=0.3)

    target_face3_value = ctk.CTkLabel(red_box_frame, text="0.00", anchor="w", font=("Arial", tempFontSiz))
    target_face3_value.place(relx=0.30, rely=0.42, relwidth=0.2)

    target_face4_label = ctk.CTkLabel(red_box_frame, text="Face 4:", font=("Arial", tempFontSiz))
    target_face4_label.place(relx=0.02, rely=0.54, relwidth=0.3)

    target_face4_value = ctk.CTkLabel(red_box_frame, text="0.00", anchor="w", font=("Arial", tempFontSiz))
    target_face4_value.place(relx=0.30, rely=0.54, relwidth=0.2)

    target_face5_label = ctk.CTkLabel(red_box_frame, text="Face 5:", font=("Arial", tempFontSiz))
    target_face5_label.place(relx=0.02, rely=0.66, relwidth=0.3)

    target_face5_value = ctk.CTkLabel(red_box_frame, text="0.00", anchor="w", font=("Arial", tempFontSiz))
    target_face5_value.place(relx=0.30, rely=0.66, relwidth=0.2)



    target_face6_label = ctk.CTkLabel(red_box_frame, text="Face 6:", font=("Arial", tempFontSiz))
    target_face6_label.place(relx=0.50, rely=0.18, relwidth=0.3)

    target_face6_value = ctk.CTkLabel(red_box_frame, text="0.00", anchor="w", font=("Arial", tempFontSiz))
    target_face6_value.place(relx=0.78, rely=0.18, relwidth=0.2)

    target_face7_label = ctk.CTkLabel(red_box_frame, text="Face 7:", font=("Arial", tempFontSiz))
    target_face7_label.place(relx=0.50, rely=0.30, relwidth=0.3)

    target_face7_value = ctk.CTkLabel(red_box_frame, text="0.00", anchor="w", font=("Arial", tempFontSiz))
    target_face7_value.place(relx=0.78, rely=0.30, relwidth=0.2)

    target_face8_label = ctk.CTkLabel(red_box_frame, text="Face 8:", font=("Arial", tempFontSiz))
    target_face8_label.place(relx=0.50, rely=0.42, relwidth=0.3)

    target_face8_value = ctk.CTkLabel(red_box_frame, text="0.00", anchor="w", font=("Arial", tempFontSiz))
    target_face8_value.place(relx=0.78, rely=0.42, relwidth=0.2)

    target_face9_label = ctk.CTkLabel(red_box_frame, text="Face 9:", font=("Arial", tempFontSiz))
    target_face9_label.place(relx=0.50, rely=0.54, relwidth=0.3)

    target_face9_value = ctk.CTkLabel(red_box_frame, text="0.00", anchor="w", font=("Arial", tempFontSiz))
    target_face9_value.place(relx=0.78, rely=0.54, relwidth=0.2)

    target_face10_label = ctk.CTkLabel(red_box_frame, text="Face 10:", font=("Arial", tempFontSiz))
    target_face10_label.place(relx=0.52, rely=0.66, relwidth=0.3)

    target_face10_value = ctk.CTkLabel(red_box_frame, text="0.00", anchor="w", font=("Arial", tempFontSiz))
    target_face10_value.place(relx=0.78, rely=0.66, relwidth=0.2)
    #     # Target Face 2 label and value
    # target_face2_label = ctk.CTkLabel(red_box_frame, text="* MAX TWO FACE ON\nSCREEN DETECTED FROM\nLEFT OR RIGHT *", font=("Arial", 10))
    # target_face2_label.place(relx=0.05, rely=0.60, relwidth=0.9)


    # Stickiness Factor label
    stickiness_label = ctk.CTkLabel(outline_face_track_frame, text="Stickiness Factor",font=("Arial", 14))
    stickiness_label.place(relx=0.72, rely=0.01, relwidth=0.2)

    # Stickiness Greater label
    stickiness_greater_label = ctk.CTkLabel(outline_face_track_frame, text=">",font=("Arial", 14))
    stickiness_greater_label.place(relx=0.65, rely=0.14, relwidth=0.1)

    # Stickiness Factor dropdown
    stickyface_var = ctk.StringVar(value="0.20")
    stickiness_dropdown = ctk.CTkOptionMenu(outline_face_track_frame, values=["0.05","0.10","0.15","0.20","0.25","0.30","0.35","0.40","0.45","0.50","0.55","0.60","0.65","0.70","0.75","0.80","0.85","0.90","0.95","1.00"],
                                            variable=stickyface_var,
                                            command=stickiness_factor_size)
    stickiness_dropdown.place(relx=0.75, rely=0.14, relwidth=0.15)


    # Stickiness Greater label
    pseudo_threshold_greater_label = ctk.CTkLabel(outline_face_track_frame, text="<",font=("Arial", 14))
    pseudo_threshold_greater_label.place(relx=0.65, rely=0.30, relwidth=0.1)

    # Pseudo Threshold dropdown
    pseudo_threshold_var = ctk.StringVar(value="0.20")
    pseudo_threshold_dropdown = ctk.CTkOptionMenu(outline_face_track_frame, values=["0.05","0.10","0.15","0.20","0.25","0.30","0.35","0.40","0.45","0.50","0.55","0.60","0.65","0.70","0.75","0.80","0.85","0.90","0.95","1.00"],
                                                variable=pseudo_threshold_var,
                                                command=pseudo_threshold_size)
    pseudo_threshold_dropdown.place(relx=0.75, rely=0.30, relwidth=0.15)

    # Pseudo Threshold label
    pseudo_threshold_label = ctk.CTkLabel(outline_face_track_frame, text="Pseudo Threshold",font=("Arial", 14))
    pseudo_threshold_label.place(relx=0.72, rely=0.42, relwidth=0.2)


    # Clear Face Tracking Data button
    clear_tracking_button = ctk.CTkButton(outline_face_track_frame, text="Reset Face Tracking", 
                                        command=clear_face_tracking_data)
    clear_tracking_button.place(relx=0.65, rely=0.55, relwidth=0.34)

    track_settings_label = ctk.CTkLabel(outline_face_track_frame, text="Embedding Weight   *   Weight Distribution   +   Position Weight            Old Weight   +   New Weight", font=("Arial", 12))
    track_settings_label.place(relx=0.01, rely=0.68, relwidth=0.96)


    embedding_weight_size_var = ctk.StringVar(value="0.60")
    embedding_weight_size_dropdown = ctk.CTkOptionMenu(outline_face_track_frame, values=["0.05","0.10","0.15","0.20","0.25","0.30","0.35","0.40","0.45","0.50","0.55","0.60","0.65","0.70","0.75","0.80","0.85","0.90","0.95","1.00"],
                                            variable=embedding_weight_size_var,
                                            command=embedding_weight_size)
    embedding_weight_size_dropdown.place(relx=0.03, rely=0.84, relwidth=0.13)

    weight_distribution_size_var = ctk.StringVar(value="1.00")
    weight_distribution_size_dropdown = ctk.CTkOptionMenu(outline_face_track_frame, values=["0.05","0.15","0.25","0.35","0.45","0.55","0.65","0.75","0.85","0.95","1.00","1.25","1.50","1.75","2.00","2.25","2.50","2.75","3.00","3.25","3.50","3.75","4.00","4.25","4.50","4.75","5.00"],
                                            variable=weight_distribution_size_var,
                                            command=weight_wistribution_size)
    weight_distribution_size_dropdown.place(relx=0.25, rely=0.84, relwidth=0.13)

    # Down size dropdown
    position_size_var = ctk.StringVar(value="0.40")
    position_size_dropdown = ctk.CTkOptionMenu(outline_face_track_frame, values=["0.05","0.10","0.15","0.20","0.25","0.30","0.35","0.40","0.45","0.50","0.55","0.60","0.65","0.70","0.75","0.80","0.85","0.90","0.95","1.00"],
                                            variable=position_size_var,
                                            command=position_size)
    position_size_dropdown.place(relx=0.48, rely=0.84, relwidth=0.13)

    # Feather ratio dropdown
    old_embedding_size_var = ctk.StringVar(value="0.90")
    old_embedding_size_dropdown = ctk.CTkOptionMenu(outline_face_track_frame, values=["0.05","0.10","0.15","0.20","0.25","0.30","0.35","0.40","0.45","0.50","0.55","0.60","0.65","0.70","0.75","0.80","0.85","0.90","0.95","1.00"],
                                            variable=old_embedding_size_var,
                                            command=old_embedding_size)
    old_embedding_size_dropdown.place(relx=0.68, rely=0.84, relwidth=0.13)

    # Feather ratio dropdown
    new_embedding_size_var = ctk.StringVar(value="0.10")
    new_embedding_size_dropdown = ctk.CTkOptionMenu(outline_face_track_frame, values=["0.05","0.10","0.15","0.20","0.25","0.30","0.35","0.40","0.45","0.50","0.55","0.60","0.65","0.70","0.75","0.80","0.85","0.90","0.95","1.00"],
                                            variable=new_embedding_size_var,
                                            command=new_embedding_size)
    new_embedding_size_dropdown.place(relx=0.84, rely=0.84, relwidth=0.13)


    # Bottom buttons
    button_width = 0.18  # Width of each button
    button_height = 0.05  # Height of each button
    button_y = 0.695  # Y position of the buttons
    space_between = (1 - (button_width * 5)) / 6  # Space between buttons

    start_button = ctk.CTkButton(ui_container, text='Start', cursor='hand2', command=lambda: select_output_path(start))
    start_button.place(relx=space_between, rely=button_y, relwidth=button_width, relheight=button_height)

    preview_button = ctk.CTkButton(ui_container, text='Preview', cursor='hand2', command=lambda: toggle_preview())
    preview_button.place(relx=space_between*2 + button_width, rely=button_y, relwidth=button_width, relheight=button_height)

    donate_button = ctk.CTkButton(ui_container, text='BuyMeACoffee', cursor='hand2', 
                                 command=lambda: webbrowser.open('https://buymeacoffee.com/ivideogameboss'),
                                 font=("Arial", 14))  # Added font parameter here
    donate_button.place(relx=space_between*3 + button_width*2, rely=button_y, relwidth=button_width, relheight=button_height)


    live_button = ctk.CTkButton(ui_container, text='Live', cursor='hand2', command=lambda: webcam_preview(), fg_color="green", hover_color="dark green")
    live_button.place(relx=space_between*4 + button_width*3, rely=button_y, relwidth=button_width, relheight=button_height)

    preview_size_var = ctk.StringVar(value="640x360")
    preview_size_dropdown = ctk.CTkOptionMenu(ui_container, values=["426x240","480x270","512x288","640x360","854x480", "960x540", "1280x720", "1920x1080"],
                                              variable=preview_size_var,
                                              command=update_preview_size,
                                              fg_color="green", button_color="dark green", button_hover_color="forest green")
    preview_size_dropdown.place(relx=space_between*5 + button_width*4, rely=button_y, relwidth=button_width, relheight=button_height)

    # --- Virtual Camera & Swapper Resolution ---
    vcam_label = ctk.CTkLabel(ui_container, text="Virtual Cam:", font=("Arial", 12), anchor="e")
    vcam_label.place(relx=0.02, rely=0.712, relwidth=0.14)

    virtual_cam_var = ctk.BooleanVar(value=modules.globals.virtual_camera_out)
    def toggle_virtual_cam():
        modules.globals.virtual_camera_out = virtual_cam_var.get()
    vcam_switch = ctk.CTkSwitch(ui_container, text='', variable=virtual_cam_var,
                                cursor='hand2', command=toggle_virtual_cam, width=40)
    vcam_switch.place(relx=0.17, rely=0.712, relwidth=0.10)

    vcam_status = ctk.CTkLabel(ui_container,
                               text="" if PYVIRTUALCAM_AVAILABLE else "(install pyvirtualcam)",
                               font=("Arial", 10), text_color="grey")
    vcam_status.place(relx=0.28, rely=0.712, relwidth=0.18)

    swapper_res_label = ctk.CTkLabel(ui_container, text="Swap Res:", font=("Arial", 12), anchor="e")
    swapper_res_label.place(relx=0.49, rely=0.712, relwidth=0.14)

    swapper_res_var = ctk.StringVar(value=str(modules.globals.swapper_resolution))
    def update_swapper_res(value):
        modules.globals.swapper_resolution = int(value)
        from modules.processors.frame.face_swapper import reset_face_swapper
        reset_face_swapper()
    swapper_res_dropdown = ctk.CTkOptionMenu(ui_container, values=["128", "256", "512"],
                                             variable=swapper_res_var,
                                             command=update_swapper_res,
                                             height=24, font=("Arial", 12))
    swapper_res_dropdown.place(relx=0.64, rely=0.712, relwidth=0.13)

    # --- Face Swap Parameters row ---
    blur_label = ctk.CTkLabel(ui_container, text="Edge Blur:", font=("Arial", 12), anchor="e")
    blur_label.place(relx=0.02, rely=0.733, relwidth=0.14)

    blur_amount_var = ctk.StringVar(value=str(modules.globals.blur_amount))
    def update_blur_amount(value):
        modules.globals.blur_amount = int(value)
    blur_dropdown = ctk.CTkOptionMenu(ui_container,
                                      values=["0","2","4","6","8","10","12","16","20","24","28","32","40"],
                                      variable=blur_amount_var,
                                      command=update_blur_amount,
                                      height=24, font=("Arial", 12))
    blur_dropdown.place(relx=0.17, rely=0.733, relwidth=0.12)

    pad_label = ctk.CTkLabel(ui_container, text="Crop Pad:", font=("Arial", 12), anchor="e")
    pad_label.place(relx=0.33, rely=0.733, relwidth=0.14)

    crop_padding_var = ctk.StringVar(value=str(modules.globals.crop_padding))
    def update_crop_padding(value):
        modules.globals.crop_padding = float(value)
    pad_dropdown = ctk.CTkOptionMenu(ui_container,
                                     values=["0.0","0.05","0.1","0.15","0.2","0.25","0.3","0.35","0.4","0.5"],
                                     variable=crop_padding_var,
                                     command=update_crop_padding,
                                     height=24, font=("Arial", 12))
    pad_dropdown.place(relx=0.48, rely=0.733, relwidth=0.12)

    forehead_label = ctk.CTkLabel(ui_container, text="Forehead:", font=("Arial", 12), anchor="e")
    forehead_label.place(relx=0.62, rely=0.733, relwidth=0.13)

    forehead_var = ctk.StringVar(value=str(modules.globals.face_forehead_var))
    def update_forehead(value):
        modules.globals.face_forehead_var = float(value)
    forehead_dropdown = ctk.CTkOptionMenu(ui_container,
                                          values=["0.0","0.05","0.1","0.15","0.2","0.25","0.3","0.4","0.5"],
                                          variable=forehead_var,
                                          command=update_forehead,
                                          height=24, font=("Arial", 12))
    forehead_dropdown.place(relx=0.76, rely=0.733, relwidth=0.11)

    occluder_var = ctk.BooleanVar(value=modules.globals.use_occluder)
    def toggle_occluder():
        modules.globals.use_occluder = occluder_var.get()
    occluder_switch = ctk.CTkSwitch(ui_container, text='Occluder', variable=occluder_var,
                                    cursor='hand2', command=toggle_occluder, width=40)
    occluder_switch.place(relx=0.88, rely=0.733, relwidth=0.11)

    # --- Color Correction Frame (outlined) ---
    cc_frame = ctk.CTkFrame(ui_container, fg_color="transparent", border_width=1, border_color="grey")
    cc_frame.place(relx=0.02, rely=0.757, relwidth=0.96, relheight=0.072)

    # Row 1: Enable toggle + R / G / B
    cc_enabled_var = ctk.BooleanVar(value=modules.globals.color_correction_enabled)
    def toggle_cc():
        modules.globals.color_correction_enabled = cc_enabled_var.get()
    cc_switch = ctk.CTkSwitch(cc_frame, text='Color Correct', variable=cc_enabled_var,
                               cursor='hand2', command=toggle_cc, width=40)
    cc_switch.place(relx=0.02, rely=0.08, relwidth=0.22, relheight=0.44)

    cc_r_label = ctk.CTkLabel(cc_frame, text="R:", font=("Arial", 11), anchor="e")
    cc_r_label.place(relx=0.26, rely=0.08, relwidth=0.05, relheight=0.44)
    cc_r_var = ctk.StringVar(value=str(modules.globals.color_r))
    def update_cc_r(v): modules.globals.color_r = int(v)
    ctk.CTkOptionMenu(cc_frame, values=[str(x) for x in range(-100, 110, 10)],
                      variable=cc_r_var, command=update_cc_r, height=22, font=("Arial", 11)
                      ).place(relx=0.31, rely=0.08, relwidth=0.14, relheight=0.44)

    cc_g_label = ctk.CTkLabel(cc_frame, text="G:", font=("Arial", 11), anchor="e")
    cc_g_label.place(relx=0.46, rely=0.08, relwidth=0.05, relheight=0.44)
    cc_g_var = ctk.StringVar(value=str(modules.globals.color_g))
    def update_cc_g(v): modules.globals.color_g = int(v)
    ctk.CTkOptionMenu(cc_frame, values=[str(x) for x in range(-100, 110, 10)],
                      variable=cc_g_var, command=update_cc_g, height=22, font=("Arial", 11)
                      ).place(relx=0.51, rely=0.08, relwidth=0.14, relheight=0.44)

    cc_b_label = ctk.CTkLabel(cc_frame, text="B:", font=("Arial", 11), anchor="e")
    cc_b_label.place(relx=0.66, rely=0.08, relwidth=0.05, relheight=0.44)
    cc_b_var = ctk.StringVar(value=str(modules.globals.color_b))
    def update_cc_b(v): modules.globals.color_b = int(v)
    ctk.CTkOptionMenu(cc_frame, values=[str(x) for x in range(-100, 110, 10)],
                      variable=cc_b_var, command=update_cc_b, height=22, font=("Arial", 11)
                      ).place(relx=0.71, rely=0.08, relwidth=0.14, relheight=0.44)

    # Row 2: Brightness / Contrast / Saturation / Sharpness
    _tone_vals = ["0.5","0.6","0.7","0.8","0.9","1.0","1.1","1.2","1.3","1.4","1.5","1.6","1.8","2.0"]

    cc_br_label = ctk.CTkLabel(cc_frame, text="Bri:", font=("Arial", 11), anchor="e")
    cc_br_label.place(relx=0.02, rely=0.56, relwidth=0.07, relheight=0.40)
    cc_br_var = ctk.StringVar(value=str(modules.globals.color_brightness))
    def update_cc_br(v): modules.globals.color_brightness = float(v)
    ctk.CTkOptionMenu(cc_frame, values=_tone_vals, variable=cc_br_var,
                      command=update_cc_br, height=22, font=("Arial", 11)
                      ).place(relx=0.10, rely=0.56, relwidth=0.14, relheight=0.40)

    cc_co_label = ctk.CTkLabel(cc_frame, text="Con:", font=("Arial", 11), anchor="e")
    cc_co_label.place(relx=0.26, rely=0.56, relwidth=0.07, relheight=0.40)
    cc_co_var = ctk.StringVar(value=str(modules.globals.color_contrast))
    def update_cc_co(v): modules.globals.color_contrast = float(v)
    ctk.CTkOptionMenu(cc_frame, values=_tone_vals, variable=cc_co_var,
                      command=update_cc_co, height=22, font=("Arial", 11)
                      ).place(relx=0.34, rely=0.56, relwidth=0.14, relheight=0.40)

    cc_sa_label = ctk.CTkLabel(cc_frame, text="Sat:", font=("Arial", 11), anchor="e")
    cc_sa_label.place(relx=0.50, rely=0.56, relwidth=0.07, relheight=0.40)
    cc_sa_var = ctk.StringVar(value=str(modules.globals.color_saturation))
    def update_cc_sa(v): modules.globals.color_saturation = float(v)
    ctk.CTkOptionMenu(cc_frame, values=["0.0","0.2","0.4","0.6","0.8","1.0","1.2","1.4","1.6","1.8","2.0"],
                      variable=cc_sa_var, command=update_cc_sa, height=22, font=("Arial", 11)
                      ).place(relx=0.58, rely=0.56, relwidth=0.14, relheight=0.40)

    cc_sh_label = ctk.CTkLabel(cc_frame, text="Sharp:", font=("Arial", 11), anchor="e")
    cc_sh_label.place(relx=0.73, rely=0.56, relwidth=0.08, relheight=0.40)
    cc_sh_var = ctk.StringVar(value=str(modules.globals.color_sharpness))
    def update_cc_sh(v): modules.globals.color_sharpness = float(v)
    ctk.CTkOptionMenu(cc_frame, values=["0.0","0.2","0.4","0.6","0.8","1.0","1.2","1.4","1.6","1.8","2.0","2.5","3.0"],
                      variable=cc_sh_var, command=update_cc_sh, height=22, font=("Arial", 11)
                      ).place(relx=0.82, rely=0.56, relwidth=0.14, relheight=0.40)

    # --- Final Blend Row ---
    fb_label = ctk.CTkLabel(ui_container, text="Final Blend:", font=("Arial", 12), anchor="e")
    fb_label.place(relx=0.02, rely=0.835, relwidth=0.14)

    fb_enabled_var = ctk.BooleanVar(value=modules.globals.final_blend_enabled)
    def toggle_fb():
        modules.globals.final_blend_enabled = fb_enabled_var.get()
    fb_switch = ctk.CTkSwitch(ui_container, text='', variable=fb_enabled_var,
                               cursor='hand2', command=toggle_fb, width=40)
    fb_switch.place(relx=0.17, rely=0.835, relwidth=0.10)

    fb_amount_label = ctk.CTkLabel(ui_container, text="Swap %:", font=("Arial", 12), anchor="e")
    fb_amount_label.place(relx=0.29, rely=0.835, relwidth=0.12)

    fb_amount_var = ctk.StringVar(value=str(modules.globals.final_blend_amount))
    def update_fb_amount(v): modules.globals.final_blend_amount = int(v)
    ctk.CTkOptionMenu(ui_container, values=[str(x) for x in range(0, 105, 5)],
                      variable=fb_amount_var, command=update_fb_amount,
                      height=24, font=("Arial", 12)
                      ).place(relx=0.42, rely=0.835, relwidth=0.13)

    button_y = 0.868  # Y position of the camera row (shifted down to make room)

    # --- Camera Selection ---
    camera_label = ctk.CTkLabel(ui_container, text="Select Camera:")
    camera_label.place(relx=0.03, rely=button_y, relwidth=button_width+0.05)

    # Get available cameras
    try:
        available_cameras = enumerate_cameras()
        camera_names = [f"{cam.name} ({cam.index})" for cam in available_cameras]
        if not camera_names:
           
           initial_camera = "Select Default Camera"
           modules.globals.camera_index = 0
        else:
          # Set the initial camera if available
          initial_camera = camera_names[0]
        
    except Exception:
          camera_names = ["Select Default Camera"]
          initial_camera = "Select Default Camera"
          modules.globals.camera_index = 0
        
    camera_var = ctk.StringVar(value=initial_camera)
    camera_optionmenu = ctk.CTkOptionMenu(ui_container, values=camera_names if camera_names else ["Select Default Camera"],
                                              variable=camera_var,
                                              command=select_camera,  # Corrected line: pass function directly
                                              fg_color="grey", button_color="dark grey", button_hover_color="light grey")
    camera_optionmenu.place(relx=0.25, rely=button_y, relwidth=0.70)

    #Store the camera_var and optionmenu in modules.globals
    modules.globals.camera_var = camera_var
    modules.globals.camera_optionmenu = camera_optionmenu

    # Select the first camera if available
    if initial_camera != "Select Default Camera":
      select_camera()

    
    # Status and donate labels
    status_label = ctk.CTkLabel(ui_container, text=None, justify='center')
    status_label.place(relx=0.05, rely=0.97, relwidth=0.9)

    if not modules.globals.face_tracking:
        pseudo_face_switch.configure(state="disabled")
        stickiness_dropdown.configure(state="disabled")
        pseudo_threshold_dropdown.configure(state="disabled")
        clear_tracking_button.configure(state="disabled")
        embedding_weight_size_dropdown.configure(state="disabled")
        weight_distribution_size_dropdown.configure(state="disabled")
        position_size_dropdown.configure(state="disabled")
        old_embedding_size_dropdown.configure(state="disabled")
        new_embedding_size_dropdown.configure(state="disabled")

    setup_hotkeys(root) # Register Hotkeys for Main Window
    return root

def select_camera(*args):
    
    camera_info = modules.globals.camera_var.get()
    
    # Extract the camera index from the selected string
    if camera_info == "Default Camera":
         modules.globals.camera_index = 0 # default 0
    else:
         camera_index = int(camera_info.split('(')[-1][:-1])
         modules.globals.camera_index = camera_index
    update_camera_resolution()


def get_available_cameras():
    pass # No need to return anything here, handled by the CTkOptionMenu


def weight_wistribution_size(*args):
    size = weight_distribution_size_var.get()
    modules.globals.weight_distribution_size = float(size)
    refresh_preview()

def embedding_weight_size(*args):
    size = embedding_weight_size_var.get()
    modules.globals.embedding_weight_size = float(size)
    refresh_preview()

def position_size(*args):
    size = position_size_var.get()
    modules.globals.position_size = float(size)
    refresh_preview()

def old_embedding_size(*args):
    size = old_embedding_size_var.get()
    modules.globals.old_embedding_weight  = float(size)
    refresh_preview()

def new_embedding_size(*args):
    size = new_embedding_size_var.get()
    modules.globals.new_embedding_weight  = float(size)
    refresh_preview()

def create_preview_image(parent: ctk.CTkToplevel) -> ctk.CTkToplevel:
    global preview_label, preview_slider, topmost_switch, mouth_mask_switch_preview

    preview = ctk.CTkToplevel(parent)
    preview.withdraw()
    preview.title('Image Preview')
    preview.configure()
    preview.protocol('WM_DELETE_WINDOW', lambda: toggle_preview())
    preview.resizable(width=True, height=True)

    preview_label = ctk.CTkLabel(preview, text=None)
    preview_label.pack(fill='y', expand=True)

    preview_slider = ctk.CTkSlider(preview, from_=0, to=0, command=lambda frame_value: update_preview(frame_value))

    return preview

def create_preview(parent: ctk.CTkToplevel) -> ctk.CTkToplevel:
    global preview_label_cam, preview_slider, topmost_switch, mouth_mask_switch_preview
    global preview_label 

    preview = ctk.CTkToplevel(parent)
    preview.withdraw()
    preview.title('Double Click hide Toolbar. Always Reset Face Tracking When no Faces, Switching Live Video Stream, or New Faces. Face Index (-1) Auto')
    preview.configure()
    preview.protocol('WM_DELETE_WINDOW', lambda: toggle_preview())
    preview.resizable(width=True, height=True)

    # Create a frame for the switches (Main Toolbar Container)
    switch_frame = ctk.CTkFrame(preview)
    switch_frame.pack(fill='x', padx=10, pady=5)

    # 1. Create a sub-frame for the buttons (Top Row)
    button_row = ctk.CTkFrame(switch_frame, fg_color="transparent")
    button_row.pack(fill='x', side='top')

    # Add the "Stay on Top" switch
    def toggle_topmost():
        is_topmost = topmost_var.get()
        preview.attributes('-topmost', is_topmost)
        if is_topmost:
            preview.lift() 

    topmost_var = ctk.BooleanVar(value=False)
    topmost_switch = ctk.CTkSwitch(button_row, text='Stay on Top', variable=topmost_var, cursor='hand2',
                                   command=toggle_topmost)
    topmost_switch.pack(side='left', padx=5, pady=5)

    # Add the "Mouth Mask" switch
    def toggle_mouthmask():
        is_mouthmask = modules.globals.mouth_mask_var.get()
        modules.globals.mouth_mask = is_mouthmask
        if hasattr(modules.globals, 'mouth_mask_switch_root'):
            # Sync main window switch
            if modules.globals.mouth_mask_switch_preview.get():
                modules.globals.mouth_mask_switch_preview.select() 
            else:
                modules.globals.mouth_mask_switch_preview.deselect()
        refresh_preview() # <--- REFRESH ADDED

    mouth_mask_switch_preview = ctk.CTkSwitch(button_row, text='Mouth Mask', 
                                              variable=modules.globals.mouth_mask_var, cursor='hand2',
                                              command=toggle_mouthmask)
    mouth_mask_switch_preview.pack(side='left', padx=5, pady=5)
    modules.globals.mouth_mask_switch_preview = mouth_mask_switch_preview

    # Add the "Flip X" switch
    def toggle_flipX():
        is_flipX = modules.globals.flipX_var.get()
        modules.globals.flip_x = is_flipX
        if hasattr(modules.globals, 'flipX_switch_preview'):
            # Sync main window switch logic if needed (usually shared variable handles it)
            pass
        refresh_preview() # <--- REFRESH ADDED

    flipX_switch = ctk.CTkSwitch(button_row, text=' Flip X', 
                                 variable=modules.globals.flipX_var, cursor='hand2',
                                 command=toggle_flipX)
    flipX_switch.pack(side='left', padx=5, pady=5)
    modules.globals.flipX_switch_preview = flipX_switch

    # Add the "Flip Y" switch
    def toggle_flipY():
        is_flipY = modules.globals.flipY_var.get()
        modules.globals.flip_y = is_flipY
        refresh_preview() # <--- REFRESH ADDED

    flipY_switch = ctk.CTkSwitch(button_row, text=' Flip Y', 
                                 variable=modules.globals.flipY_var, cursor='hand2',
                                 command=toggle_flipY)
    flipY_switch.pack(side='left', padx=5, pady=5)
    modules.globals.flipY_switch_preview = flipY_switch

    # Function to handle rotation range changes
    def update_rotation_range(size):
        modules.globals.face_rot_range = int(size)
        modules.globals.rot_range_dropdown_preview.set(size)
        refresh_preview() # <--- REFRESH ADDED

    face_rot_label = ctk.CTkLabel(button_row, text=" | Rot Range ", font=("Arial", 16))
    face_rot_label.pack(side='left', padx=5, pady=5)

    # --- AUTO ROTATE SWITCH (PREVIEW) ---
    def toggle_auto_rotate_preview():
        is_auto = modules.globals.auto_rotate_var.get()
        modules.globals.auto_rotate_value = is_auto
        if modules.globals.auto_rotate_switch_main:
            modules.globals.auto_rotate_switch_main.select() if is_auto else modules.globals.auto_rotate_switch_main.deselect()
        refresh_preview() # <--- REFRESH ADDED
    
    auto_rotate_switch_prev = ctk.CTkSwitch(button_row, text='Auto', 
                                           variable=modules.globals.auto_rotate_var,
                                           command=toggle_auto_rotate_preview,
                                           width=50)
    auto_rotate_switch_prev.pack(side='left', padx=5, pady=5)
    modules.globals.auto_rotate_switch_preview = auto_rotate_switch_prev

    # Initialize and create rotation range dropdown
    rot_range_dropdown_preview = ctk.CTkOptionMenu(button_row, values=["0", "90", "180", "-90"],
                                                   variable=modules.globals.rot_range_var,
                                                   command=update_rotation_range,width=10)
    rot_range_dropdown_preview.pack(side='left', padx=5, pady=5)
    modules.globals.rot_range_dropdown_preview = rot_range_dropdown_preview 

    modules.globals.face_index_range = -1  
    modules.globals.face_index_var = ctk.StringVar(value="-1")

    # Function to handle face range changes
    def update_face_index(size):
        modules.globals.face_index_range = int(size)
        modules.globals.face_index_dropdown_preview.set(size)
        refresh_preview() # <--- REFRESH ADDED

    face_rot_index = ctk.CTkLabel(button_row, text=" | F1 ", font=("Arial", 16))
    face_rot_index.pack(side='left', padx=5, pady=5)

    face_index_dropdown_preview = ctk.CTkOptionMenu(button_row, values=["-1","0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                                                    variable=modules.globals.face_index_var,
                                                    command=update_face_index,width=10)
    face_index_dropdown_preview.pack(side='left', padx=5, pady=5)
    modules.globals.face_index_dropdown_preview = face_index_dropdown_preview 

    modules.globals.face2_index_range = 0  
    modules.globals.face2_index_var = ctk.StringVar(value="0")

    # Function to handle face range changes
    def update_face2_index(size):
        modules.globals.face2_index_range = int(size)
        modules.globals.face2_index_dropdown_preview.set(size)
        refresh_preview() # <--- REFRESH ADDED

    face2_rot_index = ctk.CTkLabel(button_row, text=" | F2 ", font=("Arial", 16))
    face2_rot_index.pack(side='left', padx=5, pady=5)

    f2_state = "normal" if modules.globals.both_faces else "disabled"
    face2_index_dropdown_preview = ctk.CTkOptionMenu(button_row, values=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                                                    variable=modules.globals.face2_index_var,
                                                    command=update_face2_index,width=10,
                                                    state=f2_state)
    face2_index_dropdown_preview.pack(side='left', padx=5, pady=5)
    modules.globals.face2_index_dropdown_preview = face2_index_dropdown_preview 

    # Forehead logic
    def update_forehead_index(size):
        new_float_value = float(size)
        modules.globals.face_forehead_var = new_float_value
        refresh_preview() # <--- REFRESH ADDED

    face_forehead_index = ctk.CTkLabel(button_row, text=" | Forehead ", font=("Arial", 16))
    face_forehead_index.pack(side='left', padx=5, pady=5)

    face_forehead_size_var = ctk.StringVar(value="0.1")
    face_forehead_index_dropdown_preview = ctk.CTkOptionMenu(button_row, values=["0.1","0.2", "0.3", "0.4", "0.5"],
                                                    variable=face_forehead_size_var,
                                                    command=update_forehead_index,width=10)
    face_forehead_index_dropdown_preview.pack(side='left', padx=5, pady=5)

    # Hotkeys label
    hotkey_text = "Hotkeys: Auto Track (A) | Reset Track (T) | Mouth Mask (M) | Mask Face 1-10 (1-0)"
    hotkey_index = ctk.CTkLabel(switch_frame, text=hotkey_text, font=("Arial", 12), text_color="gray")
    hotkey_index.pack(side='top', fill='x', padx=10, pady=(0, 5), anchor='w')

    # Slider and Display Container
    preview_slider = ctk.CTkSlider(preview, from_=0, to=0, command=lambda frame_value: update_preview(frame_value))
    preview_slider.pack(fill='x', side='bottom', padx=10, pady=5)
    preview_slider.pack_forget() 

    display_frame = ctk.CTkFrame(preview, fg_color="transparent")
    display_frame.pack(fill='both', expand=True)
    display_frame.pack_propagate(False)

    preview_label_cam = ctk.CTkLabel(display_frame, text=None)
    preview_label_cam.pack(fill='both', expand=True)
    
    preview_label = preview_label_cam 

    switch_frame.update() 
    original_height = switch_frame.winfo_height()

    is_switch_frame_visible = True

    def on_double_click(event):
        nonlocal is_switch_frame_visible
        if is_switch_frame_visible:
            switch_frame.pack_propagate(False)
            switch_frame.configure(height=0)
        else:
            switch_frame.pack_propagate(True)
            switch_frame.configure(height=original_height)
        is_switch_frame_visible = not is_switch_frame_visible

    preview.bind("<Double-Button-1>", on_double_click)  
    setup_hotkeys(preview) 

    return preview
    
def update_status(text: str) -> None:
    status_label.configure(text=text)
    ROOT.update()

def update_tumbler(var: str, value: bool) -> None:
    modules.globals.fp_ui[var] = value

def select_source_path() -> None:
    global RECENT_DIRECTORY_SOURCE, img_ft, vid_ft

    PREVIEW.withdraw()
    # PREVIEW_IMAGE.withdraw()
    source_path = ctk.filedialog.askopenfilename(title='select an source image', initialdir=RECENT_DIRECTORY_SOURCE, filetypes=[img_ft])
    if is_image(source_path):
        modules.globals.source_path = source_path
        RECENT_DIRECTORY_SOURCE = os.path.dirname(modules.globals.source_path)
        image = render_image_preview(modules.globals.source_path, (200, 200))
        source_label.configure(image=image)
        if modules.globals.face_tracking:
            clear_face_tracking_data()
    else:
        modules.globals.source_path = None
        source_label.configure(image=None)
        if modules.globals.face_tracking:
            clear_face_tracking_data()

def fliter(*args):
    size = filter_var.get()
    modules.globals.use_pencil_filter=False
    modules.globals.use_ink_filter_white=False
    modules.globals.use_ink_filter_black=False
    modules.globals.use_black_lines=False

    if size=="White Ink":
        modules.globals.use_pencil_filter=False
        modules.globals.use_ink_filter_white=True
        modules.globals.use_ink_filter_black=False
        modules.globals.use_black_lines=True
    if size=="Black Ink":
        modules.globals.use_pencil_filter=False
        modules.globals.use_ink_filter_white=False
        modules.globals.use_ink_filter_black=True
        modules.globals.use_black_lines=False
    if size=="Pencil":
        modules.globals.use_pencil_filter=True
        modules.globals.use_ink_filter_white=False
        modules.globals.use_ink_filter_black=False
        modules.globals.use_black_lines=False
    refresh_preview()

def select_target_path() -> None:
    global RECENT_DIRECTORY_TARGET, img_ft, vid_ft

    PREVIEW.withdraw()
    # PREVIEW_IMAGE.withdraw()
    target_path = ctk.filedialog.askopenfilename(title='select an target image or video', initialdir=RECENT_DIRECTORY_TARGET, filetypes=[img_ft, vid_ft])
    if is_image(target_path):
        modules.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
        image = render_image_preview(modules.globals.target_path, (200, 200))
        target_label.configure(image=image)
        if modules.globals.face_tracking:
            clear_face_tracking_data()
            modules.globals.face_tracking = False
            face_tracking_value.set(False)  # Update the switch state
            pseudo_face_var.set(False)  # Update the switch state
            face_tracking()  # Call face_tracking to update UI elements
    elif is_video(target_path):
        modules.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
        video_frame = render_video_preview(target_path, (200, 200))
        target_label.configure(image=video_frame)
        if modules.globals.face_tracking:
            clear_face_tracking_data()
    else:
        modules.globals.target_path = None
        target_label.configure(image=None)
        if modules.globals.face_tracking:
            clear_face_tracking_data()

def select_output_path(start: Callable[[], None]) -> None:
    global RECENT_DIRECTORY_OUTPUT, img_ft, vid_ft

    if is_image(modules.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(title='save image output file', filetypes=[img_ft], defaultextension='.png', initialfile='output.png', initialdir=RECENT_DIRECTORY_OUTPUT)
    elif is_video(modules.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(title='save video output file', filetypes=[vid_ft], defaultextension='.mp4', initialfile='output.mp4', initialdir=RECENT_DIRECTORY_OUTPUT)
    else:
        output_path = None
    if output_path:
        modules.globals.output_path = output_path
        RECENT_DIRECTORY_OUTPUT = os.path.dirname(modules.globals.output_path)
        start()

def check_and_ignore_nsfw(target, destroy: Callable = None) -> bool:
    ''' Check if the target is NSFW.
    TODO: Consider to make blur the target.
    '''
    from numpy import ndarray
    from modules.predicter import predict_image, predict_video, predict_frame
    if type(target) is str: # image/video file path
        check_nsfw = predict_image if has_image_extension(target) else predict_video
    elif type(target) is ndarray: # frame object
        check_nsfw = predict_frame
    if check_nsfw and check_nsfw(target):
        if destroy: destroy(to_quit=False) # Do not need to destroy the window frame if the target is NSFW
        update_status('Processing ignored!')
        return True
    else: return False


def fit_image_to_size(image, width: int, height: int):
    if width is None and height is None:
      return image
    h, w, _ = image.shape
    ratio_h = 0.0
    ratio_w = 0.0
    if width > height:
        ratio_h = height / h
    else:
        ratio_w = width  / w
    ratio = max(ratio_w, ratio_h)
    new_size = (int(ratio * w), int(ratio * h))
    return cv2.resize(image, dsize=new_size)

def render_image_preview(image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return ctk.CTkImage(image, size=image.size)

def render_video_preview(video_path: str, size: Tuple[int, int], frame_number: int = 0) -> ctk.CTkImage:
    capture = cv2.VideoCapture(video_path)
    if frame_number:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    has_frame, frame = capture.read()
    if has_frame:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return ctk.CTkImage(image, size=image.size)
    capture.release()
    cv2.destroyAllWindows()

def toggle_preview() -> None:
    # Simplified to just handle the main PREVIEW window
    if PREVIEW.state() == 'normal':
        PREVIEW.withdraw()
    elif modules.globals.source_path and modules.globals.target_path:
        init_preview()
        update_preview()

def toggle_preview_cam() -> None:
    if PREVIEW.state() == 'normal':
        PREVIEW.withdraw()


def init_preview() -> None:
    if is_image(modules.globals.target_path):
        preview_slider.pack_forget()
    if is_video(modules.globals.target_path):
        video_frame_total = get_video_frame_total(modules.globals.target_path)
        preview_slider.configure(to=video_frame_total)
        preview_slider.pack(fill='x')
        preview_slider.set(0)


def update_preview(frame_number: int = 0) -> None:
    if modules.globals.source_path and modules.globals.target_path:
        update_status('Processing...')
        
        # (Your previous fix for Image/Video loading)
        if is_image(modules.globals.target_path):
            temp_frame = cv2.imread(modules.globals.target_path)
            # Ensure slider is hidden for images
            if preview_slider: preview_slider.pack_forget() 
        else:
            temp_frame = get_video_frame(modules.globals.target_path, frame_number)
            # Ensure slider is shown for videos
            if preview_slider: preview_slider.pack(fill='x', side='bottom')

        if temp_frame is None:
            update_status('Error: Could not load target file.')
            return

        if modules.globals.nsfw_filter and check_and_ignore_nsfw(temp_frame):
            return
        
        # Initialize variables for the selected face/s image. 
        source_images: List[Face] = []
        if modules.globals.source_path:
            source_image = cv2.imread(modules.globals.source_path)
            faces = get_many_faces(source_image)
            if faces:
                # sort faces from left to right then slice max 10
                source_images = sorted(faces, key=lambda face: face.bbox[0])[:20]

        # --- COPY OF TOOLBAR LOGIC FROM WEBCAM_PREVIEW ---
        # This ensures F1 and F2 dropdowns populate correctly based on source faces
        if source_images:
            num_faces = len(source_images)
            dropdown_values = ["-1"] + [str(i) for i in range(num_faces)]
            dropdown2_values = [str(i) for i in range(num_faces)]
            
            if modules.globals.face_index_dropdown_preview:
                modules.globals.face_index_dropdown_preview.configure(values=dropdown_values)
                # We update the values but keep current selection or default to -1/0
                if modules.globals.face_index_var.get() not in dropdown_values:
                     modules.globals.face_index_var.set("-1")
            
            if modules.globals.face2_index_dropdown_preview:
                modules.globals.face2_index_dropdown_preview.configure(values=dropdown2_values)
                if modules.globals.face2_index_var.get() not in dropdown2_values:
                    modules.globals.face2_index_var.set("0")
        # -------------------------------------------------

        # no face found
        if not source_images:
            print('No face found in source image')
            return
        
        if modules.globals.flip_x:
            temp_frame = cv2.flip(temp_frame, 1)
        if modules.globals.flip_y:
            temp_frame = cv2.flip(temp_frame, 0)

        for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
            temp_frame = frame_processor.process_frame(source_images, temp_frame)
            
        # Use PREVIEW window size instead of fixed constant for better responsiveness
        current_width = PREVIEW.winfo_width() if PREVIEW.winfo_width() > 1 else PREVIEW_MAX_WIDTH
        current_height = PREVIEW.winfo_height() if PREVIEW.winfo_height() > 1 else PREVIEW_MAX_HEIGHT

        temp_frame = fit_image_to_preview(temp_frame, current_width, current_height)
        image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
        
        # Logic to keep image contained within window
        image = ImageOps.contain(image, (current_width, current_height), Image.LANCZOS)
        
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)
        update_status('Processing succeed!')
        
        # USE PREVIEW WINDOW (The one with the toolbar)
        PREVIEW.deiconify()

def webcam_preview():
    if modules.globals.source_path is None:
        return
    
    global preview_label_cam, PREVIEW, ROOT, enhancer_switch, fps_label
    global worker # Global so update_camera_resolution can find it
    global target_face1_value, target_face2_value, target_face3_value, target_face4_value, target_face5_value
    global target_face6_value, target_face7_value, target_face8_value, target_face9_value, target_face10_value

    # --- FACE ENHANCER LOGIC (From your code) ---
    previous_enhancer_state = modules.globals.fp_ui.get('face_enhancer', False)
    if previous_enhancer_state:
        enhancer_switch.deselect()
        update_tumbler('face_enhancer', False)
    if enhancer_switch:
        enhancer_switch.configure(state="disabled")

    # --- RESET GLOBALS (From your code) ---
    modules.globals.first_face_embedding = None
    modules.globals.second_face_embedding = None
    modules.globals.first_face_id = None
    modules.globals.second_face_id = None

    # --- UI SETUP ---
    PREVIEW_WIDTH = 1100
    PREVIEW_HEIGHT = 670
    PREVIEW.deiconify()
    PREVIEW.geometry(f"{PREVIEW_WIDTH}x{PREVIEW_HEIGHT}")
    preview_label_cam.configure(width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT)

    # --- START THE WORKER ---
    worker = LiveSwapWorker()
    worker.start(modules.globals.camera_index)

    # --- CONFIGURE DROPDOWNS (From your code - logic moved here because worker has source_images) ---
    if not worker.source_images:
        print('No face found in source image')
        worker.stop()
        if enhancer_switch:
            enhancer_switch.configure(state="normal")
            if previous_enhancer_state:
                enhancer_switch.select()
                update_tumbler('face_enhancer', True)
        return
    else:
        num_faces = len(worker.source_images)
        dropdown_values = ["-1"] + [str(i) for i in range(num_faces)]
        dropdown2_values = [str(i) for i in range(num_faces)]
        
        modules.globals.face_index_dropdown_preview.configure(values=dropdown_values)
        modules.globals.face2_index_dropdown_preview.configure(values=dropdown2_values)
        modules.globals.face_index_var.set("-1")
        modules.globals.face_index_range = -1
        modules.globals.face2_index_var.set("0")
        modules.globals.face2_index_range = 0

    frame_count = 0
    start_time = time.time()

    # --- UI UPDATE LOOP ---
    def update_ui_loop():
        nonlocal frame_count, start_time

        # 1. Handle Window Close
        if PREVIEW.state() == 'withdrawn':
            worker.stop()
            if enhancer_switch:
                enhancer_switch.configure(state="normal")
                if previous_enhancer_state:
                    enhancer_switch.select()
                    update_tumbler('face_enhancer', True)
            return

        # 2. Get Frame from Worker
        frame = worker.get_frame()

        if frame is not None:
            # 3. FPS Calculation
            frame_count += 1
            current_time = time.time()
            if current_time - start_time > 1:
                fps = frame_count / (current_time - start_time)
                fps_label.configure(text=f'FPS: {fps:.2f}')
                frame_count = 0
                start_time = current_time
            
            # 4. Update Face Tracking Labels (From your code)
            target_face1_value.configure(text=f': {modules.globals.target_face1_score:.2f}')
            target_face2_value.configure(text=f': {modules.globals.target_face2_score:.2f}')
            target_face3_value.configure(text=f': {modules.globals.target_face3_score:.2f}')
            target_face4_value.configure(text=f': {modules.globals.target_face4_score:.2f}')
            target_face5_value.configure(text=f': {modules.globals.target_face5_score:.2f}')
            target_face6_value.configure(text=f': {modules.globals.target_face6_score:.2f}')
            target_face7_value.configure(text=f': {modules.globals.target_face7_score:.2f}')
            target_face8_value.configure(text=f': {modules.globals.target_face8_score:.2f}')
            target_face9_value.configure(text=f': {modules.globals.target_face9_score:.2f}')
            target_face10_value.configure(text=f': {modules.globals.target_face10_score:.2f}')

            # 5. Sync Auto-Rotation Dropdowns
            # If the worker changed the rotation logic, update the UI dropdown to match
            if modules.globals.rot_range_var.get() != str(modules.globals.face_rot_range):
                 modules.globals.rot_range_var.set(str(modules.globals.face_rot_range))
                 if modules.globals.rot_range_dropdown_preview:
                     modules.globals.rot_range_dropdown_preview.set(str(modules.globals.face_rot_range))

            # 6. Display Image
            current_width = PREVIEW.winfo_width()
            current_height = PREVIEW.winfo_height()
            
            display_frame = fit_image_to_preview(frame, current_width, current_height)
            image_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            ctk_img = ctk.CTkImage(image_pil, size=(current_width, current_height))
            preview_label_cam.configure(image=ctk_img)

        # 7. Schedule next update
        PREVIEW.after(10, update_ui_loop)

    update_ui_loop()

def fit_image_to_preview(image, preview_width, preview_height):
    h, w = image.shape[:2]
    aspect_ratio = w / h

    if preview_width / preview_height > aspect_ratio:
        new_height = preview_height
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = preview_width
        new_height = int(new_width / aspect_ratio)

    # CHANGE IS HERE: Changed INTER_LANCZOS4 to INTER_LINEAR
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Create a black canvas of the size of the preview window
    canvas = np.zeros((preview_height, preview_width, 3), dtype=np.uint8)

    # Calculate position to paste the resized image
    y_offset = (preview_height - new_height) // 2
    x_offset = (preview_width - new_width) // 2

    # Paste the resized image onto the canvas
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image

    return canvas

def update_preview_size(*args):
    global PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT, camera
    size = preview_size_var.get().split('x')
    PREVIEW_DEFAULT_WIDTH = int(size[0])
    PREVIEW_DEFAULT_HEIGHT = int(size[1])
    
    
    update_camera_resolution()
    
    # if PREVIEW.state() == 'normal':
    #     update_preview()

def update_camera_resolution():
    global camera, worker, PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT
    
    # 1. Threaded Mode (Fast)
    # If the worker exists and is running, tell IT to change resolution
    if 'worker' in globals() and worker is not None and not worker.stopped:
        worker.set_resolution(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT)
        return

    # 2. Legacy Mode (Old sequential way)
    # Only runs if worker is NOT active
    if camera is not None and camera.isOpened():
        camera_index = modules.globals.camera_index
        camera.release()
        camera = cv2.VideoCapture(camera_index) 
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_DEFAULT_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_DEFAULT_HEIGHT)
        camera.set(cv2.CAP_PROP_FPS, 60)

def both_faces(*args):
    size = both_faces_var.get()
    modules.globals.both_faces = size

    # Enable/Disable Face 2 Dropdown in Preview
    if modules.globals.face2_index_dropdown_preview:
        modules.globals.face2_index_dropdown_preview.configure(state="normal" if size else "disabled")

    modules.globals.many_faces = False
    many_faces_var.set(False)  # Update the many faces switch state

    modules.globals.face_index_range= int(-1)
    modules.globals.face_index_dropdown_preview.set(-1)

    face_index_range=0
    refresh_preview()

def many_faces(*args):
    global face_tracking_value
    size = many_faces_var.get()
    modules.globals.many_faces = size  # Use boolean directly
    if size:  # If many faces is enabled
    #     # Disable face tracking
    #     modules.globals.face_tracking = False
    #     face_tracking_value.set(False)  # Update the switch state

        modules.globals.flip_faces = False
        flip_faces_value.set(False)

        modules.globals.both_faces = False
        both_faces_var.set(False)
        
        # Force disable Face 2 Dropdown since Both Faces is off
        if modules.globals.face2_index_dropdown_preview:
             modules.globals.face2_index_dropdown_preview.configure(state="disabled")

        modules.globals.detect_face_right = False
        detect_face_right_value.set(False)

        modules.globals.face_index_range= int(-1)
        modules.globals.face_index_dropdown_preview.set(-1)

        # pseudo_face_var.set(False)  # Update the switch state
        # face_tracking()  # Call face_tracking to update UI elements
    clear_face_tracking_data()

    refresh_preview()

def face_tracking(*args):
    global pseudo_face_switch, stickiness_dropdown, pseudo_threshold_dropdown, clear_tracking_button,pseudo_face_var
    global many_faces_var,embedding_weight_size_dropdown,weight_distribution_size_dropdown,position_size_dropdown
    global old_embedding_size_dropdown,new_embedding_size_dropdown
    
    
    size = face_tracking_value.get()
    modules.globals.face_tracking = size  # Use boolean directly
    modules.globals.face_tracking_value = size

    # if size:  # If face tracking is enabled
    #     # Disable many faces
    #     modules.globals.many_faces = False
    #     many_faces_var.set(False)  # Update the many faces switch state
    
    # Enable/disable UI elements based on face tracking state
    if size:  # If face tracking is enabled
        pseudo_face_switch.configure(state="normal")
        stickiness_dropdown.configure(state="normal")
        pseudo_threshold_dropdown.configure(state="normal")
        clear_tracking_button.configure(state="normal")
        embedding_weight_size_dropdown.configure(state="normal")
        weight_distribution_size_dropdown.configure(state="normal")
        position_size_dropdown.configure(state="normal")
        old_embedding_size_dropdown.configure(state="normal")
        new_embedding_size_dropdown.configure(state="normal")
        
        modules.globals.face_index_range= int(-1)
        modules.globals.face_index_dropdown_preview.set(-1)
    else:  # If face tracking is disabled
        pseudo_face_switch.configure(state="disabled")
        stickiness_dropdown.configure(state="disabled")
        pseudo_threshold_dropdown.configure(state="disabled")
        clear_tracking_button.configure(state="disabled")
        embedding_weight_size_dropdown.configure(state="disabled")
        weight_distribution_size_dropdown.configure(state="disabled")
        position_size_dropdown.configure(state="disabled")
        old_embedding_size_dropdown.configure(state="disabled")
        new_embedding_size_dropdown.configure(state="disabled")
        pseudo_face_var.set(False)  # Update the switch state


    clear_face_tracking_data()
    refresh_preview()

def mask_size(*args):
    size = mask_size_var.get()
    modules.globals.mask_size = int(size)
    refresh_preview()

def mask_down_size(*args):
    size = mask_down_size_var.get()
    modules.globals.mask_down_size = float(size)
    refresh_preview()

def mask_feather_ratio_size(*args):
    size = mask_feather_ratio_var.get()
    modules.globals.mask_feather_ratio = int(size)
    refresh_preview()

def stickyface_size(*args):
    size = stickyface_var.get()
    modules.globals.sticky_face_value = float(size)
    refresh_preview()
    
def flip_faces(*args):
    size = flip_faces_value.get()
    modules.globals.flip_faces = int(size)
    modules.globals.flip_faces_value = True

    modules.globals.many_faces = False
    many_faces_var.set(False)  # Update the many faces switch state
    
    # Removed reset lines to preserve user selection (e.g., F1=3)
    # modules.globals.face_index_range= int(-1)
    # modules.globals.face_index_dropdown_preview.set(-1)

    if modules.globals.face_tracking:
        clear_face_tracking_data()
    refresh_preview()

def detect_faces_right(*args):
    size = detect_face_right_value.get()
    modules.globals.detect_face_right = int(size)
    modules.globals.detect_face_right_value = True

    modules.globals.many_faces = False
    many_faces_var.set(False)  # Update the many faces switch state
    
    if modules.globals.face_tracking:
        clear_face_tracking_data()
    refresh_preview()


def stickiness_factor_size(*args):
    size = stickyface_var.get()
    modules.globals.sticky_face_value = float(size)
    refresh_preview()

def pseudo_threshold_size(*args):
    size = pseudo_threshold_var.get()
    modules.globals.pseudo_face_threshold = float(size)
    refresh_preview()

def clear_face_tracking_data(*args):
    frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
    for frame_processor in frame_processors:
        if hasattr(frame_processor, 'reset_face_tracking'):
                frame_processor.reset_face_tracking()
    refresh_preview()

def face_rot_size(*args):

    size = rot_range_var.get()
    modules.globals.face_rot_range = int(size)
    refresh_preview()

# --- HOTKEY HANDLERS ---

def setup_hotkeys(window):
    # a - auto face tracking
    window.bind('a', lambda e: toggle_face_tracking_hotkey())
    # t - reset tracking
    window.bind('t', lambda e: reset_face_tracking_hotkey())
    # m - mouth mask global
    window.bind('m', lambda e: toggle_mouth_mask_global_hotkey())
    
    # 1-9, 0 for individual faces
    window.bind('1', lambda e: toggle_mouth_mask_face_hotkey(0))
    window.bind('2', lambda e: toggle_mouth_mask_face_hotkey(1))
    window.bind('3', lambda e: toggle_mouth_mask_face_hotkey(2))
    window.bind('4', lambda e: toggle_mouth_mask_face_hotkey(3))
    window.bind('5', lambda e: toggle_mouth_mask_face_hotkey(4))
    window.bind('6', lambda e: toggle_mouth_mask_face_hotkey(5))
    window.bind('7', lambda e: toggle_mouth_mask_face_hotkey(6))
    window.bind('8', lambda e: toggle_mouth_mask_face_hotkey(7))
    window.bind('9', lambda e: toggle_mouth_mask_face_hotkey(8))
    window.bind('0', lambda e: toggle_mouth_mask_face_hotkey(9))
    

def toggle_face_tracking_hotkey():
    if 'face_tracking_value' in globals():
        val = not face_tracking_value.get()
        face_tracking_value.set(val)
        modules.globals.face_tracking = val
        face_tracking(None) # Update UI state
        update_status(f"Auto Face Tracking: {'On' if val else 'Off'}")
        refresh_preview()

def reset_face_tracking_hotkey():
    clear_face_tracking_data()
    update_status("Face Tracking Reset")
    refresh_preview()

def toggle_mouth_mask_global_hotkey():
    if hasattr(modules.globals, 'mouth_mask_var'):
        val = not modules.globals.mouth_mask_var.get()
        modules.globals.mouth_mask_var.set(val)
        modules.globals.mouth_mask = val
        update_status(f"Mouth Mask: {'On' if val else 'Off'}")
        refresh_preview()

def toggle_mouth_mask_face_hotkey(index):
    modules.globals.mouth_mask_enabled_faces[index] = not modules.globals.mouth_mask_enabled_faces[index]
    state = "On" if modules.globals.mouth_mask_enabled_faces[index] else "Off"
    update_status(f"Mouth Mask Face {index + 1}: {state}")
    refresh_preview()    