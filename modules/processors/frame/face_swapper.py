from typing import Any, List, Optional, Tuple
import os
import cv2  # This is a library for working with images and videos
import insightface  # This is a library for detecting and analyzing faces
import threading  # This helps run parts of the program at the same time
import math # This is for some math functions

import modules.globals # This lets us use settings that apply to the whole program
import modules.processors.frame.core # This is for processing video frames

# These are functions from other parts of the program that help us find faces
from modules.face_analyser import get_one_face, get_many_faces, get_one_face_left, get_one_face_right, get_face_analyser
# These are special types for faces and frames (images or videos)
from modules.typing import Face, Frame
# These help us download models, find files, and check if something is an image or video
from modules.utilities import conditional_download, resolve_relative_path, is_image, is_video
from collections import deque # A special list where items are added to one end and removed from the other
import numpy as np # This is a library for math, especially with arrays
import time # This is for keeping track of time

# This is the thing that actually swaps the faces
FACE_SWAPPER = None
# This is used to make sure only one part of the program changes FACE_SWAPPER at a time
THREAD_LOCK = threading.Lock()
# This is the name of this part of the program
NAME = 'DLC.FACE-SWAPPER'

# How long to wait before swapping faces again (in seconds)
COOLDOWN_PERIOD = 1.0  # 1 second cooldown
# How much to care about the position of a face when deciding if it's the same face
POSITION_WEIGHT = 0.4
# How much to care about what the face looks like (its "embedding")
EMBEDDING_WEIGHT = 0.6
# How much to blur the edges of the face mask
BLUR_AMOUNT = 12
# How many frames a face can be lost for before we stop tracking it (assuming 30 frames per second, this is 60 seconds)
MAX_LOST_COUNT = 1800  # Assuming 30 fps, this is 60 seconds
# How "sticky" the tracking is, meaning how likely it is to stick with the same face
STICKINESS_FACTOR = 0.8  # Adjust this to change how "sticky" the tracking is

# How many frames in a row the first face has been lost
first_face_lost_count: int = 0
# How many frames in a row the second face has been lost
second_face_lost_count: int = 0
# A list that remembers the last 30 positions of a face
face_position_history = deque(maxlen=30)  # Stores last 30 positions
# When the last face swap happened
last_swap_time = 0

# The "ID" of the first face we are tracking
first_face_id: Optional[int] = None
# The "ID" of the second face we are tracking
second_face_id: Optional[int] = None
# A list of how the source faces look (their "embeddings")
source_face_embeddings: List[np.ndarray] = []
# How the first tracked face looks
first_face_embedding: Optional[np.ndarray] = None
# How the second tracked face looks
second_face_embedding: Optional[np.ndarray] = None
# Where the first tracked face is
first_face_position: Optional[Tuple[float, float]] = None
# Where the second tracked face is
second_face_position: Optional[Tuple[float, float]] = None
# How much the face position can change before we think it's a different face
position_threshold = 0.2  # Adjust this value to change sensitivity to position changes
# When we last assigned faces to the tracked faces
last_assignment_time: float = 0
# How long to wait before assigning faces again
assignment_cooldown: float = 1.0  # 1 second cooldown

# How many frames in a row a face has been lost (used in single-face tracking)
face_lost_count = 0
detection_frame_counter: int = 0
cached_detected_faces: List[Face] = []

def pre_check() -> bool:
    """
    Checks if the models we need are downloaded.
    """
    download_directory_path = resolve_relative_path('../models') # Gets the path to the models folder
    # Checks if the face swap model is downloaded and downloads it if not
    conditional_download(download_directory_path, ['https://huggingface.co/ivideogameboss/iroopdeepfacecam/blob/main/inswapper_128_fp16.onnx'])
    return True

def pre_start() -> bool:
    """
    Checks if the source and target paths are valid.
    """
    from modules.core import update_status # This is for showing messages to the user
    # Checks if the source is an image
    if not is_image(modules.globals.source_path):
        update_status('Select an image for source path.', NAME) # Tells the user to select an image
        return False
    # Checks if there is a face in the source image
    elif not get_one_face(cv2.imread(modules.globals.source_path)):
        update_status('No face in source path detected.', NAME) # Tells the user that there is no face in the source
        return False
    # Checks if the target is an image or a video
    if not is_image(modules.globals.target_path) and not is_video(modules.globals.target_path):
        update_status('Select an image or video for target path.', NAME) # Tells the user to select an image or video
        return False
    return True

def get_face_swapper() -> Any:
    """
    Gets the face swapper model. Loads the model matching the selected swapper_resolution.
    Falls back to inswapper_128_fp16.onnx if the requested resolution model is not found.
    """
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            resolution = getattr(modules.globals, 'swapper_resolution', 128)
            # Try resolution-specific fp16 model first, then plain, then fall back to 128
            candidates = [
                resolve_relative_path(f'../models/inswapper_{resolution}_fp16.onnx'),
                resolve_relative_path(f'../models/inswapper_{resolution}.onnx'),
                resolve_relative_path('../models/inswapper_128_fp16.onnx'),
            ]
            model_path = next((p for p in candidates if os.path.isfile(p)), candidates[-1])
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=modules.globals.execution_providers)
    return FACE_SWAPPER


def reset_face_swapper() -> None:
    """
    Clears the cached face swapper so it will be reloaded on next use.
    Call this when swapper_resolution changes.
    """
    global FACE_SWAPPER
    with THREAD_LOCK:
        FACE_SWAPPER = None

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """
    Swaps the source face onto the target face in the given frame.
    """
    face_swapper = get_face_swapper() # Gets the face swapper model

    # Apply the face swap
    swapped_frame = face_swapper.get(temp_frame, target_face, source_face, paste_back=True)

    # Create a mask for the target face
    target_mask = create_face_mask(target_face, temp_frame)

    # Blur the edges of the mask
    blurred_mask = blur_edges(target_mask)
    blurred_mask = blurred_mask / 255.0 # Makes the mask values between 0 and 1

    # Ensure the mask has 3 channels to match the frame
    blurred_mask_3channel = np.repeat(blurred_mask[:, :, np.newaxis], 3, axis=2)

    # Blend the swapped face with the original frame using the blurred mask
    blended_frame = (swapped_frame * blurred_mask_3channel +
                     temp_frame * (1 - blurred_mask_3channel))

    return blended_frame.astype(np.uint8) # Converts the blended frame back to a regular image

def _rotate_frame(frame: Frame, rotation_value: int) -> Frame:
    """Rotates the frame based on the rotation value and its inverse."""
    if rotation_value == -90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) # Rotates the frame 90 degrees clockwise
    elif rotation_value == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) # Rotates the frame 90 degrees counterclockwise
    elif rotation_value == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)  # 180 is its own inverse
    elif rotation_value == -180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame

def _detect_faces(frame: Frame) -> List[Face]:
    """Detects faces in the given frame, returns an empty list if detection fails."""
    face_analyser = get_face_analyser() # Gets the face analyzer model
    try:
        return face_analyser.get(frame) # Tries to detect faces and returns them as a list
    except Exception as e:
        print(f"Error detecting faces: {e}") # If there's a problem, prints an error message
        return [] # If there's an error, returns an empty list

def _select_target_faces(all_faces: List[Face]) -> List[Face]:
    """Selects which faces to process based on global settings."""
    if not all_faces: # If no faces were detected, returns an empty list
        return []

    if modules.globals.many_faces: # If we're swapping many faces, returns all the faces
        return sorted(all_faces, key=lambda face: face.bbox[0])
    elif modules.globals.both_faces: # If we're swapping two faces
        if modules.globals.detect_face_right: # If we're swapping the rightmost faces
            return sorted(all_faces, key=lambda face: -face.bbox[0])[:2] # Returns the two rightmost faces
        else:
            return sorted(all_faces, key=lambda face: face.bbox[0])[:2] # Returns the two leftmost faces
    else:  # Single face
        if modules.globals.face_tracking: # If we're tracking a single face
            if first_face_embedding is None: # If we haven't started tracking yet
                if modules.globals.detect_face_right: # If we want to track the rightmost face
                    return [max(all_faces, key=lambda face: face.bbox[0])] if all_faces else []
                else:
                    return [min(all_faces, key=lambda face: face.bbox[0])] if all_faces else []
            else: # If we have started tracking
                best_match = find_best_match(first_face_embedding, all_faces) # Finds the face that looks most like the one we're tracking
                return [best_match] if best_match else []
        else: # If we're not tracking, just use the leftmost or rightmost face
            if modules.globals.detect_face_right:
                return [max(all_faces, key=lambda face: face.bbox[0])] if all_faces else []
            else:
                return [min(all_faces, key=lambda face: face.bbox[0])] if all_faces else []

def _limit_target_faces(target_faces: List[Face]) -> List[Face]:
    """Limits the number of faces to process based on settings."""
    if modules.globals.many_faces:
        return target_faces  # No limit when many_faces is enabled
    max_faces = 2 if modules.globals.both_faces else 1 # If we're doing two faces, limit to 2, otherwise 1
    return target_faces[:max_faces]

def _compute_mouth_masks(target_faces: List[Face], frame: Frame) -> List[Tuple[np.ndarray, np.ndarray, tuple, tuple]]:
    mouth_masks = []
    face_masks = []
    
    if modules.globals.mouth_mask:
        for i, face in enumerate(target_faces):
            # Check Granular Mask Control for 'Both Faces' mode
            # i=0 is Left, i=1 is Right (because target_faces are sorted L->R)
            process_this_face = True
            
            if modules.globals.both_faces:
                option = modules.globals.mask_target_option # "Both", "Left", "Right"
                if option == "Left" and i == 1: # Skip Right Face
                    process_this_face = False
                elif option == "Right" and i == 0: # Skip Left Face
                    process_this_face = False
            
            # Check individual face toggle (Hotkeys 1-0)
            if i < len(modules.globals.mouth_mask_enabled_faces):
                if not modules.globals.mouth_mask_enabled_faces[i]:
                    process_this_face = False

            if process_this_face:
                face_mask = create_face_mask(face, frame)
                face_masks.append(face_mask)
                
                # Create the Hybrid Mask
                mouth_mask, mouth_cutout, mouth_box, viz_masks = create_hybrid_mouth_mask(face, frame)
                
                # Append real data
                mouth_masks.append((mouth_mask, mouth_cutout, mouth_box, viz_masks))
            else:
                # Append empty data so the indices stay aligned
                face_masks.append(None)
                mouth_masks.append((None, None, None, None))

    return mouth_masks, face_masks

def _get_source_index(i: int, source_face: List[Face], source_face_order: List[int]) -> int:
    """Determines which source face to use based on current settings and index."""
    if modules.globals.both_faces and len(source_face) > 1: # If we're swapping two faces and have two source faces
        return source_face_order[i % 2] # Use either the first or second source face depending on the index
    else:
        return i % len(source_face) # If we only have one source face, always use that one

def _process_face_swap(frame: Frame, source_face: List[Face], target_face: Face, source_index: int) -> Frame:
    """Performs face swapping and masking on a single face."""
    # Crop the face region
    cropped_frame, crop_info = crop_face_region(frame, target_face, modules.globals.crop_padding) # Crops out the face region

    # --- FIX START: Check for empty crop to prevent crash ---
    if cropped_frame is None or cropped_frame.size == 0 or cropped_frame.shape[0] == 0 or cropped_frame.shape[1] == 0:
        return frame
    # --- FIX END ---

    # Adjust the face bbox for the cropped frame
    adjusted_target_face = create_adjusted_face(target_face, crop_info) # Adjust the face information to the new cropped frame
    # Perform face swapping on the cropped region
    swapped_region = swap_face(source_face[source_index], adjusted_target_face, cropped_frame) # Swaps the faces
    # Create a mask for blending with blurred edges
    mask = create_edge_blur_mask(swapped_region.shape, blur_amount=modules.globals.blur_amount) # Creates a mask with feathered edges
    # Blend the swapped region with the original cropped region
    blended_region = blend_with_mask(swapped_region, cropped_frame, mask) # Blends the swapped face onto the original face
    # Paste the swapped region back into the original frame
    x, y, w, h = crop_info # Gets the original position of the face
    frame[y:y + h, x:x + w] = blended_region # Puts the blended region back into the original frame
    return frame

def _apply_mouth_masks(frame: Frame, target_faces: List[Face], mouth_masks: List[Tuple[np.ndarray, np.ndarray, tuple, tuple]], face_masks: List[np.ndarray]) -> Frame:
    """Applies mouth masks and draws colored visualizations if debug is enabled."""
    if modules.globals.mouth_mask:
        for i, (mouth_mask, mouth_cutout, mouth_box, viz_masks) in enumerate(mouth_masks):
            if mouth_mask is None or mouth_cutout is None:
                continue

            min_x, min_y, max_x, max_y = mouth_box
            
            # 1. Apply the standard mask swap (The logic remains the same)
            if max_x <= frame.shape[1] and max_y <= frame.shape[0]:
                roi = frame[min_y:max_y, min_x:max_x]
                if roi.shape[:2] == mouth_cutout.shape[:2]:
                    # Resize cutout if shapes don't match (dynamic bounds)
                    pass 
                else:
                    mouth_cutout = cv2.resize(mouth_cutout, (roi.shape[1], roi.shape[0]))

                color_corrected_mouth = apply_color_transfer(mouth_cutout, roi)
                region_mask = mouth_mask[min_y:max_y, min_x:max_x]
                if len(region_mask.shape) == 2:
                    region_mask = region_mask[:, :, np.newaxis]
                alpha = region_mask.astype(float) / 255.0
                blended = (color_corrected_mouth * alpha + roi * (1.0 - alpha)).astype(np.uint8)
                frame[min_y:max_y, min_x:max_x] = blended
            
            # 2. VISUALIZATION LOGIC (Colors!)
            if modules.globals.show_mouth_mask_box and viz_masks[0] is not None:
                smart_mask_full, geo_mask_full = viz_masks
                
                # Create a copy to draw shapes on
                overlay = frame.copy()
                
                # A. Draw GEOMETRIC Mask (Padding/Top) in BLUE
                # This shows what the sliders are doing
                overlay[geo_mask_full > 0] = (255, 0, 0) # Blue BGR
                
                # B. Draw SMART Mask (Tongue) in RED
                # This shows what the color detection is doing
                overlay[smart_mask_full > 0] = (0, 0, 255) # Red BGR
                
                # C. Draw COMBINED Feathering in WHITE
                # We find the difference between the feathered mask and a hard threshold
                # to visualize the "blur" zone
                _, hard_thresh = cv2.threshold(mouth_mask, 1, 255, cv2.THRESH_BINARY)
                feather_zone = cv2.absdiff(mouth_mask, hard_thresh)
                overlay[feather_zone > 20] = (255, 255, 255) # White glow for feather
                
                # Blend overlay with original frame (Transparency)
                # 0.5 opacity
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                
                # Add Outlines for clarity
                contours_geo, _ = cv2.findContours(geo_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours_geo, -1, (255, 0, 0), 2) # Blue Border
                
                contours_smart, _ = cv2.findContours(smart_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours_smart, -1, (0, 0, 255), 2) # Red Border

                # Add Text Labels
                cv2.putText(frame, "PAD/TOP (Chin)", (min_x, max_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, "AUTO (Tongue)", (min_x, min_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame

def _process_face_tracking_single(
    frame: Frame, source_face: List[Face], target_face: Face, active_source_index: int
) -> Frame:
    """Handles single-face tracking logic using distance and embedding matching."""
    global first_face_embedding, first_face_position, first_face_id, face_lost_count
    global face_position_history, last_swap_time
    
    current_time = time.time()
    target_embedding = extract_face_embedding(target_face)
    target_position = get_face_center(target_face)
    face_id = id(target_face)

    if first_face_embedding is None:
        # Initialization
        first_face_embedding = target_embedding
        first_face_position = target_position
        first_face_id = face_id
        face_lost_count = 0
        face_position_history.clear()
        face_position_history.append(target_position)
        last_swap_time = current_time
        return _process_face_swap(frame, source_face, target_face, active_source_index)
    
    else:
        best_match_score = 0
        best_match_face = None
        avg_position = np.mean(face_position_history, axis=0) if face_position_history else first_face_position

        detected_faces = _detect_faces(frame)
        
        if not detected_faces: # If no faces detected, increment lost count
            face_lost_count +=1
            if face_position_history and modules.globals.use_pseudo_face and best_match_score < modules.globals.pseudo_face_threshold:
                avg_position = np.mean(face_position_history, axis=0)
                pseudo_face = create_pseudo_face(avg_position)
                return _process_face_swap(frame, source_face, pseudo_face, active_source_index)
            else:
                 return frame
             
        for face in detected_faces:
            
            target_embedding = extract_face_embedding(face)
            target_position = get_face_center(face)

            # Calculate embedding similarity
            embedding_similarity = cosine_similarity(first_face_embedding, target_embedding)
            # Calculate position consistency score 
            position_consistency = 1 / (1 + np.linalg.norm(np.array(target_position) - np.array(avg_position)) + 1e-6)

            # Calculate total match score
            EMBEDDING_WEIGHT = modules.globals.embedding_weight_size
            POSITION_WEIGHT = modules.globals.position_size
            total = modules.globals.old_embedding_weight + modules.globals.new_embedding_weight
            OLD_WEIGHT = modules.globals.old_embedding_weight / total
            NEW_WEIGHT = modules.globals.new_embedding_weight / total
            TOTAL_WEIGHT = EMBEDDING_WEIGHT * modules.globals.weight_distribution_size + POSITION_WEIGHT
            match_score = ((EMBEDDING_WEIGHT * embedding_similarity +
                            POSITION_WEIGHT * position_consistency) / TOTAL_WEIGHT)
            
            if id(face) == first_face_id:
                match_score *= (1 + STICKINESS_FACTOR)

            if match_score > best_match_score:
                best_match_score = match_score
                best_match_face = face

        modules.globals.target_face1_score = best_match_score

        if best_match_face is not None and best_match_score > modules.globals.sticky_face_value:
            face_lost_count = 0
            
            # Update the embedding using weighted average
            first_face_embedding = OLD_WEIGHT * first_face_embedding + NEW_WEIGHT * extract_face_embedding(best_match_face)
            first_face_position = get_face_center(best_match_face)
            first_face_id = id(best_match_face)
            face_position_history.append(first_face_position)
            
            return _process_face_swap(frame, source_face, best_match_face, active_source_index)
        else:
            face_lost_count += 1
            if face_position_history and modules.globals.use_pseudo_face and best_match_score < modules.globals.pseudo_face_threshold:
                avg_position = np.mean(face_position_history, axis=0)
                pseudo_face = create_pseudo_face(avg_position)
                return _process_face_swap(frame, source_face, pseudo_face, active_source_index)

    return frame


def _process_face_tracking_both(
    frame: Frame, source_face: List[Face], target_face: Face, source_index: int, source_face_order: List[int]
) -> Frame:
    """Handles two-face tracking logic with flickering reduction, focusing on smoothing."""
    global first_face_embedding, second_face_embedding, first_face_position, second_face_position
    global first_face_id, second_face_id
    if 'first_face_position_history' not in globals():
        globals()['first_face_position_history'] = deque(maxlen=30)
    if 'second_face_position_history' not in globals():
        globals()['second_face_position_history'] = deque(maxlen=30)
    first_face_position_history = globals()['first_face_position_history']
    second_face_position_history = globals()['second_face_position_history']

    target_embedding = extract_face_embedding(target_face)
    target_position = get_face_center(target_face)
    face_id = id(target_face)
    
    # Store tracked face data as a dictionary
    # Globals for bounding boxes (Need to persist these)
    global first_face_bbox, second_face_bbox
    if 'first_face_bbox' not in globals(): globals()['first_face_bbox'] = None
    if 'second_face_bbox' not in globals(): globals()['second_face_bbox'] = None

    # Store tracked face data as a dictionary
    tracked_faces = {
        0: {
            "embedding": first_face_embedding,
            "position": first_face_position,
            "bbox": first_face_bbox,
            "id": first_face_id,
            "history": first_face_position_history
        },
        1: {
            "embedding": second_face_embedding,
            "position": second_face_position,
            "bbox": second_face_bbox,
            "id": second_face_id,
            "history": second_face_position_history
        },
    }
    
    # 1. ATTEMPT MATCH AGAINST EXISTING TRACKS
    best_match_score = -1
    best_match_index = -1
    
    # --- FAST PATH: IOU CHECK ---
    # If the face overlaps significantly with a known tracker, skip the heavy math.
    iou_threshold = 0.50  # 50% overlap means it's likely the same face
    
    for track_id, track_data in tracked_faces.items():
        if track_data["bbox"] is not None:
            iou = calculate_iou(target_face.bbox, track_data["bbox"])
            if iou > iou_threshold:
                # High confidence match based on position
                best_match_index = track_id
                best_match_score = 100.0 # Force high score
                break
    
    # --- ROBUST PATH: EMBEDDING CHECK ---
    # If IOU didn't find a match (face moved fast, or first frame), do the heavy lifting
    if best_match_index == -1:
        EMBEDDING_WEIGHT = modules.globals.embedding_weight_size
        POSITION_WEIGHT = modules.globals.position_size
        total = modules.globals.old_embedding_weight + modules.globals.new_embedding_weight
        OLD_WEIGHT = modules.globals.old_embedding_weight / total
        NEW_WEIGHT = modules.globals.new_embedding_weight / total
        TOTAL_WEIGHT = EMBEDDING_WEIGHT * modules.globals.weight_distribution_size + POSITION_WEIGHT
    
        for track_id, track_data in tracked_faces.items():
            track_embedding = track_data["embedding"]
            track_position = track_data["position"]
            track_history = track_data["history"]
            
            if track_embedding is not None:
                similarity = cosine_similarity(track_embedding, target_embedding)
                position_consistency = 1 / (1 + np.linalg.norm(np.array(target_position) - np.array(np.mean(track_history, axis=0) if track_history else track_position))) if track_position is not None else 0
                
                score = ((EMBEDDING_WEIGHT * similarity +
                            POSITION_WEIGHT * position_consistency) / TOTAL_WEIGHT)
                
                if track_data["id"] == face_id:
                    score *= (1 + STICKINESS_FACTOR)
                    
                if score > best_match_score:
                    best_match_score = score
                    best_match_index = track_id
    else:
        # Define weights if we took fast path, so the update logic below works
        total = modules.globals.old_embedding_weight + modules.globals.new_embedding_weight
        OLD_WEIGHT = modules.globals.old_embedding_weight / total
        NEW_WEIGHT = modules.globals.new_embedding_weight / total

    matched_track_id = -1
    use_pseudo_face = False

    # 2. DECIDE: MATCH, INIT, OR PSEUDO
    if best_match_index != -1 and best_match_score > modules.globals.sticky_face_value:
        # Match found
        matched_track_id = best_match_index
        
        # Update Track
        # Update Track
        tracked_face = tracked_faces[matched_track_id]
        
        # Only do heavy embedding updates if we didn't use the fast path (score 100 means fast path)
        if best_match_score != 100.0:
             tracked_face["embedding"] = OLD_WEIGHT * tracked_face["embedding"] + NEW_WEIGHT * target_embedding
        
        # Always update position and bbox
        avg_position = np.mean(tracked_face["history"], axis=0) if tracked_face["history"] else tracked_face["position"]
        if avg_position is not None:
            tracked_face["position"] = np.array(avg_position) * 0.8 + np.array(target_position) * 0.2
        else:
            tracked_face["position"] = np.array(target_position)
            
        tracked_face["bbox"] = target_face.bbox # SAVE THE BOX FOR IOU
        tracked_face["id"] = face_id
        tracked_face["history"].append(target_position)
        
        # Sync Globals
        if matched_track_id == 0:
            first_face_bbox = target_face.bbox
        else:
            second_face_bbox = target_face.bbox
        
        # Update Scores Global
        if matched_track_id == 0:
            modules.globals.target_face1_score = best_match_score
        else:
            modules.globals.target_face2_score = best_match_score

    else:
        # No match, try to initialize an empty track
        if first_face_embedding is None:
            # Init Track 0
            matched_track_id = 0
            first_face_embedding = target_embedding
            first_face_position = target_position
            first_face_id = face_id
            first_face_position_history.append(target_position)
        elif second_face_embedding is None:
            # Init Track 1
            matched_track_id = 1
            second_face_embedding = target_embedding
            second_face_position = target_position
            second_face_id = face_id
            second_face_position_history.append(target_position)
        elif modules.globals.use_pseudo_face and best_match_score < modules.globals.pseudo_face_threshold:
            # Both full, pseudo face fallback
            use_pseudo_face = True
            matched_track_id = best_match_index # Use best bad match for pseudo location
    
    # 3. DETERMINE SOURCE INDEX (Applying Flip Logic)
    if matched_track_id != -1:
        # Resolve explicit user selections
        selection_1 = modules.globals.face_index_range
        if selection_1 == -1: selection_1 = 0
        
        selection_2 = 0
        if hasattr(modules.globals, 'face2_index_range'):
            selection_2 = modules.globals.face2_index_range
        
        final_idx = 0
        
        # Apply Logic:
        # Track 0 (Left/F1) -> Normal: Sel1, Flip: Sel2
        # Track 1 (Right/F2) -> Normal: Sel2, Flip: Sel1
        
        if matched_track_id == 0:
            if modules.globals.flip_faces:
                final_idx = selection_2
            else:
                final_idx = selection_1
        elif matched_track_id == 1:
            if modules.globals.flip_faces:
                final_idx = selection_1
            else:
                final_idx = selection_2

        # 4. SWAP
        if use_pseudo_face:
            # Generate pseudo face at track location
            track_hist = tracked_faces[matched_track_id]["history"]
            track_pos = tracked_faces[matched_track_id]["position"]
            avg_pos = np.mean(track_hist, axis=0) if track_hist else track_pos
            pseudo_face = create_pseudo_face(avg_pos)
            return _process_face_swap(frame, source_face, pseudo_face, final_idx)
        else:
            return _process_face_swap(frame, source_face, target_face, final_idx)
            
    return frame

def _process_face_tracking_many(
    frame: Frame, source_face: List[Face], target_face: Face, source_index: int, source_face_order: List[int]
) -> Frame:
    """Handles multi-face tracking logic for up to 10 faces with dynamic score updates."""
    global first_face_embedding, second_face_embedding, first_face_position, second_face_position
    global first_face_id, second_face_id
    
    if 'tracked_faces_many' not in globals():
        globals()['tracked_faces_many'] = {}
    tracked_faces_many = globals()['tracked_faces_many']
    
    if 'position_histories_many' not in globals():
         globals()['position_histories_many'] = {}
    position_histories_many = globals()['position_histories_many']
    
    target_embedding = extract_face_embedding(target_face)
    target_position = get_face_center(target_face)
    face_id = id(target_face)
    use_pseudo_face = False

    
    
    best_match_score = -1
    best_match_key = None
        
    for key, track_data in tracked_faces_many.items():
           
            track_embedding = track_data.get("embedding")
            track_position = track_data.get("position")
            track_history = position_histories_many.get(key, deque(maxlen=30))
           
            if track_embedding is not None and track_position is not None:
               
                similarity = cosine_similarity(track_embedding, target_embedding)
                position_consistency = 1 / (1 + np.linalg.norm(np.array(target_position) - np.array(np.mean(track_history, axis=0) if track_history else track_position))) if track_position is not None else 0
                
                EMBEDDING_WEIGHT = modules.globals.embedding_weight_size
                POSITION_WEIGHT = modules.globals.position_size
                total = modules.globals.old_embedding_weight + modules.globals.new_embedding_weight
                OLD_WEIGHT = modules.globals.old_embedding_weight / total
                NEW_WEIGHT = modules.globals.new_embedding_weight / total
                TOTAL_WEIGHT = EMBEDDING_WEIGHT * modules.globals.weight_distribution_size + POSITION_WEIGHT
                
                score = ((EMBEDDING_WEIGHT * similarity +
                            POSITION_WEIGHT * position_consistency) / TOTAL_WEIGHT)
                if track_data.get("id") == face_id:
                    score *= (1 + STICKINESS_FACTOR)
               
                if score > best_match_score:
                    best_match_score = score
                    best_match_key = key


    if best_match_key is not None and best_match_score > modules.globals.sticky_face_value:
           
        tracked_face = tracked_faces_many[best_match_key]
        track_history = position_histories_many[best_match_key]

        # Update the tracked face with a weighted average of the new embedding
        tracked_face["embedding"] =  OLD_WEIGHT * tracked_face["embedding"] + NEW_WEIGHT * target_embedding

        #Update position with weighted average
        avg_position = np.mean(track_history, axis=0) if track_history else tracked_face["position"]
        if avg_position is not None:
            tracked_face["position"] = np.array(avg_position) * 0.8 + np.array(target_position) * 0.2
        else:
            tracked_face["position"] = np.array(target_position)
        tracked_face["id"] = face_id
        track_history.append(target_position)
        
        if best_match_key < 10:
            setattr(modules.globals, f"target_face{best_match_key + 1}_score", best_match_score)
        
        source_index = best_match_key % len(source_face) # Correctly get index for multi faces
        
        return _process_face_swap(frame, source_face, target_face, source_index)

    elif modules.globals.use_pseudo_face and best_match_score < modules.globals.pseudo_face_threshold:
           
        use_pseudo_face = True
        avg_position = np.mean(position_histories_many[best_match_key], axis=0) if position_histories_many.get(best_match_key) else tracked_faces_many[best_match_key].get("position") if tracked_faces_many.get(best_match_key) else target_position
        pseudo_face = create_pseudo_face(avg_position)
        source_index = 0 if not source_face else 0 % len(source_face)
        return _process_face_swap(frame, source_face, pseudo_face, source_index)
    else: # If no good match
        
        if len(tracked_faces_many) < 10: # If we have less than 10 tracked faces, lets add the new face if it does not exist
           new_key = len(tracked_faces_many)
           tracked_faces_many[new_key] = {
             "embedding": target_embedding,
             "position": target_position,
             "id": face_id
            }
           position_histories_many[new_key] = deque(maxlen=30)
           position_histories_many[new_key].append(target_position)
           source_index = new_key % len(source_face)
           
           if new_key < 10:
            setattr(modules.globals, f"target_face{new_key + 1}_score", 0.00)
           return _process_face_swap(frame, source_face, target_face, source_index)
        else:
            return frame # If we have max faces then do not update the faces

def process_frame(source_face: List[Face], temp_frame: Frame) -> Frame:
    """Main function to process a single frame."""
    global first_face_embedding, second_face_embedding, first_face_position, second_face_position
    global first_face_id, second_face_id
    global first_face_lost_count, second_face_lost_count
    # Optimization Globals
    global detection_frame_counter, cached_detected_faces

    # Rotate the frame
    temp_frame = _rotate_frame(temp_frame, modules.globals.face_rot_range)

    # --- OPTIMIZATION: DETECTION SKIPPING ---
    # Only run heavy face detection once every 3 frames.
    # On the frames in between, use the cached faces.
    detection_frame_counter += 1
    
    # If the counter is 0 (first run) or it's the 3rd frame, OR we don't have cached faces yet
    if detection_frame_counter % modules.globals.detection_frequency == 0 or not cached_detected_faces:
        all_faces = _detect_faces(temp_frame)
        # Update cache regardless. If no faces found, cache becomes empty.
        cached_detected_faces = all_faces
    else:
        # Use the faces found in the previous frame
        all_faces = cached_detected_faces
    # ----------------------------------------

    # Handle face tracking reset logic
    if modules.globals.face_tracking: # If we're using face tracking
        if modules.globals.detect_face_right_value: # If the "detect right face" button was pressed
            reset_face_tracking() # Reset the face tracking
            modules.globals.detect_face_right_value = False # Set the button to false
        if modules.globals.flip_faces_value: # If the "flip faces" button was pressed
            reset_face_tracking() # Reset the face tracking
            modules.globals.flip_faces_value = False # Set the button to false
    elif modules.globals.face_tracking_value: # If the "turn on face tracking" button was pressed
        reset_face_tracking() # Reset the face tracking
        modules.globals.face_tracking_value = False # Set the button to false

     # Select which faces to process
    if modules.globals.face_tracking and modules.globals.both_faces:
         # If tracking both, we MUST look at ALL faces to find the tracks, 
         # regardless of position.
         if modules.globals.detect_face_right:
             target_faces = sorted(all_faces, key=lambda face: -face.bbox[0])
         else:
             target_faces = sorted(all_faces, key=lambda face: face.bbox[0])
    else:
        # Standard Selection (Strict Limits)
        target_faces = _select_target_faces(all_faces)
        target_faces = _limit_target_faces(target_faces)

    # Pre-compute mouth masks if needed
    # Note: _compute_mouth_masks logic handles variable length lists safely
    mouth_masks, face_masks = _compute_mouth_masks(target_faces, temp_frame)

    # Determine source face order
    active_source_index = 1 if modules.globals.flip_faces else 0 # If we should flip the source faces
    source_face_order = [1, 0] if modules.globals.flip_faces else [0, 1] # If we should flip the source faces

    if modules.globals.many_faces: # If we should swap many faces
         if modules.globals.face_tracking: # If we are tracking faces
            for i, target_face in enumerate(target_faces): # Loop through all the faces
                if modules.globals.face_index_range != -1:
                    source_index= modules.globals.face_index_range
                else: 
                    source_index = i % len(source_face) # Get the index of the source face to use
                temp_frame = _process_face_tracking_many(temp_frame, source_face, target_face, source_index, source_face_order )  # Track many faces
         else: # If we are not tracking faces
            for i, target_face in enumerate(target_faces): # Loop through all the faces
                if modules.globals.face_index_range != -1:
                    source_index= modules.globals.face_index_range
                else: 
                    source_index = i % len(source_face)  # Get the index of the source face to use
                temp_frame = _process_face_swap(temp_frame, source_face, target_face, source_index) # Swap the face
    else:
        # Determine loop range
        if modules.globals.face_tracking and modules.globals.both_faces:
             # Loop over ALL candidates to find the tracks
             loop_range = len(target_faces)
        else:
             # Standard limit
             faces_to_process = 2 if modules.globals.both_faces and len(source_face) > 1 else 1 
             loop_range = min(faces_to_process, len(target_faces))

        for i in range(loop_range):
            if modules.globals.face_index_range != -1:
                source_index= modules.globals.face_index_range
            else: 
                source_index = _get_source_index(i, source_face, source_face_order) # Get the index of the source face to use
            if modules.globals.face_tracking: # If we're tracking faces
                if modules.globals.both_faces: # If we're tracking two faces
                   temp_frame = _process_face_tracking_both(
                        temp_frame, source_face, target_faces[i], source_index, source_face_order # Track both faces
                    )
                else:
                    if modules.globals.face_index_range != -1:
                        active_source_index= modules.globals.face_index_range

                    temp_frame = _process_face_tracking_single(
                        temp_frame, source_face, target_faces[i], active_source_index # Track one face
                    )

            else: # If we're not tracking faces
                if modules.globals.face_index_range != -1:
                    source_index= modules.globals.face_index_range
                if modules.globals.both_faces:
                    # Resolve explicit selections
                    selection_1 = modules.globals.face_index_range
                    if selection_1 == -1: selection_1 = 0
                    
                    selection_2 = 0
                    if hasattr(modules.globals, 'face2_index_range'):
                        selection_2 = modules.globals.face2_index_range
                    
                    final_idx = 0
                    
                    if i == 0: # Left Target
                        if modules.globals.flip_faces:
                            final_idx = selection_2 # Flip: Get F2's selection
                        else:
                            final_idx = selection_1 # Normal: Get F1's selection
                    elif i == 1: # Right Target
                        if modules.globals.flip_faces:
                            final_idx = selection_1 # Flip: Get F1's selection
                        else:
                            final_idx = selection_2 # Normal: Get F2's selection

                    temp_frame = _process_face_swap(temp_frame, source_face, target_faces[i], final_idx)
                else:
                    temp_frame = _process_face_swap(temp_frame, source_face, target_faces[i], source_index) # Swap the faces without
    # Apply mouth masks
    temp_frame = _apply_mouth_masks(temp_frame, target_faces, mouth_masks, face_masks)

    # Draw face boxes and landmarks if enabled
    if modules.globals.show_target_face_box:
        for face in target_faces:
            temp_frame = draw_all_landmarks(temp_frame, face) # Draw the face landmarks

    # Rotate back the frame
    temp_frame = _rotate_frame(temp_frame, -modules.globals.face_rot_range)


    # Apply filters based on settings
    if modules.globals.use_pencil_filter:
      temp_frame = apply_pencil_filter(temp_frame)
    elif modules.globals.use_ink_filter_white:
         modules.globals.use_black_lines=False
         temp_frame = apply_ink_filter(temp_frame)
    elif modules.globals.use_ink_filter_black:
        modules.globals.use_black_lines=True
        temp_frame = apply_ink_filter(temp_frame)

    return temp_frame

def apply_pencil_filter(frame: Frame) -> Frame:
    """
    Applies a basic pencil sketch filter to the input frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    inverted = 255 - gray # Invert the grayscale image
    blurred = cv2.GaussianBlur(inverted, (21, 21), sigmaX=0, sigmaY=0) # Gaussian blur
    
    # Simple division to create the "dodge" effect
    with np.errstate(divide='ignore', invalid='ignore'):
        sketch = (gray / (255 - blurred)) * 255 # Dodge
    sketch[sketch > 255] = 255 # Clamp values

    sketch = sketch.astype(np.uint8) # Convert to uint8
    sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR) # Convert back to BGR

    return sketch

def apply_ink_filter(frame: Frame) -> Frame:
    """
    Applies a basic ink filter with pen-thin lines and no blur.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Directly apply adaptive thresholding without blurring
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 3) # No blur

    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert back to BGR

    # Invert colors for black lines on white background
    if modules.globals.use_black_lines:
        edges = 255 - edges

    return edges


def process_frames(source_path: str, temp_frame_paths: List[str], progress: Any = None) -> None:
    """
    Processes all the frames for a video or a list of images.
    """
    source_face: List[Face] = [] # A list to store the source faces
    if source_path: # If we have a source image
        source_image = cv2.imread(source_path) # Load the source image
        faces = get_many_faces(source_image) # Detect the faces in the source image
        if faces: # If there are any faces
            source_face = sorted(faces, key=lambda face: face.bbox[0])[:10] # Sort the faces from left to right and take the first 10

    for temp_frame_path in temp_frame_paths: # Loop through all the frames
        temp_frame = cv2.imread(temp_frame_path) # Load the current frame
        try:
            if modules.globals.flip_x: # If we should flip the frame horizontally
                temp_frame = cv2.flip(temp_frame, 1) # Flip it
            if modules.globals.flip_y: # If we should flip the frame vertically
                temp_frame = cv2.flip(temp_frame, 0) # Flip it
            result = process_frame(source_face, temp_frame) # Process the current frame
            cv2.imwrite(temp_frame_path, result) # Save the processed frame
        except Exception as exception:
            print(exception) # If there's an error, print it
            pass
        if progress:
            progress.update(1) # Update the progress bar

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """
    Processes a single image.
    """
    source_face: List[Face] = [] # A list to store the source faces
    if source_path: # If we have a source image
        source_image = cv2.imread(source_path) # Load the source image
        faces = get_many_faces(source_image) # Detect the faces in the source image
        if faces: # If there are any faces
            source_face = sorted(faces, key=lambda face: face.bbox[0])[:10] # Sort the faces from left to right and take the first 10
    target_frame = cv2.imread(target_path) # Load the target image

    if modules.globals.flip_x: # If we should flip the frame horizontally
        target_frame = cv2.flip(target_frame, 1) # Flip it
    if modules.globals.flip_y: # If we should flip the frame vertically
        target_frame = cv2.flip(target_frame, 0) # Flip it

    result = process_frame(source_face, target_frame) # Process the image
    cv2.imwrite(output_path, result) # Save the processed image

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """
    Processes a video.
    """
    if modules.globals.face_tracking: # If we're tracking faces
        reset_face_tracking() # Reset the face tracking
    modules.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames) # Use the built-in video processing function

def create_face_mask(face: Face, frame: Frame) -> np.ndarray:
    """
    Creates a mask of the face.
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8) # Create a black mask with the same size as the frame
    landmarks = face.landmark_2d_106 # Get the face landmarks
    if landmarks is not None: # If landmarks exist
        face_outline_indices = [1, 43, 48, 49, 104, 105, 17, 25, 26, 27, 28, 29, 30, 31, 32, 18, 19, 20, 21, 22, 23, 24, 0, 8,
                                7, 6, 5, 4, 3, 2, 16, 15, 14, 13, 12, 11, 10, 9, 1] # These numbers represent parts of the face
        face_outline_landmarks = landmarks[face_outline_indices].astype(np.float32) # Get the points for the outline of the face
        left_eyebrow_top = landmarks[49].astype(np.float32) # Get the top of the left eyebrow
        right_eyebrow_top = landmarks[104].astype(np.float32) # Get the top of the right eyebrow
        face_height = np.abs(np.mean([left_eyebrow_top[1], right_eyebrow_top[1]]) -
                            np.max(face_outline_landmarks[:, 1])) # Calculate the height of the face
        forehead_padding = int(face_height * modules.globals.face_forehead_var) # Calculate how much padding we should add for the forehead
        forehead_points = [] # Create a list to store the forehead points
        for point in face_outline_landmarks: # Loop through all the points of the face outline
            x, y = point
            if y < face_outline_landmarks[:, 1].mean(): # If this point is on the forehead
                y_offset = forehead_padding
                forehead_points.extend([ # Add points to make the forehead bigger
                    [x - 5, y - y_offset],
                    [x, y - y_offset],
                    [x + 5, y - y_offset]
                ])
        all_points = np.vstack([face_outline_landmarks, np.array(forehead_points)]) # Add the forehead points to the face outline points
        face_center = np.mean(all_points, axis=0) # Calculate the center of the face
        face_width = np.linalg.norm(
            face_outline_landmarks[0] - face_outline_landmarks[len(face_outline_indices) // 2]
        ) # Calculate the width of the face
        padding = int(face_width * 0.05) # Calculate how much padding to add to the face
        hull = cv2.convexHull(all_points.astype(np.int32)) # Find the outline that contains all the points
        hull_padded = [] # Create a list to store the padded points

        for point in hull: # Loop through the points of the outline
            x, y = point[0]
            direction = np.array([x, y]) - face_center # Calculate the direction from the center to the point
            direction_norm = np.linalg.norm(direction) # Get the length of this vector
            if direction_norm > 0: # If the point is not at the center
                direction = direction / direction_norm # Make the vector have a length of 1
                current_padding = padding # Set the padding for this point
                if y < all_points[:, 1].mean(): # If this is a forehead point
                    current_padding *= 1.5 # Increase the padding
                padded_point = np.array([x, y]) + direction * current_padding # Calculate the padded point
                hull_padded.append(padded_point) # Add the padded point to the list

        hull_padded = np.array(hull_padded, dtype=np.int32) # Convert the list of padded points to a numpy array
        cv2.fillConvexPoly(mask, hull_padded, 255) # Fill the mask with white using the padded points
        mask = cv2.GaussianBlur(mask, (5, 5), 3) # Blur the edges of the mask
    return mask

def blur_edges(mask: np.ndarray, blur_amount: int = 40) -> np.ndarray:
    """
    Blurs the edges of the mask.
    """
    blur_amount = blur_amount if blur_amount % 2 == 1 else blur_amount + 1 # Make sure blur amount is an odd number
    return cv2.GaussianBlur(mask, (blur_amount, blur_amount), 0) # Blur the mask

def apply_color_transfer(source, target):
    """
    Apply color transfer from target to source image
    """
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32") # Change the colorspace of source image to lab color
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32") # Change the colorspace of target image to lab color

    source_mean, source_std = cv2.meanStdDev(source) # Calculate the average and variation of colors in the source image
    target_mean, target_std = cv2.meanStdDev(target) # Calculate the average and variation of colors in the target image

    # Reshape mean and std to be broadcastable
    source_mean = source_mean.reshape(1, 1, 3)
    source_std = source_std.reshape(1, 1, 3)
    target_mean = target_mean.reshape(1, 1, 3)
    target_std = target_std.reshape(1, 1, 3)

    # Perform the color transfer
    source = (source - source_mean) * (target_std / source_std) + target_mean # Change colors in source image to be like target image

    return cv2.cvtColor(np.clip(source, 0, 255).astype("uint8"), cv2.COLOR_LAB2BGR) # Change colorspace back and return the image

def create_feathered_mask(shape, feather_amount):
    """
    Creates a circular mask with blurred edges.
    """
    mask = np.zeros(shape[:2], dtype=np.float32) # Create a black mask with the same size as the frame
    center = (shape[1] // 2, shape[0] // 2) # Find the center of the mask

    # Ensure the feather amount doesn't exceed half the smaller dimension
    max_feather = min(shape[0] // 2, shape[1] // 2) - 1
    feather_amount = min(feather_amount, max_feather)

    # Ensure the axes are at least 1 pixel
    axes = (max(1, shape[1] // 2 - feather_amount),
            max(1, shape[0] // 2 - feather_amount))

    cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1) # Draw a white circle in the mask
    # Ensure the kernel size is odd and at least 3
    kernel_size = max(3, feather_amount * 2 + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0) # Blur the edges of the mask
    return mask / np.max(mask) # Make sure all values are between 0 and 1

def create_mouth_mask(face: Face, frame: Frame) -> (np.ndarray, np.ndarray, tuple):
    """
    Creates a mask for the mouth.
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8) # Create a black mask with the same size as the frame
    mouth_cutout = None # Make a variable to hold the cropped mouth
    landmarks = face.landmark_2d_106 # Get the face landmarks
    if landmarks is not None:
        nose_tip = landmarks[80].astype(np.float32) # Get the tip of the nose
        center_bottom = landmarks[73].astype(np.float32) # Get the bottom of the center of the mouth

        # Recreate mask polygon
        center_to_nose = nose_tip - center_bottom # A vector from the bottom center to the nose
        largest_vector = center_to_nose  # Simplified for this example
        base_height = largest_vector * 0.8
        mask_height = np.linalg.norm(base_height) * modules.globals.mask_size * 0.3 # Calculate the mask height

        mask_top = nose_tip + center_to_nose * 0.2 + np.array([0, +modules.globals.mask_down_size]) # The top of the mask
        mask_bottom = mask_top + center_to_nose * (mask_height / np.linalg.norm(center_to_nose)) # The bottom of the mask

        mouth_points = landmarks[52:71].astype(np.float32) # Get the mouth landmark points
        mouth_width = np.max(mouth_points[:, 0]) - np.min(mouth_points[:, 0]) # Calculate the width of the mouth
        base_width = mouth_width * 0.4
        mask_width = base_width * modules.globals.mask_size * 0.8 # Calculate the mask width

        mask_direction = np.array([-center_to_nose[1], center_to_nose[0]]) # A vector to determine the direction the mask should expand to the left and right
        mask_direction /= np.linalg.norm(mask_direction) # Make the vector have a length of 1

        mask_polygon = np.array([ # Create the polygon points
            mask_top + mask_direction * (mask_width / 2),
            mask_top - mask_direction * (mask_width / 2),
            mask_bottom - mask_direction * (mask_width / 2),
            mask_bottom + mask_direction * (mask_width / 2)
        ]).astype(np.int32)

        # Ensure the mask stays within the frame
        # mask_polygon[:, 0] = np.clip(mask_polygon[:, 0], 0, frame.shape[1] - 1)
        # mask_polygon[:, 1] = np.clip(mask_polygon[:, 1], 0, frame.shape[0] - 1)

        # Draw the mask
        cv2.fillPoly(mask, [mask_polygon], 255) # Fill the mask with white

        # Calculate bounding box for the mouth cutout
        min_x, min_y = np.min(mask_polygon, axis=0) # Find the top-left corner of the mask
        max_x, max_y = np.max(mask_polygon, axis=0) # Find the bottom-right corner of the mask

        # Extract the masked area from the frame
        mouth_cutout = frame[min_y:max_y, min_x:max_x].copy() # Crop out the mouth

    return mask, mouth_cutout, (min_x, min_y, max_x, max_y)

def create_lower_mouth_mask(face: Face, frame: Frame) -> (np.ndarray, np.ndarray, tuple, np.ndarray):
    """
    Creates a mask for the lower part of the mouth.
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8) # Create a black mask with the same size as the frame
    mouth_cutout = None # Make a variable to hold the cropped mouth
    landmarks = face.landmark_2d_106 # Get the face landmarks
    if landmarks is not None:
        #                  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20
        lower_lip_order = [65, 66, 62, 70, 69, 18, 19, 20, 21, 22, 23, 24, 0, 8, 7, 6, 5, 4, 3, 2, 65] # These numbers represent the points that make up the lower lip
        lower_lip_landmarks = landmarks[lower_lip_order].astype(np.float32)  # Use float for precise calculations # Get the points of the lower lip

        # Calculate the center of the landmarks
        center = np.mean(lower_lip_landmarks, axis=0) # Find the center of the lower lip

        # Expand the landmarks outward
        expansion_factor = 1 + modules.globals.mask_down_size  # Adjust this for more or less expansion # How much to expand the mask by
        expanded_landmarks = (lower_lip_landmarks - center) * expansion_factor + center # Expand the mask

        # Extend the top lip part
        toplip_indices = [20, 0, 1, 2, 3, 4, 5]  # Indices for landmarks 2, 65, 66, 62, 70, 69, 18 # These numbers represent the points of the upper lip
        toplip_extension = modules.globals.mask_size * 0.5  # Adjust this factor to control the extension # How much to expand the top lip by
        for idx in toplip_indices:
            direction = expanded_landmarks[idx] - center # Calculate the direction from the center to the landmark
            direction = direction / np.linalg.norm(direction) # Make the vector have a length of 1
            expanded_landmarks[idx] += direction * toplip_extension # Expand the top lip

        # Extend the bottom part (chin area)
        chin_indices = [11, 12, 13, 14, 15, 16]  # Indices for landmarks 21, 22, 23, 24, 0, 8 # These numbers represent the points of the chin
        chin_extension = 2 * 0.2  # Adjust this factor to control the extension # How much to extend the chin by
        for idx in chin_indices:
            expanded_landmarks[idx][1] += (expanded_landmarks[idx][1] - center[1]) * chin_extension # Extend the chin

        # Convert back to integer coordinates
        expanded_landmarks = expanded_landmarks.astype(np.int32) # Convert the points to integers

        # Calculate bounding box for the expanded lower mouth
        min_x, min_y = np.min(expanded_landmarks, axis=0) # Find the top-left corner
        max_x, max_y = np.max(expanded_landmarks, axis=0) # Find the bottom-right corner

        # Add some padding to the bounding box
        padding = int((max_x - min_x) * 0.1)  # 10% padding # Calculate how much padding to add
        min_x = max(0, min_x - padding) # Make sure the top-left corner isn't outside the frame
        min_y = max(0, min_y - padding)
        max_x = min(frame.shape[1], max_x + padding) # Make sure the bottom-right corner isn't outside the frame
        max_y = min(frame.shape[0], max_y + padding)

        # Ensure the bounding box dimensions are valid
        if max_x <= min_x or max_y <= min_y:
            if (max_x - min_x) <= 1:
                max_x = min_x + 1 # Make sure width is at least 1
            if (max_y - min_y) <= 1:
                max_y = min_y + 1 # Make sure height is at least 1

        # Create the mask
        mask_roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8) # Create a black mask for the mouth
        cv2.fillPoly(mask_roi, [expanded_landmarks - [min_x, min_y]], 255) # Fill the mask with white

        # Apply Gaussian blur to soften the mask edges
        mask_roi = cv2.GaussianBlur(mask_roi, (15, 15), 5) # Blur the edges of the mask

        # Place the mask ROI in the full-sized mask
        mask[min_y:max_y, min_x:max_x] = mask_roi # Put the mouth mask into the larger mask

        # Extract the masked area from the frame
        mouth_cutout = frame[min_y:max_y, min_x:max_x].copy() # Crop out the mouth

        # Return the expanded lower lip polygon in original frame coordinates
        lower_lip_polygon = expanded_landmarks # Store the expanded lip polygon points

    return mask, mouth_cutout, (min_x, min_y, max_x, max_y), lower_lip_polygon

def draw_mouth_mask_visualization(frame: Frame, face: Face, mouth_mask_data: tuple) -> Frame:
    """
    Draws a visualization of the mouth mask.
    """
    landmarks = face.landmark_2d_106 # Get the face landmarks
    if landmarks is not None and mouth_mask_data is not None: # If landmarks and mask data exist
        mask, mouth_cutout, (min_x, min_y, max_x, max_y), lower_lip_polygon = mouth_mask_data # Get all the mask data

        vis_frame = frame.copy() # Copy the original frame

        # Ensure coordinates are within frame bounds
        height, width = vis_frame.shape[:2]
        min_x, min_y = max(0, min_x), max(0, min_y) # Make sure the top-left corner is inside the frame
        max_x, max_y = min(width, max_x), min(height, max_y) # Make sure the bottom-right corner is inside the frame

        # Adjust mask to match the region size
        mask_region = mask[0:max_y - min_y, 0:max_x - min_x]

        # Remove the color mask overlay
        # color_mask = cv2.applyColorMap((mask_region * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Ensure shapes match before blending
        vis_region = vis_frame[min_y:max_y, min_x:max_x]
        # Remove blending with color_mask
        # if vis_region.shape[:2] == color_mask.shape[:2]:
        #     blended = cv2.addWeighted(vis_region, 0.7, color_mask, 0.3, 0)
        #     vis_frame[min_y:max_y, min_x:max_x] = blended

        # Draw the lower lip polygon
        cv2.polylines(vis_frame, [lower_lip_polygon], True, (0, 255, 0), 2) # Draw a green outline

        # Remove the red box
        # cv2.rectangle(vis_frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)

        # Visualize the feathered mask
        feather_amount = max(1, min(30, (max_x - min_x) // modules.globals.mask_feather_ratio,
                                    (max_y - min_y) // modules.globals.mask_feather_ratio)) # Calculate how much to feather the mask
        # Ensure kernel size is odd
        kernel_size = 2 * feather_amount + 1
        feathered_mask = cv2.GaussianBlur(mask_region.astype(float), (kernel_size, kernel_size), 0) # Feather the mask
        feathered_mask = (feathered_mask / feathered_mask.max() * 255).astype(np.uint8) # Convert values between 0 and 255
        # Remove the feathered mask color overlay
        # color_feathered_mask = cv2.applyColorMap(feathered_mask, cv2.COLORMAP_VIRIDIS)

        # Ensure shapes match before blending feathered mask
        # if vis_region.shape == color_feathered_mask.shape:
        #     blended_feathered = cv2.addWeighted(vis_region, 0.7, color_feathered_mask, 0.3, 0)
        #     vis_frame[min_y:max_y, min_x:max_x] = blended_feathered

        # Add labels
        cv2.putText(vis_frame, "Lower Mouth Mask", (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) # Add a text label above the mouth
        cv2.putText(vis_frame, "Feathered Mask", (min_x, max_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) # Add a text label below the mouth

        return vis_frame
    return frame

def apply_mouth_area(frame: np.ndarray, mouth_cutout: np.ndarray, mouth_box: tuple, face_mask: np.ndarray,
                    mouth_polygon: np.ndarray) -> np.ndarray:
    """
    Applies the mouth area to the frame.
    """
    min_x, min_y, max_x, max_y = mouth_box # Get the mask box
    box_width = max_x - min_x # Calculate the width of the box
    box_height = max_y - min_y # Calculate the height of the box

    if mouth_cutout is None or box_width is None or box_height is None or face_mask is None or mouth_polygon is None: # Check that everything exists
        return frame

    try:
        resized_mouth_cutout = cv2.resize(mouth_cutout, (box_width, box_height)) # Resize the mouth cutout
        roi = frame[min_y:max_y, min_x:max_x] # Get the region in the original frame that we'll be changing

        if roi.shape != resized_mouth_cutout.shape: # If the sizes don't match
            resized_mouth_cutout = cv2.resize(resized_mouth_cutout, (roi.shape[1], roi.shape[0])) # Resize the mouth to fit in the frame

        color_corrected_mouth = apply_color_transfer(resized_mouth_cutout, roi) # Make the colors match the original frame

        # Use the provided mouth polygon to create the mask
        polygon_mask = np.zeros(roi.shape[:2], dtype=np.uint8) # Create a black mask for the polygon
        adjusted_polygon = mouth_polygon - [min_x, min_y] # Move the polygon to the correct location
        cv2.fillPoly(polygon_mask, [adjusted_polygon], 255) # Fill the mask with white

        # Apply feathering to the polygon mask
        feather_amount = min(30, box_width // modules.globals.mask_feather_ratio,
                            box_height // modules.globals.mask_feather_ratio) # Calculate how much to feather the mask
        feathered_mask = cv2.GaussianBlur(polygon_mask.astype(float), (0, 0), feather_amount) # Feather the mask
        feathered_mask = feathered_mask / feathered_mask.max() # Convert values between 0 and 1

        face_mask_roi = face_mask[min_y:max_y, min_x:max_x] # Get the portion of the face mask
        combined_mask = feathered_mask * (face_mask_roi / 255.0) # Multiply the feathered mask with the face mask

        combined_mask = combined_mask[:, :, np.newaxis] # Add a dimension to the mask to make it work with the color channels
        blended = (color_corrected_mouth * combined_mask + roi * (1 - combined_mask)).astype(np.uint8) # Blend the mouth onto the original frame

        # Apply face mask to blended result
        face_mask_3channel = np.repeat(face_mask_roi[:, :, np.newaxis], 3, axis=2) / 255.0 # Add color channels to the face mask
        final_blend = blended * face_mask_3channel + roi * (1 - face_mask_3channel) # Blend the face onto the frame

        frame[min_y:max_y, min_x:max_x] = final_blend.astype(np.uint8) # Put the result back into the frame
    except Exception as e:
        pass

    return frame

def reset_face_tracking():
    """
    Resets all face tracking variables, including multi-face tracking data.
    """
    global first_face_embedding, second_face_embedding
    global first_face_position, second_face_position
    global first_face_bbox, second_face_bbox
    global first_face_id, second_face_id
    global first_face_lost_count, second_face_lost_count
    global face_position_history
    
    if 'tracked_faces_many' in globals():
        tracked_faces_many = globals()['tracked_faces_many']
        tracked_faces_many.clear()
    
    if 'position_histories_many' in globals():
        position_histories_many = globals()['position_histories_many']
        position_histories_many.clear()

    first_face_embedding = None
    second_face_embedding = None
    first_face_position = None
    second_face_position = None
    first_face_bbox = None
    second_face_bbox = None
    first_face_id = None
    second_face_id = None
    first_face_lost_count = 0
    face_position_history.clear()
    
    modules.globals.target_face1_score = 0.00
    modules.globals.target_face2_score = 0.00
    modules.globals.target_face3_score = 0.00
    modules.globals.target_face4_score = 0.00
    modules.globals.target_face5_score = 0.00
    modules.globals.target_face6_score = 0.00
    modules.globals.target_face7_score = 0.00
    modules.globals.target_face8_score = 0.00
    modules.globals.target_face9_score = 0.00
    modules.globals.target_face10_score = 0.00

def get_face_center(face: Face) -> Tuple[float, float]:
    """
    Gets the center of the face.
    """
    bbox = face.bbox # Get the bounding box for the face
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2) # Calculate the center using the bounding box

def extract_face_embedding(face: Face) -> np.ndarray:
    """
    Extracts the face embedding (how the face looks).
    """
    try:
        if hasattr(face, 'embedding') and face.embedding is not None: # If the face already has an embedding
            embedding = face.embedding # Get the existing embedding
        else:
            # If the face doesn't have an embedding, we need to generate one
            embedding = get_face_analyser().get(face) # Generate an embedding for the face

        # Normalize the embedding
        return embedding / np.linalg.norm(embedding) # Make the embedding have a length of 1
    except Exception as e:
        print(f"Error extracting face embedding: {e}") # Print an error message if something goes wrong
        # Return a default embedding (all zeros) if extraction fails
        return np.zeros(512, dtype=np.float32) # Return an empty embedding

def find_best_match(embedding: np.ndarray, faces: List[Face]) -> Face:
    """
    Finds the face that is most similar to the given embedding.
    """
    if embedding is None:
        # Handle case where embedding is None, maybe log a message or skip processing
        print("No embedding to match against, skipping face matching.") # If no embedding, we can't match
        return None
    best_match = None # Make a variable to store the best face
    best_similarity = -1 # Make a variable to store the best similarity score

    for face in faces: # Loop through all the faces
        face_embedding = extract_face_embedding(face) # Get the embedding for this face
        similarity = cosine_similarity(embedding, face_embedding) # Calculate how similar this face is

        if similarity > best_similarity: # If this is the most similar face
            best_similarity = similarity # Store the similarity score
            best_match = face # Store the face

    return best_match

def update_face_assignments(target_faces: List[Face]):
    """
    Updates which faces are assigned to the tracked faces.
    """
    global first_face_embedding, second_face_embedding, first_face_position, second_face_position, last_assignment_time

    if len(target_faces) == 0: # If there are no faces to assign, don't do anything
        return

    current_time = time.time() # Get the current time
    if current_time - last_assignment_time < assignment_cooldown: # If it's too soon to update face assignments, don't do anything
        return  # Don't update assignments during cooldown period

    try:
        face_embeddings = [extract_face_embedding(face) for face in target_faces] # Get the embeddings for all the faces
        face_positions = [get_face_center(face) for face in target_faces] # Get the positions for all the faces

        if first_face_embedding is None and face_embeddings: # If we haven't assigned a first face yet
            first_face_embedding = face_embeddings[0] # Assign the first detected face to be the first tracked face
            first_face_position = face_positions[0] # Remember where the first face is
            last_assignment_time = current_time # Remember when we did this

        if modules.globals.both_faces and second_face_embedding is None and len(face_embeddings) > 1: # If we haven't assigned a second face yet and we should
            second_face_embedding = face_embeddings[1] # Assign the second detected face to be the second tracked face
            second_face_position = face_positions[1] # Remember where the second face is
            last_assignment_time = current_time # Remember when we did this

        if first_face_embedding is not None: # If we have a first tracked face
            best_match_index = -1
            best_match_score = -1
            for i, (embedding, position) in enumerate(zip(face_embeddings, face_positions)): # Loop through all the faces
                embedding_similarity = cosine_similarity(first_face_embedding, embedding) # How similar this face is to the first tracked face
                position_consistency = 1 / (
                            1 + np.linalg.norm(np.array(position) - np.array(first_face_position))) if first_face_position else 0 # How close this face is to the first tracked face
                combined_score = embedding_similarity * 0.7 + position_consistency * 0.3 # Calculate a combined score for similarity and position

                if combined_score > best_match_score and combined_score > 0.8:  # Increased threshold
                    best_match_score = combined_score # Remember the score
                    best_match_index = i # Remember the face index

            if best_match_index != -1: # If we found a good match
                first_face_embedding = face_embeddings[best_match_index] # Assign this face to the first tracked face
                first_face_position = face_positions[best_match_index] # Remember where the face is
                last_assignment_time = current_time # Remember when we did this

        if modules.globals.both_faces and second_face_embedding is not None: # If we are using two faces and we already have a second face assigned
            remaining_embeddings = face_embeddings[:best_match_index] + face_embeddings[best_match_index + 1:] # Get the embeddings of all faces that are NOT the first face
            remaining_positions = face_positions[:best_match_index] + face_positions[best_match_index + 1:] # Get the positions of all faces that are NOT the first face

            if remaining_embeddings: # If there are any other faces
                best_match_index = -1
                best_match_score = -1
                for i, (embedding, position) in enumerate(zip(remaining_embeddings, remaining_positions)): # Loop through all other faces
                    embedding_similarity = cosine_similarity(second_face_embedding, embedding) # How similar this face is to the second tracked face
                    position_consistency = 1 / (
                                1 + np.linalg.norm(np.array(position) - np.array(second_face_position))) if second_face_position else 0 # How close this face is to the second tracked face
                    combined_score = embedding_similarity * 0.7 + position_consistency * 0.3 # Calculate the combined score

                    if combined_score > best_match_score and combined_score > 0.8:  # Increased threshold
                        best_match_score = combined_score # Remember the score
                        best_match_index = i # Remember the face index

                if best_match_index != -1: # If we found a good match
                    second_face_embedding = remaining_embeddings[best_match_index] # Assign the best match to the second face
                    second_face_position = remaining_positions[best_match_index] # Remember where the face is
                    last_assignment_time = current_time # Remember when we did this

    except Exception as e:
        print(f"Error in update_face_assignments: {e}") # Print an error message if something goes wrong

def cosine_similarity(a, b):
    """
    Calculates how similar two embeddings are.
    """
    if a is None or b is None:
        # Log an error message or handle the None case appropriately
        # print("Warning: One of the embeddings is None.")
        return 0  # or handle it as needed # If either embedding doesn't exist, they are not similar
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) # Calculate the cosine similarity

def get_best_match(embedding: np.ndarray, face_embeddings: List[np.ndarray]) -> int:
    """
    Gets the index of the best matching embedding.
    """
    similarities = [cosine_similarity(embedding, face_embedding) for face_embedding in face_embeddings] # Calculate how similar each face is
    return np.argmax(similarities) # Return the index of the most similar face

def crop_face_region(frame: Frame, face: Face, padding: float = 0.2) -> Tuple[Frame, Tuple[int, int, int, int]]:
    """
    Crops the region around a face with padding.
    """
    h, w = frame.shape[:2] # Get the frame height and width
    bbox = face.bbox # Get the bounding box for the face
    x1, y1, x2, y2 = map(int, bbox) # Convert to integers

    # Add padding
    pw, ph = int((x2 - x1) * padding), int((y2 - y1) * padding) # Calculate how much padding to add
    x1, y1 = max(0, x1 - pw), max(0, y1 - ph) # Make sure the top-left corner isn't outside the frame
    x2, y2 = min(w, x2 + pw), min(h, y2 + ph) # Make sure the bottom-right corner isn't outside the frame

    cropped_frame = frame[y1:y2, x1:x2].copy()  # Use .copy() to ensure we're not working with a view # Crop the frame
    return cropped_frame, (x1, y1, x2 - x1, y2 - y1) # Return the cropped frame and it's coordinates

def adjust_face_bbox(face: Face, crop_info: Tuple[int, int, int, int]) -> Face:
    """
    Adjusts the face bounding box and landmarks relative to the cropped region.
    """
    x, y, _, _ = crop_info # Get the top-left coordinates of the cropped region
    adjusted_face = Face() # Make a new face object
    adjusted_face.bbox = face.bbox - np.array([x, y, x, y]) # Shift the bounding box to match the cropped region
    adjusted_face.kps = face.kps - np.array([x, y]) # Shift the keypoints to match the cropped region
    adjusted_face.landmark_2d_106 = face.landmark_2d_106 - np.array([x, y]) # Shift the 2D landmarks
    adjusted_face.landmark_3d_68 = face.landmark_3d_68.copy() # Copy the 3D landmarks
    adjusted_face.landmark_3d_68[:, :2] -= np.array([x, y]) # Shift the 3D landmarks to match the cropped region
    # Copy other necessary attributes
    for attr in ['det_score', 'gender', 'age', 'embedding', 'embedding_norm', 'normed_embedding']: # Copy the other attributes
        if hasattr(face, attr):
            setattr(adjusted_face, attr, getattr(face, attr))
    return adjusted_face

def create_adjusted_face(face: Face, crop_info: Tuple[int, int, int, int]) -> Face:
    """
    Creates a new face object with adjusted bounding box and landmarks relative to the cropped region.
    """
    x, y, _, _ = crop_info # Get the top-left coordinates of the cropped region
    adjusted_face = Face(bbox=face.bbox - np.array([x, y, x, y]), # Shift the bounding box to match the cropped region
                         kps=face.kps - np.array([x, y]), # Shift the keypoints to match the cropped region
                         det_score=face.det_score, # Copy the detection score
                         landmark_3d_68=face.landmark_3d_68.copy(), # Copy the 3D landmarks
                         landmark_2d_106=face.landmark_2d_106 - np.array([x, y]), # Shift the 2D landmarks
                         gender=face.gender, # Copy the gender
                         age=face.age, # Copy the age
                         embedding=face.embedding) # Copy the embedding

    # Adjust 3D landmarks
    adjusted_face.landmark_3d_68[:, :2] -= np.array([x, y]) # Shift the 3D landmarks to match the cropped region

    return adjusted_face

def create_ellipse_mask(shape, feather_amount=5):
    """
    Creates an ellipse mask.
    """
    mask = np.zeros(shape[:2], dtype=np.float32) # Create a black mask
    center = (shape[1] // 2, shape[0] // 2) # Find the center of the mask
    axes = (int(shape[1] * 0.42), int(shape[0] * 0.48)) # Calculate the size of the ellipse
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1) # Draw a white ellipse
    mask = cv2.GaussianBlur(mask, (0, 0), feather_amount * min(shape[:2])) # Blur the edges
    return mask

def create_edge_blur_mask(shape, blur_amount=40):
    """
    Creates a mask with blurred edges.
    """
    h, w = shape[:2] # Get the frame height and width
    mask = np.ones((h, w), dtype=np.float32) # Create a white mask

    # Adjust blur_amount to not exceed half the minimum dimension
    blur_amount = min(blur_amount, min(h, w) // 4) # Don't let the blur go too far

    # Create smoother gradients
    v_gradient = np.power(np.linspace(0, 1, blur_amount), 2) # Create a smooth gradient for the vertical edges
    h_gradient = np.power(np.linspace(0, 1, blur_amount), 2) # Create a smooth gradient for the horizontal edges

    # Apply vertical gradients
    mask[:blur_amount, :] = v_gradient[:, np.newaxis] # Make the top of the mask transparent
    mask[-blur_amount:, :] = v_gradient[::-1, np.newaxis] # Make the bottom of the mask transparent

    # Apply horizontal gradients
    mask[:, :blur_amount] *= h_gradient[np.newaxis, :] # Make the left of the mask transparent
    mask[:, -blur_amount:] *= h_gradient[::-1][np.newaxis, :] # Make the right of the mask transparent

    return cv2.GaussianBlur(mask, (0, 0), sigmaX=blur_amount / 4, sigmaY=blur_amount / 4) # Blur the mask

def blend_with_mask(swapped_region, original_region, mask):
    """
    Blends two regions using a mask.
    """
    mask_3d = np.expand_dims(mask, axis=2).repeat(3, axis=2) # Add color channels to the mask
    return (swapped_region * mask_3d + original_region * (1 - mask_3d)).astype(np.uint8) # Blend the images using the mask

def create_pseudo_face(position):
    """
    Creates a fake face object for face tracking when a real face is not detected.
    """
    class PseudoFace: # Create a new class for the fake face
        def __init__(self, position):
            x, y = position # Get the center of the face
            width = height = 100  # Approximate face size # Choose a size for the fake face

            # More realistic bbox
            self.bbox = np.array([x - width / 2, y - height / 2, x + width / 2, y + height / 2]) # Create a bounding box for the face

            # Generate landmarks
            self.landmark_2d_106 = generate_anatomical_landmarks(position) # Create landmarks for the face

            # Extract kps from landmarks
            self.kps = np.array([
                self.landmark_2d_106[36],  # Left eye
                self.landmark_2d_106[90],  # Right eye
                self.landmark_2d_106[80],  # Nose tip
                self.landmark_2d_106[57],  # Left mouth corner
                self.landmark_2d_106[66]  # Right mouth corner
            ]) # Create key points from the landmarks

            # Generate 3D landmarks (just add a random z-coordinate to 2D landmarks)
            self.landmark_3d_68 = np.column_stack((self.landmark_2d_106[:68], np.random.normal(0, 5, 68))) # Create 3d landmarks

            # Other attributes
            self.det_score = 0.99 # Set a high detection score
            self.embedding = np.random.rand(512)  # Random embedding # Create a random embedding
            self.embedding_norm = np.linalg.norm(self.embedding) # Calculate the length of the embedding
            self.gender = 0 # Set the gender to male
            self.age = 25 # Set the age to 25
            self.pose = np.zeros(3) # Set the pose to 0
            self.normed_embedding = self.embedding / self.embedding_norm # Normalize the embedding

    return PseudoFace(position) # Create a fake face at the given position

def generate_anatomical_landmarks(position):
    """
    Generates fake face landmarks for the pseudo face.
    """
    x, y = position # Get the center position
    landmarks = [] # List to store the landmarks

    # Right side face (0-16)
    for i in range(17):
        landmarks.append([x - 40 + i * 2, y - 30 + i * 3]) # Create landmarks along the right side of the face

    # Left side face (17-32)
    for i in range(16):
        landmarks.append([x + 40 - i * 2, y - 30 + i * 3]) # Create landmarks along the left side of the face

    # Right eye (33-42)
    eye_center = [x - 20, y - 10] # Set the position of the right eye
    for i in range(10):
        angle = i * (2 * np.pi / 10) # Calculate the angle around the eye
        landmarks.append([eye_center[0] + 10 * np.cos(angle), eye_center[1] + 5 * np.sin(angle)]) # Create landmarks around the eye

    # Right eyebrow (43-51)
    for i in range(9):
        landmarks.append([x - 35 + i * 5, y - 30]) # Create landmarks for the right eyebrow

    # Mouth (52-71)
    mouth_center = [x, y + 30] # Set the position of the mouth
    for i in range(20):
        angle = i * (2 * np.pi / 20) # Calculate the angle around the mouth
        landmarks.append([mouth_center[0] + 15 * np.cos(angle), mouth_center[1] + 7 * np.sin(angle)]) # Create landmarks around the mouth

    # Nose (72-86)
    for i in range(15):
        landmarks.append([x - 7 + i, y + 10 + i // 2]) # Create landmarks for the nose

    # Left eye (87-96)
    eye_center = [x + 20, y - 10] # Set the position of the left eye
    for i in range(10):
        angle = i * (2 * np.pi / 10) # Calculate the angle around the eye
        landmarks.append([eye_center[0] + 10 * np.cos(angle), eye_center[1] + 5 * np.sin(angle)]) # Create landmarks around the eye

    # Left eyebrow (97-105)
    for i in range(9):
        landmarks.append([x + 5 + i * 5, y - 30]) # Create landmarks for the left eyebrow

    return np.array(landmarks, dtype=np.float32)

def draw_all_landmarks(frame: Frame, face: Face) -> Frame:
    """
    Draws all face landmarks on the frame.
    """
    if face.landmark_2d_106 is None: # If there are no landmarks, don't draw anything
        return frame

    landmarks = face.landmark_2d_106.astype(np.int32) # Get the face landmarks and convert them to integers

    # Define colors for different parts of the face
    colors = { # A dictionary with colors for each facial feature
        'face_outline': (255, 255, 255),  # White
        'lefteye': (0, 255, 0),  # Green
        'righteye': (0, 255, 0),  # Green
        'eyebrows': (235, 213, 52),  # Cyan
        'nose': (0, 255, 255),  # Yellow
        'uppermouth': (0, 0, 255),  # Red
        'lowermouth': (255, 150, 0)  # Blue
    }

    # Draw face outline (landmarks 0-33)
    for i in range(33):
        cv2.circle(frame, tuple(landmarks[i]), 1, colors['face_outline'], -1) # Draw a circle for each landmark

    # Right Eye (landmarks 33-43)
    for i in range(33, 43):
        cv2.circle(frame, tuple(landmarks[i]), 1, colors['righteye'], -1) # Draw a circle for each landmark

    # Right Eyebrow (landmarks 43-52)
    for i in range(43, 52):
        cv2.circle(frame, tuple(landmarks[i]), 1, colors['eyebrows'], -1) # Draw a circle for each landmark

    # Lower mouth (landmarks 52-62)
    lower_lip_order = [65, 66, 62, 70, 69, 18, 19, 20, 21, 22, 23, 24, 0, 8, 7, 6, 5, 4, 3, 2, 65] # Numbers representing the lower lip
    lower_lip_landmarks = landmarks[lower_lip_order] # Get the landmarks

    # mouth (landmarks 52-72)
    for i in range(52, 62):
        cv2.circle(frame, tuple(landmarks[i]), 1, colors['lowermouth'], -1) # Draw a circle for each landmark

    # Draw polyline connecting lower lip points
    cv2.polylines(frame, [lower_lip_landmarks], isClosed=True, color=colors['lowermouth'], thickness=1) # Draw a line connecting the lower lip points

    # mouth (landmarks 52-72)
    for i in range(62, 72):
        cv2.circle(frame, tuple(landmarks[i]), 1, colors['uppermouth'], -1) # Draw a circle for each landmark

    # nose (landmarks 72-87)
    for i in range(72, 87):
        cv2.circle(frame, tuple(landmarks[i]), 1, colors['nose'], -1) # Draw a circle for each landmark
    
    # Left Eye (landmarks 87-97)
    for i in range(87, 97):
        cv2.circle(frame, tuple(landmarks[i]), 1, colors['lefteye'], -1) # Draw a circle for each landmark

    # Left Eyebrow (landmarks 97-106)
    for i in range(97, 106):
        cv2.circle(frame, tuple(landmarks[i]), 1, colors['eyebrows'], -1) # Draw a circle for each landmark

    point_72 = tuple(landmarks[72])
    point_80 = tuple(landmarks[80])
    
    cv2.line(frame, point_72, point_80, (255, 255, 0), 2) # Yellow line for nose line
    
    # Calculate the angle of the line relative to the vertical
    dx = point_72[0] - point_80[0]
    dy = point_72[1] - point_80[1]
    
    angle_radians = math.atan2(-dx, -dy)
    angle_degrees = math.degrees(angle_radians)

    # Adjust the angle to be relative to the vertical (0 degrees for straight up, positive for right)
    angle_degrees = angle_degrees 

    # Normalize the angle between -180 to 180 degrees
    if angle_degrees > 180:
        angle_degrees -= 360
    elif angle_degrees < -180:
        angle_degrees += 360

    # Display the rotation angle
    text_position = (point_72[0] + 25, point_72[1] + 20)  # Position text near the nose tip
    cv2.putText(frame, f"{-angle_degrees:.2f} deg", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1) # Draw the angle near the nose

    return frame

def apply_mouth_area_with_landmarks(temp_frame, mouth_cutout, mouth_box, face_mask, target_face):
    """
    Applies the mouth area to the frame using landmarks for the mask.
    """
    landmarks = target_face.landmark_2d_106 # Get the landmarks
    if landmarks is not None: # If there are landmarks
        nose_tip = landmarks[80].astype(np.float32) # Get the nose tip
        center_bottom = landmarks[73].astype(np.float32) # Get the bottom of the center of the mouth
        center_to_nose = nose_tip - center_bottom # A vector from the center to the nose
        mask_height = np.linalg.norm(center_to_nose) * modules.globals.mask_size * 0.3 # Calculate the mask height
        mask_top = nose_tip + center_to_nose * 0.2 + np.array([0, -modules.globals.mask_size * 0.1]) # Calculate the top of the mask
        mask_bottom = mask_top + center_to_nose * (mask_height / np.linalg.norm(center_to_nose)) # Calculate the bottom of the mask
        mouth_points = landmarks[52:71].astype(np.float32) # Get the mouth landmark points
        mouth_width = np.max(mouth_points[:, 0]) - np.min(mouth_points[:, 0]) # Calculate the width of the mouth
        base_width = mouth_width * 0.4
        mask_width = base_width * modules.globals.mask_size * 0.8 # Calculate the mask width
        mask_direction = np.array([-center_to_nose[1], center_to_nose[0]], dtype=np.float32) # A vector to determine the mask width
        mask_direction /= np.linalg.norm(mask_direction) # Make the vector have a length of 1
        mouth_polygon = np.array([
            mask_top + mask_direction * (mask_width / 2),
            mask_top - mask_direction * (mask_width / 2),
            mask_bottom - mask_direction * (mask_width / 2),
            mask_bottom + mask_direction * (mask_width / 2)
        ]).astype(np.int32) # Create the mask polygon

        return apply_mouth_area(temp_frame, mouth_cutout, mouth_box, face_mask, mouth_polygon) # Apply the mouth area using landmarks
    else:
        return apply_mouth_area(temp_frame, mouth_cutout, mouth_box, face_mask, None) # Apply the mouth area without landmarks

def get_two_faces(frame: Frame) -> List[Face]:
    """
    Gets the two leftmost faces in the frame.
    """
    faces = get_many_faces(frame) # Detect the faces in the frame
    if faces: # If any faces were detected
        # Sort faces from left to right based on the x-coordinate of the bounding box
        sorted_faces = sorted(faces, key=lambda x: x.bbox[0]) # Sort the faces from left to right
        return sorted_faces[:2]  # Return up to two faces, leftmost and rightmost
    return [] # If no faces were detected, return an empty list

def calculate_iou(boxA, boxB):
    """
    Calculates Intersection over Union (IOU) between two bounding boxes.
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def estimate_head_rotation(face: Face) -> float:
    """
    Calculates the head rotation angle in degrees relative to the vertical axis.
    Returns 0.0 if landmarks are missing.
    """
    if face.landmark_2d_106 is None:
        return 0.0

    landmarks = face.landmark_2d_106.astype(np.int32)
    
    # Use nose bridge (72) and nose tip (80) to determine angle
    if len(landmarks) < 81:
         return 0.0

    point_72 = landmarks[72]
    point_80 = landmarks[80]

    dx = point_72[0] - point_80[0]
    dy = point_72[1] - point_80[1]

    angle_radians = math.atan2(-dx, -dy)
    angle_degrees = math.degrees(angle_radians)

    # Normalize angle
    if angle_degrees > 180:
        angle_degrees -= 360
    elif angle_degrees < -180:
        angle_degrees += 360
        
    return angle_degrees

def create_hybrid_mouth_mask(face: Face, frame: Frame) -> Tuple[np.ndarray, np.ndarray, tuple, tuple]:
    """
    Hybrid Mask logic:
    1. Smart Color Detection (Red/Tongue).
    2. Geometric Mask (Blue Circle) -> PINNED below the NOSE (Landmark 80).
    3. Standard Gaussian Feathering (Blurs Outward & Inward).
    """
    h, w = frame.shape[:2]
    final_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Visualization masks
    smart_mask_full = np.zeros((h, w), dtype=np.uint8)
    geo_mask_full = np.zeros((h, w), dtype=np.uint8)
    
    mouth_cutout = None

    landmarks = face.landmark_2d_106
    if landmarks is not None:
        # --- PART A: SMART COLOR MASK (Red/Tongue) ---
        mouth_points = landmarks[52:71].astype(np.int32)
        min_x, min_y = np.min(mouth_points, axis=0)
        max_x, max_y = np.max(mouth_points, axis=0)
        
        pad = int((max_x - min_x) * 0.2)
        r_min_x, r_min_y = max(0, min_x - pad), max(0, min_y - pad)
        r_max_x, r_max_y = min(w, max_x + pad), min(h, max_y + pad)
        
        mouth_roi = frame[r_min_y:r_max_y, r_min_x:r_max_x]
        
        if mouth_roi.size > 0:
            hsv_roi = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2HSV)
            # Red/Pink
            mask1 = cv2.inRange(hsv_roi, np.array([0, 60, 60]), np.array([15, 255, 255]))
            mask2 = cv2.inRange(hsv_roi, np.array([160, 60, 60]), np.array([180, 255, 255]))
            # Dark Interior
            mask3 = cv2.inRange(hsv_roi, np.array([0, 0, 0]), np.array([180, 255, 80]))
            
            smart_mask_roi = cv2.bitwise_or(mask1, mask2)
            smart_mask_roi = cv2.bitwise_or(smart_mask_roi, mask3)
            
            smart_mask_full[r_min_y:r_max_y, r_min_x:r_max_x] = smart_mask_roi
            final_mask[r_min_y:r_max_y, r_min_x:r_max_x] = smart_mask_roi

        # --- PART B: GEOMETRIC MASK (The Eating Oval) ---
        padding_val = modules.globals.mask_down_size
        top_val = int(modules.globals.mask_size)
        
        # 1. Radius/Size Calculation
        mouth_width = np.linalg.norm(landmarks[52] - landmarks[61])
        
        radius_scale = (padding_val - 0.4) * 1.5 
        if radius_scale < 0: radius_scale = 0
        
        radius_w = int(mouth_width * radius_scale * 1.2)
        radius_h = int(mouth_width * radius_scale * 1.0)
        
        # 2. Determine Anchor Points
        chin_indices = [65, 66, 62, 70, 69, 18, 19, 20, 21, 22, 23, 24, 0, 8, 7, 6, 5, 4, 3, 2]
        chin_points = landmarks[chin_indices].astype(np.float32)
        center = np.mean(chin_points, axis=0)
        
        center_x = center[0]
        center_y = center[1] + (top_val * 4)

        # 3. PINNING LOGIC (Below Nose Tip 80)
        nose_tip_y = landmarks[80][1]
        upper_lip_y = landmarks[71][1]
        
        # Philtrum buffer (20%)
        philtrum_h = upper_lip_y - nose_tip_y
        pin_limit_y = nose_tip_y + (philtrum_h * 0.2)
        
        current_oval_top = center_y - radius_h
        
        # Slide down if hitting nose limit
        if current_oval_top < pin_limit_y:
            shift_needed = pin_limit_y - current_oval_top
            center_y = center_y + shift_needed
        
        # 4. Draw Oval
        if radius_w > 0 and radius_h > 0:
            cv2.ellipse(
                geo_mask_full, 
                (int(center_x), int(center_y)), 
                (radius_w, radius_h), 
                0, 0, 360, 
                255, -1
            )
            
        # --- PART C: COMBINE ---
        final_mask = cv2.bitwise_or(final_mask, geo_mask_full)
        
        # --- PART D: STANDARD FEATHERING (Outward + Inward) ---
        # Reverted to GaussianBlur for natural blending
        feather_val = int(modules.globals.mask_feather_ratio)
        if feather_val > 0:
            k_size = (feather_val * 2) + 1
            final_mask = cv2.GaussianBlur(final_mask, (k_size, k_size), 0)

        # Cutout logic
        coords = cv2.findNonZero(final_mask)
        if coords is not None:
            x, y, w, h_rect = cv2.boundingRect(coords)
            x = max(0, x - 10)
            y = max(0, y - 10)
            w = min(frame.shape[1] - x, w + 20)
            h_rect = min(frame.shape[0] - y, h_rect + 20)
            
            mouth_cutout = frame[y:y+h_rect, x:x+w].copy()
            
            return final_mask, mouth_cutout, (x, y, x+w, y+h_rect), (smart_mask_full, geo_mask_full)

    return final_mask, None, (0,0,0,0), (None, None)