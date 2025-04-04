# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import statistics
import math
import json
import io # For handling image bytes
import requests # For fetching URL
from flask import Flask, request, jsonify

# --- Try importing Open3D ---
try:
    import open3d as o3d
    print(f"Open3D version: {o3d.__version__}")
    OPEN3D_AVAILABLE = True
except ImportError:
    print("ERROR: Open3D library not found. Validation will fail.")
    OPEN3D_AVAILABLE = False
    o3d = None # Define o3d as None if import fails

# --- CONFIGURATION (Copied from Colab Script V16.1 - Adjust as needed) ---
DEPTH_ENCODING_HIGH_IS_CLOSE = False # Marigold
config = {
    "DEPTH_THRESHOLD": 25 * 256,
    "COLLISION_IOU_THRESHOLD": 0.05,
    "SURFACE_DEPTH_TOLERANCE": 15000, # Tune this!
    "MIN_BOX_AREA_FOR_DEPTH_SAMPLING": 100,
    "RANSAC_DISTANCE_THRESHOLD": 2000, # Tune this!
    "RANSAC_N": 4,
    "RANSAC_ITERATIONS": 1000,
    "RANSAC_MIN_PLANE_POINTS": 5000,
    "PLANE_NORMAL_THRESHOLD": 0.85 # Tune this!
}

# --- HELPER FUNCTIONS (Copied & Modified load_depth_map) ---

def clip_box(box, W, H):
    # ... (Keep V7 version) ...
    if not isinstance(box,(list,tuple)) or len(box)!=4: return [0,0,0,0]
    xmin,ymin,xmax,ymax=box; clipped_xmin=max(0,int(round(xmin))); clipped_ymin=max(0,int(round(ymin))); clipped_xmax=min(W,int(round(xmax))); clipped_ymax=min(H,int(round(ymax)))
    if clipped_xmin>=clipped_xmax: clipped_xmax=clipped_xmin+1;
    if clipped_ymin>=clipped_ymax: clipped_ymax=clipped_ymin+1;
    clipped_xmax=min(W,clipped_xmax); clipped_ymax=min(H,clipped_ymax); return [clipped_xmin,clipped_ymin,clipped_xmax,clipped_ymax]

def load_depth_map(image_source, is_url=False):
    """Loads depth map from URL or local path, converts to PNG if needed, returns uint16 NumPy array."""
    img = None
    source_desc = f"URL '{image_source}'" if is_url else f"path '{image_source}'"
    was_converted_to_png = False # Flag to track conversion
    try:
        if is_url:
            print(f"Fetching depth map from {source_desc}...")
            response = requests.get(image_source, stream=True, timeout=15) # Increased timeout slightly
            response.raise_for_status()
            print("Image data fetched successfully.")
            # Load image from response content
            img_bytes = io.BytesIO(response.content)
            img = Image.open(img_bytes)
            print(f"Loaded image initially. Format: {img.format}, Mode: {img.mode}, Size: {img.size}")

            # --- NEW: Convert to PNG if necessary ---
            if img.format != 'PNG':
                print(f"  Converting image from {img.format} to PNG in memory...")
                png_buffer = io.BytesIO()
                # Preserve mode if possible during conversion (important for depth data)
                # If the original mode is suitable for saving as PNG, use it.
                # Otherwise, let Pillow handle conversion (might affect depth interpretation later)
                try:
                    img.save(png_buffer, format='PNG') 
                    png_buffer.seek(0) # Rewind the buffer
                    img.close() # Close the original image object
                    img = Image.open(png_buffer) # Re-open the image from the PNG buffer
                    print(f"  Converted to PNG. New Mode: {img.mode}, New Size: {img.size}")
                    # --- NEW LOG 1 ---
                    print("  Successfully converted and loaded image as PNG.")
                    was_converted_to_png = True
                    # --- END NEW LOG 1 ---
                    
                    # --- NEW: Save the converted PNG for debugging ---
                    try:
                        save_path = "debug_converted_depth.png" 
                        print(f"  Saving converted PNG to '{save_path}' for debugging...")
                        img.save(save_path)
                        print(f"  Successfully saved debug PNG to '{save_path}'.")
                    except Exception as save_err:
                        print(f"  ERROR: Failed to save debug PNG '{save_path}': {save_err}")
                    # --- END NEW ---

                except Exception as conversion_err:
                    print(f"  ERROR during PNG conversion: {conversion_err}. Proceeding with original format.")
                    # If conversion fails, fall back to the original image object
                    img_bytes.seek(0) # Rewind original bytes
                    img = Image.open(img_bytes) 
            # --- END NEW ---
            
        else:
            # Load from local path (assume it's already in a good format or handle locally if needed)
            img = Image.open(image_source)
            print(f"Loaded depth map from {source_desc}, Format: {img.format}, Mode: {img.mode}, Size: {img.size}")
            # Optional: Add similar PNG conversion logic for local files if required
            # Note: PNG conversion for local files is not implemented here, add if needed.

        # --- NEW LOG 2 ---
        log_suffix = "(PNG format)" if was_converted_to_png else f"({img.format} format)"
        print(f"Proceeding to extract depth data from image {log_suffix}...")
        # --- END NEW LOG 2 ---

        # --- Conversion Logic (Keep from V7, now operates on potentially PNG-converted image) ---
        if img.mode == 'I;16' or img.mode == 'I': 
             depth_array = np.array(img, dtype=np.uint16)
        elif img.mode == 'L': 
            print("  WARN: 8-bit input (after potential PNG conversion)."); 
            depth_array = np.array(img, dtype=np.uint8).astype(np.uint16) * 257
        elif img.mode == 'F': 
            print("  WARN: 32-bit float input (after potential PNG conversion)."); 
            float_array = np.array(img, dtype=np.float32); 
            depth_array = (np.clip(float_array, 0.0, 1.0) * 65535.0).astype(np.uint16)
        else:
            print(f"  Attempting conversion from mode {img.mode} to 16-bit integer (after potential PNG conversion)..."); 
            img_i = img.convert('I'); 
            depth_array = np.array(img_i, dtype=np.uint16)

        print(f"  -> Final NumPy shape:{depth_array.shape}, dtype:{depth_array.dtype}, Range:[{np.min(depth_array)}-{np.max(depth_array)}]");
        return depth_array

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to fetch depth map from {source_desc}: {e}")
        return None
    except FileNotFoundError:
        print(f"ERROR: Depth map file not found at '{image_source}'")
        return None
    except Exception as e:
        import traceback
        print(f"ERROR: Could not load/process depth map from {source_desc}:")
        traceback.print_exc() # Print full traceback for better debugging
        return None
    finally:
        # Ensure image object is closed if it exists
        if img:
            try:
                img.close()
            except Exception:
                pass # Ignore errors during close

def get_depth_at_pixel(x, y, depth_array):
    # ... (Keep V7 version) ...
    h,w=depth_array.shape; x,y=int(round(x)),int(round(y));
    if 0<=y<h and 0<=x<w: return int(depth_array[y,x])
    return None

def get_median_depth_in_box(box, depth_array):
    # ... (Keep V7 version) ...
    if box is None: return None
    xmin,ymin,xmax,ymax=box; h,w=depth_array.shape
    if xmin>=xmax or ymin>=ymax or xmin>=w or ymin>=h or xmax<=0 or ymax<=0: return None
    box_area=(xmax-xmin)*(ymax-ymin)
    if box_area<config["MIN_BOX_AREA_FOR_DEPTH_SAMPLING"]: return None
    box_depths=depth_array[ymin:ymax, xmin:xmax].flatten()
    if box_depths.size==0: return None
    return np.median(box_depths)

def calculate_iou(boxA, boxB):
    # ... (Keep V7 version) ...
    if not isinstance(boxA,(list,tuple)) or len(boxA)!=4 or not isinstance(boxB,(list,tuple)) or len(boxB)!=4: return 0.0
    axmin,aymin,axmax,aymax=[round(c) for c in boxA]; bxmin,bymin,bxmax,bymax=[round(c) for c in boxB]
    x_left,y_top=max(axmin,bxmin), max(aymin,bymin); x_right,y_bottom=min(axmax,bxmax), min(aymax,bymax)
    if x_right<x_left or y_bottom<y_top: return 0.0
    intersectionArea=(x_right-x_left)*(y_bottom-y_top); boxAArea=(axmax-axmin)*(aymax-aymin); boxBArea=(bxmax-bxmin)*(bymax-bymin)
    if boxAArea<0: boxAArea=0;
    if boxBArea<0: boxBArea=0;
    if intersectionArea<0: intersectionArea=0
    unionArea=float(boxAArea+boxBArea-intersectionArea);
    if unionArea<=0: return 0.0
    iou=intersectionArea/unionArea; return max(0.0, min(iou, 1.0))

# --- SURFACE DEPTH RANGE CALCULATION (Keep V16.1 - RANSAC) ---
def calculateSurfaceDepthRanges(depth_array, image_dims):
    # ... (Paste the ENTIRE V16.1 function here, no changes needed internally) ...
    if not OPEN3D_AVAILABLE: print("ERROR: Open3D unavailable."); return {"floor":None,"wall":None,"ceiling":None}
    h, w = depth_array.shape
    print("\n--- Calculating Surface Depth Ranges using Open3D RANSAC (V16.1) ---")
    img_w_orig, img_h_orig = image_dims['width'], image_dims['height']
    cx, cy, fx, fy = w / 2.0, h / 2.0, float(w), float(w)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    print(f"  Intrinsics (est): w={w},h={h}, f={fx:.1f}, c=({cx:.1f},{cy:.1f})")
    try:
        o3d_depth = o3d.geometry.Image(depth_array); pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, intrinsic, depth_scale=1.0, depth_trunc=65535.0, stride=1)
        if not pcd.has_points(): print("ERROR: No points in cloud."); return {"floor":None,"wall":None,"ceiling":None}
        print(f"  PointCloud created: {len(pcd.points)} points.")
    except Exception as e: print(f"ERROR creating point cloud: {e}"); return {"floor":None,"wall":None,"ceiling":None}
    planes=[]; remaining_pcd=pcd; original_indices_map=np.arange(len(pcd.points)); num_planes_to_find=5
    print("Finding planes...");
    for i in range(num_planes_to_find):
        try: plane_model, inliers_relative = remaining_pcd.segment_plane(distance_threshold=config["RANSAC_DISTANCE_THRESHOLD"],ransac_n=config["RANSAC_N"],num_iterations=config["RANSAC_ITERATIONS"])
        except Exception as e: print(f"  ERROR RANSAC iter {i+1}: {e}"); break
        num_inliers=len(inliers_relative)
        if num_inliers<config["RANSAC_MIN_PLANE_POINTS"]: print(f"  Iter {i+1}: Not enough inliers ({num_inliers}). Stop."); break
        current_inlier_original_indices=original_indices_map[inliers_relative]; normal=np.array(plane_model[:3]); normal/=np.linalg.norm(normal)
        planes.append({"model":plane_model,"original_indices":current_inlier_original_indices,"normal":normal,"num_inliers":num_inliers}); print(f"  Iter {i+1}: Found plane {num_inliers} pts. Normal:[{normal[0]:.2f},{normal[1]:.2f},{normal[2]:.2f}]")
        remaining_pcd=remaining_pcd.select_by_index(inliers_relative,invert=True); original_indices_map=np.delete(original_indices_map,inliers_relative)
        if not remaining_pcd.has_points(): print("  No points remaining."); break
    if not planes: print("WARN: RANSAC found no planes."); return {"floor":None,"wall":None,"ceiling":None}
    identified_surfaces={"floor":None,"wall":[],"ceiling":None}; y_axis=np.array([0,1,0]); thresh=config["PLANE_NORMAL_THRESHOLD"]
    print("Identifying planes..."); floor_plane_idx, ceil_plane_idx = -1, -1
    for i, plane in enumerate(planes):
        normal=plane["normal"]; dot_y=np.dot(normal, y_axis)
        if dot_y<-thresh:
            if identified_surfaces["floor"] is None or plane["num_inliers"]>identified_surfaces["floor"]["num_inliers"]: identified_surfaces["floor"]=plane; plane["type"]="floor"; floor_plane_idx=i; print(f"  Plane {i} -> FLOOR (NormalY:{dot_y:.2f})")
        elif dot_y>thresh:
             if identified_surfaces["ceiling"] is None or plane["num_inliers"]>identified_surfaces["ceiling"]["num_inliers"]: identified_surfaces["ceiling"]=plane; plane["type"]="ceiling"; ceil_plane_idx=i; print(f"  Plane {i} -> CEILING (NormalY:{dot_y:.2f})")
        elif abs(dot_y)<(1.0-thresh): identified_surfaces["wall"].append(plane); plane["type"]="wall"; print(f"  Plane {i} -> WALL (NormalY:{dot_y:.2f})")
    if len(identified_surfaces["wall"])>=1: identified_surfaces["wall"]=max(identified_surfaces["wall"], key=lambda p: p["num_inliers"]); print(f"  Selected largest wall.")
    else: identified_surfaces["wall"]=None
    ranges={"floor":None,"wall":None,"ceiling":None}; tolerance=config["SURFACE_DEPTH_TOLERANCE"]; all_points_np=np.asarray(pcd.points)
    print("Calculating depth ranges from inliers...")
    for surface_type, plane_info in identified_surfaces.items():
        if isinstance(plane_info, dict):
            try:
                original_inlier_indices=plane_info["original_indices"]
                if len(original_inlier_indices)>0:
                    inlier_depths=all_points_np[original_inlier_indices,2]; median_depth=np.median(inlier_depths)
                    min_z_overall, max_z_overall=np.min(all_points_np[:,2]), np.max(all_points_np[:,2])
                    if (min_z_overall+10)<median_depth<(max_z_overall-10):
                        ranges[surface_type]={"median":median_depth,"min":max(0,median_depth-tolerance),"max":min(65535,median_depth+tolerance)}
                        print(f"Final {surface_type.upper()}: Med Z={median_depth:.1f} -> Z Range:[{ranges[surface_type]['min']:.1f}-{ranges[surface_type]['max']:.1f}]")
                    else: print(f"WARN: Median Z ({median_depth:.1f}) for {surface_type} unreliable.")
                else: print(f"WARN: No inliers for {surface_type}?")
            except Exception as e: print(f"ERROR calc range {surface_type}: {e}")
        else: print(f"WARN: No plane for {surface_type}.")
    print("Performing sanity check (Low Z = Close)...")
    valid_ranges={"floor":None,"wall":None,"ceiling":None}; fm=ranges.get("floor",{}).get("median") if ranges.get("floor") else None; wm=ranges.get("wall",{}).get("median") if ranges.get("wall") else None; cm=ranges.get("ceiling",{}).get("median") if ranges.get("ceiling") else None
    floor_ok,wall_ok,ceiling_ok=True,True,True; min_sep=max(500,tolerance*0.1)
    if DEPTH_ENCODING_HIGH_IS_CLOSE: print(" ERROR!"); valid_ranges={k:None for k in valid_ranges}
    else:
        if cm is not None and fm is not None and cm>=fm-min_sep: print(f"  WARN (Sanity Z): Ceil ({cm:.1f})!<<Floor ({fm:.1f})"); floor_ok,ceiling_ok=False,False
        if cm is not None and wm is not None and cm>=wm-min_sep: print(f"  WARN (Sanity Z): Ceil ({cm:.1f})!<<Wall ({wm:.1f})"); ceiling_ok,wall_ok=False,False
        if wm is not None and fm is not None and wm>=fm+tolerance: print(f"  WARN (Sanity Z): Wall ({wm:.1f})>>Floor ({fm:.1f})"); wall_ok=False
    if floor_ok and ranges.get("floor"): valid_ranges["floor"]=ranges["floor"]
    if wall_ok and ranges.get("wall"): valid_ranges["wall"]=ranges["wall"]
    if ceiling_ok and ranges.get("ceiling"): valid_ranges["ceiling"]=ranges["ceiling"]
    if not valid_ranges["floor"] and not valid_ranges["wall"] and not valid_ranges["ceiling"]: print("WARN: No reliable ranges after sanity.")
    elif not ranges.get("floor") or not ranges.get("wall") or not ranges.get("ceiling") or not valid_ranges["floor"] or not valid_ranges["wall"] or not valid_ranges["ceiling"]: print("WARN: Some ranges failed calc/sanity.")
    else: print("Sanity check passed.")
    print("--- Finished Calculating Surface Depth Ranges ---")
    return valid_ranges


# --- VALIDATION CHECK FUNCTIONS (Keep V16.1 versions) ---
def check_bounds(obj, dimensions):
    # ... (Keep as before) ...
    box = obj.get('bounding_box'); w, h = dimensions['width'], dimensions['height']
    if not isinstance(box,(list,tuple)) or len(box)!=4 or not all(isinstance(c,(int,float)) for c in box): return {"object":obj.get("object","?"),"check":"DataError","message":f"Invalid bbox format: {box}"}
    xmin,ymin,xmax,ymax = box; message=''
    if xmin<0 or ymin<0 or xmax>w or ymax>h: message = f"Box [{int(xmin)},{int(ymin)},{int(xmax)},{int(ymax)}] outside ({w}x{h})."
    elif xmin>=xmax or ymin>=ymax: message = f"Invalid dims: [{int(xmin)},{int(ymin)},{int(xmax)},{int(ymax)}]."
    if message: return {"object":obj["object"],"check":"Bounds","message":message}
    return None

def check_spatial_anchor(obj, depth_array, surface_ranges):
    # ... (Keep V16.1 version - compares Img Depth Median to Plane Z Range) ...
    anchor = obj.get('spatial_anchor'); box = obj.get('bounding_box');
    if anchor is None or box is None: return None
    expected_range_info = surface_ranges.get(anchor)
    if not expected_range_info: print(f"  Skipping anchor: {obj['object']} ('{anchor}') - Range invalid."); return None
    expected_min, expected_max = expected_range_info['min'], expected_range_info['max']
    h, w = depth_array.shape; clipped_box = clip_box(box, w, h)
    xmin, ymin, xmax, ymax = clipped_box
    if xmin >= xmax or ymin >= ymax: return None
    object_median_depth = get_median_depth_in_box(clipped_box, depth_array)
    if object_median_depth is None:
         if (xmax-xmin)*(ymax-ymin)>=config["MIN_BOX_AREA_FOR_DEPTH_SAMPLING"]: pass
         return None
    check_min = max(0, expected_min); check_max = min(65535, expected_max)
    if not (check_min <= object_median_depth <= check_max):
         range_str = f"[{expected_min:.0f}-{expected_max:.0f}]"
         msg = f"Anchor '{anchor}' failed. Object Img Depth ({object_median_depth:.0f}) outside Plane Z Range {range_str}."
         return {"object": obj["object"], "check": "SpatialAnchor", "message": msg}
    return None

def check_overlap(objA, objB, depth_array):
    # ... (Keep V16.1 version - uses Img Depth Medians) ...
    boxA = objA.get('bounding_box'); boxB = objB.get('bounding_box'); anchorA = objA.get('spatial_anchor'); anchorB = objB.get('spatial_anchor')
    if boxA is None or boxB is None or anchorA is None or anchorB is None: return None
    iou = calculate_iou(boxA, boxB);
    if iou == 0.0: return None
    if anchorA != anchorB: return None
    h, w = depth_array.shape; clipped_boxA, clipped_boxB = clip_box(boxA, w, h), clip_box(boxB, w, h)
    medianDepthA = get_median_depth_in_box(clipped_boxA, depth_array)
    medianDepthB = get_median_depth_in_box(clipped_boxB, depth_array)
    if medianDepthA is None or medianDepthB is None: return None
    depth_diff = abs(medianDepthA - medianDepthB)
    if depth_diff > config["DEPTH_THRESHOLD"]: return None # Use 16-bit threshold
    if iou > config["COLLISION_IOU_THRESHOLD"]:
        msg = (f"Collision risk. Img Depths (A:{medianDepthA:.0f},B:{medianDepthB:.0f}, Diff:{depth_diff:.0f}) "
               f"Overlap (IoU:{iou:.2f}>{config['COLLISION_IOU_THRESHOLD']}).")
        return {"object": objA["object"], "relatedObject": objB["object"], "check": "Overlap", "message": msg}
    return None

# --- MAIN VALIDATION FUNCTION (Keep robust V7 version) ---
def validateLayout(layout, img_dims, depth_arr):
    # ... (Keep robust loop as in V7) ...
    errors = [];
    if not OPEN3D_AVAILABLE: errors.append({"object":"System","check":"Setup","message":"Open3D unavailable."}); return errors
    if depth_arr is None: errors.append({"object":"System","check":"Setup","message":"Depth map failed."}); return errors
    surface_ranges = calculateSurfaceDepthRanges(depth_arr, img_dims) # Uses RANSAC results
    valid_indices = []
    for i, obj in enumerate(layout):
        obj_name = obj.get('object', f'UNKNOWN_Index_{i}'); box=obj.get('bounding_box'); anchor=obj.get('spatial_anchor'); data_ok=True
        if not isinstance(box,(list,tuple)) or len(box)!=4 or not all(isinstance(c,(int,float)) for c in box): errors.append({"object":obj_name,"check":"DataError","message":f"Invalid bbox:{box}"}); data_ok=False
        if anchor not in ['floor','wall','ceiling']: errors.append({"object":obj_name,"check":"DataError","message":f"Invalid anchor:{anchor}"}); data_ok=False
        if not data_ok: continue
        valid_indices.append(i)
        bounds_error=check_bounds(obj, img_dims); anchor_error=check_spatial_anchor(obj, depth_arr, surface_ranges)
        if bounds_error: errors.append(bounds_error)
        if anchor_error: errors.append(anchor_error)
    for i_idx in range(len(valid_indices)):
        for j_idx in range(i_idx + 1, len(valid_indices)):
            idx1,idx2=valid_indices[i_idx],valid_indices[j_idx]; objA,objB=layout[idx1],layout[idx2]
            overlap_error = check_overlap(objA, objB, depth_arr);
            if overlap_error: errors.append(overlap_error)
    return errors

# --- FLASK APP SETUP ---
app = Flask(__name__)

@app.route('/validate_layout', methods=['POST'])
def handle_validation():
    print("Received validation request...")
    # --- Input Validation ---
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json()
    layout = data.get('layout')
    img_dims = data.get('image_dimensions')
    depth_url = data.get('depth_map_url')

    if not all([layout, img_dims, depth_url]):
        return jsonify({"error": "Missing required fields: layout, image_dimensions, depth_map_url"}), 400
    if not isinstance(layout, list):
         return jsonify({"error": "Field 'layout' must be a list"}), 400
    if not isinstance(img_dims, dict) or 'width' not in img_dims or 'height' not in img_dims:
         return jsonify({"error": "Field 'image_dimensions' must be a dict with 'width' and 'height'"}), 400

    # --- Load Depth Map ---
    print(f"Fetching depth map from: {depth_url}")
    depth_data = load_depth_map(depth_url, is_url=True)

    if depth_data is None:
         return jsonify({"error": "Failed to load or process depth map from URL"}), 500
    if depth_data.dtype != np.uint16:
         # Should be handled by load_depth_map, but double check
         print(f"ERROR: Depth map loaded, but is not uint16 (dtype: {depth_data.dtype}). Check load_depth_map conversion.")
         return jsonify({"error": "Depth map processing failed, not uint16."}), 500
    if not OPEN3D_AVAILABLE:
         print("ERROR: Open3D is required but not installed.")
         return jsonify({"error": "Server configuration error: Open3D missing."}), 500

    # --- Run Validation ---
    try:
        validation_errors = validateLayout(layout, img_dims, depth_data)
    except Exception as e:
        print(f"CRITICAL ERROR during validation: {e}")
        # Add more detailed logging here if needed (traceback)
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error during validation."}), 500

    # --- Format Response ---
    if not validation_errors:
        print("Validation successful.")
        return jsonify({"status": "success", "errors": []})
    else:
        print(f"Validation failed with {len(validation_errors)} errors.")
        # Convert NumPy types in errors to standard Python types for JSON serialization
        serializable_errors = []
        for err in validation_errors:
            serializable_err = {}
            for k, v in err.items():
                if isinstance(v, np.integer):
                    serializable_err[k] = int(v)
                elif isinstance(v, np.floating):
                    serializable_err[k] = float(v)
                else:
                    serializable_err[k] = v
            serializable_errors.append(serializable_err)

        return jsonify({"status": "error", "errors": serializable_errors})

# --- Run Flask App ---
if __name__ == '__main__':
    # Note: For production, use a proper WSGI server like gunicorn or waitress
    app.run(debug=True, host='0.0.0.0', port=5001) # Runs on port 5001 