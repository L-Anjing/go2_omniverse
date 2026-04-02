import json
import numpy as np
import re # Needed for array formatting

def generate_lidar_config(
    name="Unitree L1 Lidar",
    near_range_m=0.2,
    far_range_m=20.0,
    start_azimuth_deg=0.0,
    end_azimuth_deg=360.0,
    up_elevation_deg=0.0,
    down_elevation_deg=-90.0,
    vertical_resolution_deg=1.0,
    horizontal_resolution_deg=1.0,
    scan_rate_base_hz=100.0,
    range_resolution_m=0.001,
    range_accuracy_m=0.02,
    avg_power_w=0.005,
    min_reflectance=0.1,
    min_reflectance_range=50.0,
    wavelength_nm=905.0,
    pulse_time_ns=6,
    azimuth_error_mean=0.0,
    azimuth_error_std=0.005,
    elevation_error_mean=0.0,
    elevation_error_std=0.005,
    max_returns=2
):
    """
    Generates a configuration dictionary for the Isaac Sim RTX Lidar.

    Args:
        name (str): Name of the LiDAR.
        near_range_m (float): Minimum range in meters.
        far_range_m (float): Maximum range in meters.
        start_azimuth_deg (float): Starting azimuth angle in degrees.
        end_azimuth_deg (float): Ending azimuth angle in degrees.
        up_elevation_deg (float): Upper elevation angle in degrees.
        down_elevation_deg (float): Lower elevation angle in degrees.
        vertical_resolution_deg (float): Angular spacing between vertical beams (degrees).
        horizontal_resolution_deg (float): Angular spacing between horizontal samples (degrees).
        scan_rate_base_hz (float): Scan frequency of the LiDAR (LiDAR FPS).
        range_resolution_m (float): Range resolution in meters.
        range_accuracy_m (float): Range accuracy in meters.
        avg_power_w (float): Average power in Watts.
        min_reflectance (float): Minimum reflectance for detection.
        min_reflectance_range (float): Range for minimum reflectance.
        wavelength_nm (float): Laser wavelength in nanometers.
        pulse_time_ns (int): Pulse time in nanoseconds.
        azimuth_error_mean (float): Mean of azimuth error.
        azimuth_error_std (float): Standard deviation of azimuth error.
        elevation_error_mean (float): Mean of elevation error.
        elevation_error_std (float): Standard deviation of elevation error.
        max_returns (int): Maximum number of returns per ray.
    """

    # Calculate the number of vertical emitters based on resolution
    vertical_fov_total = up_elevation_deg - down_elevation_deg
    num_emitters_vertical = int(round(vertical_fov_total / vertical_resolution_deg))
    if num_emitters_vertical == 0: # Ensure at least 1 emitter
        num_emitters_vertical = 1

    # Calculate the number of samples per horizontal scan
    horizontal_scan_total = end_azimuth_deg - start_azimuth_deg
    if horizontal_scan_total == 0: horizontal_scan_total = 360.0 # If 0, assume 360 for calculation
    num_samples_per_scan = int(round(horizontal_scan_total / horizontal_resolution_deg))
    if num_samples_per_scan == 0: num_samples_per_scan = 1 # Ensure at least 1 sample

    # Calculate reportRateBaseHz (total points per second)
    report_rate_base_hz = num_emitters_vertical * num_samples_per_scan * scan_rate_base_hz / 100

    # Calculate elevation angles for emitters
    if num_emitters_vertical > 1:
        # Adjust to center beams within the total range
        # First beam: down_elevation_deg + half of vertical resolution
        # Last beam: up_elevation_deg - half of vertical resolution
        elevation_deg_list = np.linspace(
            down_elevation_deg + vertical_resolution_deg / 2,
            up_elevation_deg - vertical_resolution_deg / 2,
            num_emitters_vertical
        ).round(2).tolist()
    else:
        elevation_deg_list = [(up_elevation_deg + down_elevation_deg) / 2.0]

    azimuth_deg_list = [0.0] * num_emitters_vertical
    fire_time_ns_list = [0] * num_emitters_vertical

    config = {
        "class": "sensor",
        "type": "lidar",
        "name": name,
        "driveWorksId": "GENERIC",
        "profile": {
            "scanType": "rotary",
            "intensityProcessing": "normalization",
            "rayType": "IDEALIZED",
            "nearRangeM": near_range_m,
            "farRangeM": far_range_m,
            "startAzimuthDeg": start_azimuth_deg,
            "endAzimuthDeg": end_azimuth_deg,
            "upElevationDeg": up_elevation_deg,
            "downElevationDeg": down_elevation_deg,
            "rangeResolutionM": range_resolution_m,
            "rangeAccuracyM": range_accuracy_m,
            "avgPowerW": avg_power_w,
            "minReflectance": min_reflectance,
            "minReflectanceRange": min_reflectance_range,
            "wavelengthNm": wavelength_nm,
            "pulseTimeNs": pulse_time_ns,
            "azimuthErrorMean": azimuth_error_mean,
            "azimuthErrorStd": azimuth_error_std,
            "elevationErrorMean": elevation_error_mean,
            "elevationErrorStd": elevation_error_std,
            "maxReturns": max_returns,
            "scanRateBaseHz": scan_rate_base_hz,
            "reportRateBaseHz": int(report_rate_base_hz),
            "numberOfEmitters": num_emitters_vertical,
            "emitters": {
                "elevationDeg": elevation_deg_list,
                "azimuthDeg": azimuth_deg_list,
                "fireTimeNs": fire_time_ns_list
            },
            "intensityMappingType": "LINEAR"
        }
    }
    return config

# --- Function to Reformat Arrays in JSON String (for visualization) ---
def reformat_json_array(json_str, key, items_per_line, indent_level):
    # Regex to find the array associated with the key
    pattern = rf'("{key}":\s*\[\n\s*)(.*?)(\n\s*\])'

    def replacer(match):
        prefix = match.group(1)
        content = match.group(2)
        suffix = match.group(3)

        numbers = [item.strip() for item in content.replace('\n', '').split(',') if item.strip()]

        if not numbers:
            return f'"{key}": []'

        formatted_lines = []
        inner_indent = " " * (indent_level * 2 + 2) # Indent for items inside the array

        for i in range(0, len(numbers), items_per_line):
            chunk = numbers[i : i + items_per_line]
            formatted_lines.append(inner_indent + ", ".join(chunk))

        close_bracket_indent = " " * (indent_level * 2)

        return f'"{key}": [\n' + ",\n".join(formatted_lines) + "\n" + close_bracket_indent + "]"

    return re.sub(pattern, replacer, json_str, flags=re.DOTALL)


# --- Configurations for Unitree L1 ---
config_l1_vertical_resolution_deg = 1.0
config_l1_horizontal_resolution_deg = 1.0
config_l1 = generate_lidar_config(
    name="Unitree L1 Lidar Default",
    up_elevation_deg=90.0,
    down_elevation_deg=0.0,
    vertical_resolution_deg=config_l1_vertical_resolution_deg,
    horizontal_resolution_deg=config_l1_horizontal_resolution_deg,
    scan_rate_base_hz=100.0,
)

# --- JSON File Generation and Saving ---
output_filename = "Unitree_L1.json"
# Note: This path is specific to your local setup and Isaac Sim installation.
# output_filename = "/home/vinicius/miniconda3/envs/env_isaaclab/lib/python3.10/site-packages/isaacsim/exts/isaacsim.sensors.rtx/data/lidar_configs/Unitree_L1.json"

# Generate the initial JSON string with indentation
json_string = json.dumps(config_l1, indent=2)

# Reformat long arrays to have 'NUM_ITEMS_PER_LINE' items per line
NUM_ITEMS_PER_LINE = 10
json_string = reformat_json_array(json_string, "elevationDeg", NUM_ITEMS_PER_LINE, 4)
json_string = reformat_json_array(json_string, "azimuthDeg", NUM_ITEMS_PER_LINE, 4)
json_string = reformat_json_array(json_string, "fireTimeNs", NUM_ITEMS_PER_LINE, 4)

with open(output_filename, 'w') as f:
    f.write(json_string)

print(f"Generated Configuration Details: '{config_l1['name']}'")
print(f"  Vertical FOV: {config_l1['profile']['downElevationDeg']} to {config_l1['profile']['upElevationDeg']} degrees.")
print(f"  Number of Vertical Emitters: {config_l1['profile']['numberOfEmitters']}")
print(f"  Vertical Resolution: {config_l1_vertical_resolution_deg} degrees.")
print(f"  Horizontal Resolution: {config_l1_horizontal_resolution_deg} degrees.")
print(f"  Estimated Total Points Per Second: {config_l1['profile']['reportRateBaseHz']}")
