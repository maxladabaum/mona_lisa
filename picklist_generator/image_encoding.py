"""RGB image sampling, barcode encoding, and independent Echo runs."""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from picklist_core import ReplacementSelection, generate_picklist, replacement_fractions, calculate_mixing_volumes
from template_generator import EXTENSION_COLUMN_SPACING_NM, EXTENSION_ROW_SPACING_NM, EXTENSION_EXTRA_COLUMN_GAP_NM

IMAGE_WIDTH, IMAGE_HEIGHT = 24, 32
PIXEL_X_NM = EXTENSION_COLUMN_SPACING_NM / 4
PIXEL_Y_NM = EXTENSION_ROW_SPACING_NM / 4
ROWS = ["H{:02d}".format(row) for row in range(1, 16, 2)]
IMAGE_COLUMNS = ["{:02d}".format(column) for column in range(6, 0, -1)]
COLOR_BITS = {"R1 R": (True, True), "R1 B": (True, False), "R1 G": (False, False)}
CHANNELS = {"R1 R": 0, "R1 G": 1, "R1 B": 2}
SHIFT_STATES = {"0 — off / off": (False, False), "1 — off / on": (False, True),
                "2 — on / off": (True, False), "3 — on / on": (True, True)}
BIT_SITES = [tuple((row, column) for row in rows) for rows, column in (
    (("H01", "H03", "H05"), "11"), (("H01", "H03", "H05"), "09"),
    (("H01", "H03", "H05"), "07"), (("H11", "H13", "H15"), "11"),
    (("H11", "H13", "H15"), "09"), (("H11", "H13", "H15"), "07"))]
NINE_BIT_SITES = [tuple((row, column) for row in rows)
                  for rows, columns in ((("H01", "H03"), ("12", "10", "08")),
                                        (("H07", "H09"), ("11", "09", "07")),
                                        (("H13", "H15"), ("12", "10", "08")))
                  for column in columns]
FIT_MODES = ("Fit with black padding", "Center crop", "Stretch")


@dataclass(frozen=True)
class EncodingScheme:
    label: str
    x_states: int
    y_states: int
    joint: bool = True
    bit_count: int = 6
    parity: bool = False
    tiled: bool = False
    parent_scheme: str = ""

    @property
    def x_counts(self):
        return ((self.x_states+1)//2, self.x_states//2) if self.tiled else (self.x_states,)

    @property
    def y_counts(self):
        return ((self.y_states+1)//2, self.y_states//2) if self.tiled else (self.y_states,)

    @property
    def bit_sites(self):
        return NINE_BIT_SITES if self.bit_count == 9 else BIT_SITES

    @property
    def design_count(self):
        return 3 * self.x_states * self.y_states


DEFAULT_SCHEME = "fixed_4x4"
ENCODING_SCHEMES = {
    DEFAULT_SCHEME: EncodingScheme("4×4 — 24×32 pixels, 48 designs (original)", 4, 4, False),
    "joint_6x3": EncodingScheme("6×3 — 36×24 pixels, 54 designs (nearly square pixels)", 6, 3),
    "joint_7x3": EncodingScheme("7×3 — 42×24 pixels, 63 designs (maximum six-bit capacity)", 7, 3),
    "joint_5x4": EncodingScheme("5×4 — 30×32 pixels, 60 designs", 5, 4),
    "quadrants_2x2": EncodingScheme("4 quadrants — 24×32 pixels, 48 designs, 118.8×81.6 nm", 4, 4, False, tiled=True, parent_scheme=DEFAULT_SCHEME),
    "quadrants_joint_6x3": EncodingScheme("4 quadrants · 6×3 budget — 36×24 samples, 54 types", 6, 3, tiled=True, parent_scheme="joint_6x3"),
    "quadrants_joint_7x3": EncodingScheme("4 quadrants · 7×3 budget — 42×24 samples, 63 types", 7, 3, tiled=True, parent_scheme="joint_7x3"),
    "quadrants_joint_5x4": EncodingScheme("4 quadrants · 5×4 budget — 30×32 samples, 60 types", 5, 4, tiled=True, parent_scheme="joint_5x4"),
    "nine_fixed_16x8": EncodingScheme("9-bit fixed 16×8 — 96×64 pixels, 384 designs", 16, 8, False, 9),
    "nine_joint_18x9": EncodingScheme("9-bit joint 18×9 — 108×72 pixels, 486 designs", 18, 9, True, 9),
    "nine_parity_8x8": EncodingScheme("9-bit parity 8×8 — 48×64 pixels, 192 designs", 8, 8, False, 9, True),
}


def encoding_scheme(scheme_id=DEFAULT_SCHEME):
    try:
        return ENCODING_SCHEMES[scheme_id]
    except KeyError as exc:
        raise ValueError("Unknown image encoding scheme: {}".format(scheme_id)) from exc


def barcode_bits(color, x_shift, y_shift, scheme_id=DEFAULT_SCHEME):
    scheme = encoding_scheme(scheme_id)
    if color not in CHANNELS or not 0 <= x_shift < scheme.x_states or not 0 <= y_shift < scheme.y_states:
        raise ValueError("Color or shift is outside the selected encoding scheme.")
    if scheme.tiled and not scheme.joint:
        return list(map(int, COLOR_BITS[color])) + [y_shift//2, x_shift//2, x_shift%2, y_shift%2]
    if not scheme.joint:
        x_bits, y_bits = (scheme.x_states-1).bit_length(), (scheme.y_states-1).bit_length()
        bits = list(map(int, COLOR_BITS[color]))
        bits += [(x_shift >> shift) & 1 for shift in range(x_bits-1, -1, -1)]
        bits += [(y_shift >> shift) & 1 for shift in range(y_bits-1, -1, -1)]
        if scheme.parity:
            bits.append(sum(bits) % 2)  # Even parity across all nine bits.
        return bits
    code = CHANNELS[color] * scheme.x_states * scheme.y_states + y_shift * scheme.x_states + x_shift
    return [(code >> shift) & 1 for shift in range(scheme.bit_count-1, -1, -1)]


def rotate_point(x, y, span_x, span_y, rotation_degrees):
    """Clockwise rotation in image coordinates, translated to a nonnegative frame."""
    if rotation_degrees == 0:
        return x, y
    if rotation_degrees == 90:
        return span_y - y, x
    if rotation_degrees == 180:
        return span_x - x, span_y - y
    if rotation_degrees == 270:
        return y, span_x - x
    raise ValueError("Origami rotation must be 0, 90, 180, or 270 degrees clockwise.")


def rotate_pixels(pixels, rotation_degrees):
    rotate_point(0, 0, 0, 0, rotation_degrees)
    result = pixels
    for _ in range(rotation_degrees // 90):
        result = [list(row) for row in zip(*result[::-1])]
    return result


def physical_raster_size(scheme_id=DEFAULT_SCHEME):
    """Square-pixel raster with exact physical aspect and whole cells for every bin."""
    scheme = encoding_scheme(scheme_id)
    width, height = 6 * scheme.x_states, 8 * scheme.y_states
    aspect = Fraction(str(EXTENSION_COLUMN_SPACING_NM)) * 6 / (Fraction(str(EXTENSION_ROW_SPACING_NM)) * 8)
    factor = math.lcm(width // math.gcd(width, aspect.numerator), height // math.gcd(height, aspect.denominator))
    return aspect.numerator * factor, aspect.denominator * factor


def physical_reference_raster(raster, scheme_id=DEFAULT_SCHEME, rotation_degrees=0):
    """Render a sample grid using its physical cell sizes, including unequal quadrants."""
    from PIL import Image
    scheme = encoding_scheme(scheme_id)
    size = physical_raster_size(scheme_id)
    if scheme.tiled:
        result = Image.new("RGB",size)
        for qy,ny in enumerate(scheme.y_counts):
            for qx,nx in enumerate(scheme.x_counts):
                x0,y0 = qx*6*scheme.x_counts[0],qy*8*scheme.y_counts[0]
                patch = raster.crop((x0,y0,x0+6*nx,y0+8*ny))
                left,top = qx*size[0]//2,qy*size[1]//2
                right,bottom = (qx+1)*size[0]//2,(qy+1)*size[1]//2
                result.paste(patch.resize((right-left,bottom-top),Image.Resampling.NEAREST),(left,top))
    else:
        result = raster.resize(size,Image.Resampling.NEAREST)
    return result.rotate(-rotation_degrees,expand=True)


def site_name(row, column):
    return "U-Apt_{}-{}_R1".format(row, column)


def axis_placement(counts, shift, site_count, site_pitch):
    quadrant = int(len(counts) > 1 and shift >= counts[0])
    local = shift - (counts[0] if quadrant else 0)
    pitch = site_pitch/counts[quadrant]
    # Half-cell correction keeps equal-size quadrant boundaries when pitches differ.
    offset = quadrant*site_count*site_pitch + (pitch-site_pitch/counts[0])/2
    return quadrant, local, counts[quadrant], pitch, offset


def sampling_axis(counts, site_count, site_pitch):
    centers, edges = [], [0.]
    weights = []
    first_pitch = site_pitch/counts[0]
    for quadrant, count in enumerate(counts):
        pitch = site_pitch/count
        offset = quadrant*site_count*site_pitch + (pitch-first_pitch)/2
        for index in range(site_count*count):
            centers.append(offset + index*pitch)
            edges.append(quadrant*site_count*site_pitch + (index+1)*pitch)
            weights.append(sum(counts)/(len(counts)*count))
    return centers, edges, weights


def placement(scheme, x, y):
    if scheme.tiled:
        qx,lx,nx,px,ox = axis_placement(scheme.x_counts,x,6,EXTENSION_COLUMN_SPACING_NM)
        qy,ly,ny,py,oy = axis_placement(scheme.y_counts,y,8,EXTENSION_ROW_SPACING_NM)
        return {"quadrant": 2*qy+qx, "quadrant_x": qx, "quadrant_y": qy,
                "local_x_shift": lx, "local_y_shift": ly,
                "local_x_states": nx, "local_y_states": ny,
                "quadrant_offset_x_nm": ox, "quadrant_offset_y_nm": oy,
                "local_x_shift_nm": lx*px, "local_y_shift_nm": ly*py,
                "x_shift_nm": ox+lx*px, "y_shift_nm": oy+ly*py,
                "quadrant_normalization_weight": scheme.x_states*scheme.y_states/(4*nx*ny)}
    return {"x_shift_nm": x*EXTENSION_COLUMN_SPACING_NM/scheme.x_states,
            "y_shift_nm": y*EXTENSION_ROW_SPACING_NM/scheme.y_states}


def geometry(scheme_id=DEFAULT_SCHEME, rotation_degrees=0):
    scheme = encoding_scheme(scheme_id)
    width, height = 6 * scheme.x_states, 8 * scheme.y_states
    pixel_x, pixel_y = EXTENSION_COLUMN_SPACING_NM / scheme.x_counts[0], EXTENSION_ROW_SPACING_NM / scheme.y_counts[0]
    rotate_point(0, 0, 0, 0, rotation_degrees)
    portrait = rotation_degrees in (90, 270)
    codebook = []
    for color in CHANNELS:
        for y in range(scheme.y_states):
            for x in range(scheme.x_states):
                bits = barcode_bits(color, x, y, scheme_id)
                codebook.append({"code": int("".join(map(str, bits)), 2), "bits": bits,
                                 "color": color, "x_shift": x, "y_shift": y,
                                 **placement(scheme,x,y)})
    extent = 2 if scheme.tiled else 1
    centers_x, edges_x, weights_x = sampling_axis(scheme.x_counts,6,EXTENSION_COLUMN_SPACING_NM)
    centers_y, edges_y, weights_y = sampling_axis(scheme.y_counts,8,EXTENSION_ROW_SPACING_NM)
    last_x, last_y = EXTENSION_COLUMN_SPACING_NM/scheme.x_counts[-1], EXTENSION_ROW_SPACING_NM/scheme.y_counts[-1]
    display_first_x, display_first_y = {0:(pixel_x,pixel_y),90:(last_y,pixel_x),180:(last_x,last_y),270:(pixel_y,last_x)}[rotation_degrees]
    result = {
        "origami_rotation_degrees_clockwise": rotation_degrees,
        "source_image_rotation_degrees": 0,
        "array_coordinate_frame": "origami-local; rotate clockwise by origami_rotation_degrees_clockwise for upright image display",
        "display_columns": height if portrait else width, "display_rows": width if portrait else height,
        "display_pixel_pitch_x_nm": display_first_x,
        "display_pixel_pitch_y_nm": display_first_y,
        "display_field_width_nm": extent * (8 * EXTENSION_ROW_SPACING_NM if portrait else 6 * EXTENSION_COLUMN_SPACING_NM),
        "display_field_height_nm": extent * (6 * EXTENSION_COLUMN_SPACING_NM if portrait else 8 * EXTENSION_ROW_SPACING_NM),
        "encoding_scheme": scheme_id, "encoding_label": scheme.label,
        "encoding_mode": "joint_color_xy" if scheme.joint else "fixed_color_x_y_bits",
        "bit_count": scheme.bit_count, "sites_per_bit": len(scheme.bit_sites[0]),
        "parity": "even; bit 9 checks bits 1–8" if scheme.parity else None,
        "bit_order": "bit 1 is the most significant bit; bit {} is the least significant bit".format(scheme.bit_count),
        "code_formula": ("channel_index * x_states * y_states + y_shift * x_states + x_shift"
                         if scheme.joint else ("payload = color_code * {} + x_shift * {} + y_shift; code = (payload << 1) | (popcount(payload) % 2)".format(scheme.x_states*scheme.y_states,scheme.y_states)
                                              if scheme.parity else "color_code * {} + x_shift * {} + y_shift".format(scheme.x_states*scheme.y_states,scheme.y_states))),
        "channel_indices": CHANNELS, "codebook": codebook,
        "reserved_codes": sorted(set(range(1 << scheme.bit_count)) - {entry["code"] for entry in codebook}),
        "x_states": scheme.x_states, "y_states": scheme.y_states, "design_count": scheme.design_count,
        "columns": width, "rows": height,
        "site_pitch_x_nm": EXTENSION_COLUMN_SPACING_NM,
        "site_pitch_y_nm": EXTENSION_ROW_SPACING_NM,
        "pixel_pitch_x_nm": pixel_x, "pixel_pitch_y_nm": pixel_y,
        "field_width_nm": extent * 6 * EXTENSION_COLUMN_SPACING_NM,
        "field_height_nm": extent * 8 * EXTENSION_ROW_SPACING_NM,
        "center_span_x_nm": centers_x[-1], "center_span_y_nm": centers_y[-1],
        "pixel_centers_x_nm": centers_x, "pixel_centers_y_nm": centers_y,
        "pixel_edges_x_nm": edges_x, "pixel_edges_y_nm": edges_y,
        "sampling_weights_x": weights_x, "sampling_weights_y": weights_y,
        "nonuniform_sampling": len(set(scheme.x_counts))>1 or len(set(scheme.y_counts))>1,
        "parent_scheme": scheme.parent_scheme,
        "image_columns_left_to_right": IMAGE_COLUMNS,
        "image_rows_top_to_bottom": ROWS,
        "image_origin_from_C12_H01_nm": [6 * EXTENSION_COLUMN_SPACING_NM + EXTENSION_EXTRA_COLUMN_GAP_NM, 0],
        "positive_shift_direction": "right / down",
        "barcode_bits": [{"bit": index + 1, "sites": list(sites)} for index, sites in enumerate(scheme.bit_sites)],
        "pixel_mapping": "pixel_x = site_column_index * x_states + x_shift; pixel_y = site_row_index * y_states + y_shift",
    }
    if scheme.tiled:
        result.update(encoding_mode="joint_color_quadrant_local_xy" if scheme.joint else "color_quadrant_local_xy",
            quadrant_count=4, tile_field_width_nm=59.4, tile_field_height_nm=40.8,
            local_x_states=scheme.x_counts[0] if len(set(scheme.x_counts))==1 else list(scheme.x_counts),
            local_y_states=scheme.y_counts[0] if len(set(scheme.y_counts))==1 else list(scheme.y_counts),
            quadrant_order=["top-left", "top-right", "bottom-left", "bottom-right"],
            quadrant_sampling=[{"quadrant":2*qy+qx,"x_states":nx,"y_states":ny,"design_count":3*nx*ny,
                                "pixel_pitch_x_nm":9.9/nx,"pixel_pitch_y_nm":5.1/ny,
                                "normalization_weight":scheme.x_states*scheme.y_states/(4*nx*ny)}
                               for qy,ny in enumerate(scheme.y_counts) for qx,nx in enumerate(scheme.x_counts)],
            bit_meaning="Joint six-bit codebook identifies color, quadrant and local XY shift." if scheme.joint else
                        "bits 1–2 color; bit 3 quadrant Y; bit 4 quadrant X; bit 5 local X; bit 6 local Y",
            pixel_mapping="Concatenate quadrant sampling grids; use pixel_centers/edges for nonuniform physical spacing.",
            quadrant_offset_convention="field offset plus half the difference between this quadrant's pitch and the first quadrant's pitch")
        if not scheme.joint:
            result["code_formula"] = "color_code * 16 + quadrant_y * 8 + quadrant_x * 4 + local_x * 2 + local_y"
    if not scheme.joint:
        result["color_bits"] = COLOR_BITS
        result["x_shift_bits"] = 1 if scheme.tiled else (scheme.x_states-1).bit_length()
        result["y_shift_bits"] = 1 if scheme.tiled else (scheme.y_states-1).bit_length()
        if scheme_id == DEFAULT_SCHEME:
            result["shift_state_bits"] = {str(index): list(bits) for index, bits in enumerate(SHIFT_STATES.values())}
    return result


def bin_quadrants(fitted, scheme, rotation_degrees, linear_light):
    import numpy as np
    from PIL import Image
    from color_light import display_rgb
    local = np.rot90(np.asarray(fitted), rotation_degrees//90)
    result = np.zeros((8*scheme.y_states,6*scheme.x_states,3),dtype=float)
    for qy,ny in enumerate(scheme.y_counts):
        for qx,nx in enumerate(scheme.x_counts):
            patch = local[qy*local.shape[0]//2:(qy+1)*local.shape[0]//2,
                          qx*local.shape[1]//2:(qx+1)*local.shape[1]//2]
            x0,y0 = qx*6*scheme.x_counts[0],qy*8*scheme.y_counts[0]
            for c in range(3):
                plane = Image.fromarray(patch[:,:,c].astype(np.float32))
                result[y0:y0+8*ny,x0:x0+6*nx,c] = np.asarray(plane.resize((6*nx,8*ny),Image.Resampling.BOX))
    return np.asarray(display_rgb(result,True)).tolist() if linear_light else np.rint(result).astype(int).tolist()


def bin_image(path, mode=FIT_MODES[0], rotation_degrees=0, scheme_id=DEFAULT_SCHEME, linear_light=False):
    """Fit the upright source to a rotated origami field; return origami-local pixels."""
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise ValueError("RGB import requires Pillow. Run: python -m pip install -r requirements.txt") from exc
    if mode not in FIT_MODES:
        raise ValueError("Unknown image fitting mode.")
    if rotation_degrees not in (0, 90, 180, 270):
        raise ValueError("Rotation must be 0, 90, 180, or 270 degrees clockwise.")
    with Image.open(path) as source:
        oriented = ImageOps.exif_transpose(source).convert("RGBA")
        background = Image.new("RGBA", oriented.size, (0, 0, 0, 255))
        background.alpha_composite(oriented)
        rgb = background.convert("RGB")
    scheme = encoding_scheme(scheme_id)
    width, height = 6 * scheme.x_states, 8 * scheme.y_states
    physical_raster = physical_raster_size(scheme_id)
    if rotation_degrees in (90, 270):
        width, height = height, width
        physical_raster = physical_raster[::-1]
    if linear_light:
        from color_light import fit_linear_source, display_rgb
        if scheme.tiled:
            return bin_quadrants(fit_linear_source(path,mode,physical_raster),scheme,rotation_degrees,True)
        binned = display_rgb(fit_linear_source(path, mode, physical_raster, (width,height)), True)
        upright = [[list(binned.getpixel((x,y))) for x in range(width)] for y in range(height)]
        return rotate_pixels(upright, (-rotation_degrees) % 360)
    if mode == FIT_MODES[0]:
        fitted = ImageOps.pad(rgb, physical_raster, method=Image.Resampling.LANCZOS, color="black")
    elif mode == FIT_MODES[1]:
        fitted = ImageOps.fit(rgb, physical_raster, method=Image.Resampling.LANCZOS)
    else:
        fitted = rgb.resize(physical_raster, Image.Resampling.LANCZOS)
    if scheme.tiled:
        return bin_quadrants(fitted,scheme,rotation_degrees,False)
    binned = fitted.resize((width, height), Image.Resampling.BOX)
    upright = [[list(binned.getpixel((x, y))) for x in range(width)] for y in range(height)]
    return rotate_pixels(upright, (-rotation_degrees) % 360)


def compile_designs(pixels, base_volume, multipliers, scheme_id=DEFAULT_SCHEME, rotation_degrees=0, linear_light=False):
    """Deinterleave each RGB channel using the selected barcode codebook."""
    scheme = encoding_scheme(scheme_id)
    width, height = 6 * scheme.x_states, 8 * scheme.y_states
    pixel_x, pixel_y = EXTENSION_COLUMN_SPACING_NM / scheme.x_counts[0], EXTENSION_ROW_SPACING_NM / scheme.y_counts[0]
    if len(pixels) != height or any(len(row) != width for row in pixels):
        raise ValueError("RGB image must contain exactly {} columns and {} rows.".format(width, height))
    for row in pixels:
        for pixel in row:
            if len(pixel) != 3 or any(not isinstance(value, int) or not 0 <= value <= 255 for value in pixel):
                raise ValueError("Each pixel must have three integer RGB values from 0 to 255.")
    rotate_point(0, 0, 0, 0, rotation_degrees)
    info = geometry(scheme_id,rotation_degrees)
    from color_light import srgb_to_linear
    intensity_lut = srgb_to_linear([v/255 for v in range(256)]) if linear_light else [v/255 for v in range(256)]
    designs = []
    for color, channel in CHANNELS.items():
        multiplier = multipliers[color]
        steps = len(replacement_fractions(base_volume, multiplier)) - 1
        for y_shift in range(scheme.y_states):
            for x_shift in range(scheme.x_states):
                bits = barcode_bits(color, x_shift, y_shift, scheme_id)
                position = placement(scheme,x_shift,y_shift)
                fractions = {}
                for state, sites in zip(bits, scheme.bit_sites):
                    for row, column in sites:
                        fractions[site_name(row, column)] = float(state)
                sites = []
                for row_index, row in enumerate(ROWS):
                    for column_index, column in enumerate(IMAGE_COLUMNS):
                        x, y = column_index * scheme.x_states + x_shift, row_index * scheme.y_states + y_shift
                        if scheme.tiled:
                            x = position["quadrant_x"]*6*scheme.x_counts[0] + column_index*position["local_x_states"] + position["local_x_shift"]
                            y = position["quadrant_y"]*8*scheme.y_counts[0] + row_index*position["local_y_states"] + position["local_y_shift"]
                        value = pixels[y][x][channel]
                        droplets = math.floor(float(intensity_lut[value]) * steps + 0.5)
                        fraction = droplets / steps
                        name = site_name(row, column)
                        fractions[name] = fraction
                        display_x, display_y = rotate_point(x, y, width - 1, height - 1, rotation_degrees)
                        x_nm,y_nm = info["pixel_centers_x_nm"][x],info["pixel_centers_y_nm"][y]
                        display_x_nm, display_y_nm = rotate_point(x_nm,y_nm,
                            info["center_span_x_nm"],info["center_span_y_nm"],rotation_degrees)
                        sites.append({
                            "display_pixel_x": display_x, "display_pixel_y": display_y,
                            "display_x_nm": display_x_nm, "display_y_nm": display_y_nm,
                            "name": name, "row": row, "column": column,
                            "pixel_x": x, "pixel_y": y, "rgb": pixels[y][x][:],
                            "x_nm": x_nm, "y_nm": y_nm,
                            "channel_value": value, "replacement_fraction": fraction,
                            "replacement_nl": droplets * 25, "original_nl": (steps - droplets) * 25,
                        })
                suffix = "" if scheme_id == DEFAULT_SCHEME else "_{}x{}".format(scheme.x_states,scheme.y_states)
                design_name = ("{}{}_Q{}_X{}_Y{}".format(color.replace(" ","_"),suffix,position["quadrant"],position["local_x_shift"],position["local_y_shift"])
                               if scheme.tiled else "{}{}_X{}_Y{}".format(color.replace(" ","_"),suffix,x_shift,y_shift))
                designs.append({
                    "name": design_name,
                    "origami_rotation_degrees_clockwise": rotation_degrees,
                    "linear_light": linear_light, "encoding_scheme": scheme_id, "barcode_code": int("".join(map(str, bits)), 2),
                    "color": color, "channel": channel, "x_shift": x_shift, "y_shift": y_shift,
                    **position,
                    "bits": [int(bit) for bit in bits], "base_volume_nl": base_volume,
                    "excess_multiplier": multiplier, "site_fractions": fractions, "image_sites": sites,
                })
    return designs


def reconstruct(designs):
    scheme_id = designs[0].get("encoding_scheme", DEFAULT_SCHEME) if designs else DEFAULT_SCHEME
    if any(design.get("encoding_scheme", DEFAULT_SCHEME) != scheme_id for design in designs):
        raise ValueError("Cannot reconstruct designs from different encoding schemes together.")
    scheme = encoding_scheme(scheme_id)
    pixels = [[[0, 0, 0] for _ in range(6 * scheme.x_states)] for _ in range(8 * scheme.y_states)]
    for design in designs:
        for site in design["image_sites"]:
            pixels[site["pixel_y"]][site["pixel_x"]][design["channel"]] = round(site["replacement_fraction"] * 255)
    return pixels


def selection_panel(design):
    fractions = design["site_fractions"]
    selected = [(row, column) for row in ROWS for column in ("{:02d}".format(c) for c in range(12, 0, -1))
                if fractions.get(site_name(row, column), 0) > 0]
    return {
        "origami_rotation_degrees_clockwise": design.get("origami_rotation_degrees_clockwise", 0),
        "encoding_scheme": design.get("encoding_scheme", DEFAULT_SCHEME),
        "group": design["color"], "label": design["name"], "rows": ROWS[:],
        "columns": ["{:02d}".format(c) for c in range(12, 0, -1)],
        "active": [(row, "{:02d}".format(c)) for row in ROWS for c in range(1, 13)],
        "selected": selected, "selected_names": [site_name(row, column) for row, column in selected],
        "site_fractions": dict(fractions), "x_shift": design["x_shift"], "y_shift": design["y_shift"],
        "spacing_x_nm": EXTENSION_COLUMN_SPACING_NM, "spacing_y_nm": EXTENSION_ROW_SPACING_NM,
        "mirror_columns": True,
    }


def prepare_runs(designs, base_path, sources, available_wells, settings, mixing):
    """Allocate separate wells and validate every recipe before returning any runs."""
    remaining = list(available_wells)
    if len(set(remaining)) != len(remaining):
        raise ValueError("Destination wells must be unique.")
    if len(remaining) < len(designs):
        raise ValueError("This batch needs at least {} destination wells; only {} are available on the selected plate.".format(len(designs),len(remaining)))
    runs = []
    for design in designs:
        source_path, source_plate = sources[design["color"]]
        selection = ReplacementSelection(Path(source_path), list(design["site_fractions"]), source_plate,
                                         design["excess_multiplier"], site_fractions=design["site_fractions"])
        try:
            picklist = generate_picklist(Path(base_path), [selection], remaining,
                                        transfer_volume_nl=design["base_volume_nl"], **settings)
            occupied = list(dict.fromkeys(row["Destination Well"] for row in picklist))
            remaining = [well for well in remaining if well not in occupied]
            total = sum(row["Transfer Volume"] for row in picklist)
            effective_staples = total / (design["base_volume_nl"] * settings["transfers_per_source"])
            recipe = calculate_mixing_volumes(number_of_staples=effective_staples, **mixing)
            if recipe[0]["Volume_uL"] > total / 1000 + 1e-9:
                raise ValueError("Requested staple mix exceeds the available picklist volume; adjust Run & Mixing Settings.")
        except ValueError as exc:
            raise ValueError("{}: {}".format(design["name"], exc)) from exc
        runs.append({"run_name": design["name"], "design_names": [name for name, fraction in design["site_fractions"].items() if fraction > 0],
                     "hinge_type": settings["hinge_type"], "selection_panels": [selection_panel(design)],
                     "destination_plate_name": settings["destination_plate_name"], "destination_wells": occupied,
                     "transfer_count": len(picklist), "picklist": picklist, "mixing_recipe": recipe,
                     "image_design": design})
    return runs
