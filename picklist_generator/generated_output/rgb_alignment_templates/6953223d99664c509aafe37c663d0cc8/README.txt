PAINT Analysis alignment templates

Load the PNGs in barcodes/ as custom barcode templates for classification.
Each contains the ON barcode groups and permanent fiducials for its design.
fiducials/ contains shared fiducial-only references, when defined.
PNG files embed nm-per-pixel calibration and the complete logical-bit model;
the same metadata is provided in adjacent JSON files.
Templates use the full origami lattice; analog image pixels are excluded.
manifest.json maps filenames to design names, codes, colors, shifts and wells.
All-off designs without fiducials have no bright sites and are listed as skipped.
