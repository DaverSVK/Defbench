"""defectbench - reproducible benchmark for surface-defect detection and segmentation.

The package is deliberately split so that the light-weight parts (configuration,
unified prediction format, subprocess runners) import only ``numpy`` and ``yaml``.
Framework specific code (torchvision, Ultralytics, RF-DETR, DEIMv2, Detectron2,
MMDetection, ...) is imported lazily inside the corresponding adapter / runner.
"""

__version__ = "0.1.0"
