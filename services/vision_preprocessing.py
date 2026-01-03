import os
from typing import List, Union
from PIL import Image
from pdf2image import convert_from_path
import numpy as np
import cv2

class VisionPreprocessor:
    """
    Preprocessing class for the vision module.
    - Converts input PDF document to images
    - Applies minimal preprocessing for ViT compatibility
    """

    def __init__(self, dpi: int = 300, image_format: str = "RGB"):
        self.dpi = dpi
        self.image_format = image_format


    def convert_pdf_to_image(self, pdf_path: str) -> List[Image.Image]:
        """
        Converts each page of a PDF to a PIL Image.
        Args:
            pdf_path (str): Path to the PDF file.
        Returns:
            List[Image.Image]: List of PIL Images, one per page.
        """
        images = convert_from_path(pdf_path, dpi=self.dpi)
        # Ensure all images are in the desired format
        images = [img.convert(self.image_format) for img in images]
        return [images[0]]

    def preprocess(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Applies minimal preprocessing: denoising, deskewing, format conversion.
        Args:
            image (PIL.Image or np.ndarray): Input image.
        Returns:
            np.ndarray: Preprocessed image as numpy array.
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Denoising
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        # Deskewing
        image = self._deskew(image)

        # Ensure RGB format for ViT
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return image

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Deskew image using moments. Returns deskewed image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        coords = np.column_stack(np.where(gray > 0))
        if coords.size == 0:
            return image
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return deskewed