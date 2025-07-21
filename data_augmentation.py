"""
Advanced Data Augmentation for HTR Models
This module provides various augmentation techniques specifically designed for handwritten text recognition
"""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import cv2


class HTRDataAugmentation:
    """Advanced data augmentation for handwritten text recognition"""
    
    def __init__(self, target_height=40, augment_prob=0.5):
        self.target_height = target_height
        self.augment_prob = augment_prob
        
    def __call__(self, image):
        """Apply random augmentations to the image"""
        if random.random() > self.augment_prob:
            return image
        
        # Convert to PIL if tensor
        if torch.is_tensor(image):
            image = TF.to_pil_image(image)
        
        # Apply random augmentations
        augmentations = [
            self.random_perspective,
            self.random_rotation,
            self.random_shear,
            self.random_blur,
            self.random_noise,
            self.random_brightness_contrast,
            self.random_elastic_transform,
            self.random_erosion_dilation,
        ]
        
        # Apply 1-3 random augmentations
        num_augs = random.randint(1, 3)
        selected_augs = random.sample(augmentations, num_augs)
        
        for aug in selected_augs:
            try:
                image = aug(image)
            except Exception as e:
                print(f"Augmentation failed: {e}")
                continue
        
        return image
    
    def random_perspective(self, image):
        """Apply random perspective transformation"""
        if random.random() < 0.3:
            width, height = image.size
            
            # Define perspective transformation strength
            strength = 0.1
            
            # Random corner displacement
            corners = [
                (0, 0),
                (width, 0),
                (width, height),
                (0, height)
            ]
            
            new_corners = []
            for x, y in corners:
                dx = random.uniform(-strength * width, strength * width)
                dy = random.uniform(-strength * height, strength * height)
                new_corners.extend([x + dx, y + dy])
            
            # Apply perspective transformation
            image = image.transform(
                image.size,
                Image.PERSPECTIVE,
                new_corners,
                resample=Image.BILINEAR
            )
        
        return image
    
    def random_rotation(self, image):
        """Apply small random rotation"""
        if random.random() < 0.4:
            angle = random.uniform(-3, 3)  # Small rotation angles
            image = image.rotate(angle, expand=True, fillcolor='white')
        return image
    
    def random_shear(self, image):
        """Apply random shear transformation"""
        if random.random() < 0.3:
            shear_x = random.uniform(-0.1, 0.1)
            shear_y = random.uniform(-0.05, 0.05)
            
            # Apply shear using affine transformation
            image = TF.affine(
                image, 
                angle=0, 
                translate=[0, 0], 
                scale=1, 
                shear=[shear_x * 180 / np.pi, shear_y * 180 / np.pi],
                fillcolor='white'
            )
        return image
    
    def random_blur(self, image):
        """Apply random blur"""
        if random.random() < 0.2:
            radius = random.uniform(0.5, 1.5)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        return image
    
    def random_noise(self, image):
        """Add random noise"""
        if random.random() < 0.3:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Add Gaussian noise
            noise = np.random.normal(0, 10, img_array.shape)
            noisy_img = img_array + noise
            noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
            
            image = Image.fromarray(noisy_img)
        
        return image
    
    def random_brightness_contrast(self, image):
        """Adjust brightness and contrast"""
        if random.random() < 0.5:
            # Brightness
            brightness_factor = random.uniform(0.8, 1.2)
            image = ImageEnhance.Brightness(image).enhance(brightness_factor)
            
            # Contrast
            contrast_factor = random.uniform(0.8, 1.2)
            image = ImageEnhance.Contrast(image).enhance(contrast_factor)
        
        return image
    
    def random_elastic_transform(self, image):
        """Apply elastic transformation for handwriting variation"""
        if random.random() < 0.2:
            img_array = np.array(image)
            
            # Create displacement fields
            h, w = img_array.shape[:2]
            dx = np.random.uniform(-1, 1, (h//8, w//8)) * 2
            dy = np.random.uniform(-1, 1, (h//8, w//8)) * 2
            
            # Resize displacement fields
            dx = cv2.resize(dx, (w, h), interpolation=cv2.INTER_CUBIC)
            dy = cv2.resize(dy, (w, h), interpolation=cv2.INTER_CUBIC)
            
            # Create coordinate grids
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            x = (x + dx).astype(np.float32)
            y = (y + dy).astype(np.float32)
            
            # Apply transformation
            if len(img_array.shape) == 3:
                transformed = cv2.remap(img_array, x, y, cv2.INTER_LINEAR, borderValue=255)
            else:
                transformed = cv2.remap(img_array, x, y, cv2.INTER_LINEAR, borderValue=255)
            
            image = Image.fromarray(transformed)
        
        return image
    
    def random_erosion_dilation(self, image):
        """Apply morphological operations to simulate pen thickness variation"""
        if random.random() < 0.2:
            img_array = np.array(image)
            
            # Convert to binary
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Threshold
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Random kernel size
            kernel_size = random.randint(1, 3)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # Random operation
            if random.random() < 0.5:
                # Erosion (thinner text)
                processed = cv2.erode(binary, kernel, iterations=1)
            else:
                # Dilation (thicker text)
                processed = cv2.dilate(binary, kernel, iterations=1)
            
            # Convert back
            processed = cv2.bitwise_not(processed)
            
            if len(img_array.shape) == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            
            image = Image.fromarray(processed)
        
        return image


class SyntheticDataGenerator:
    """Generate synthetic handwritten text data"""
    
    def __init__(self, vocab, target_height=40):
        self.vocab = vocab
        self.target_height = target_height
        
        # Try to load fonts for synthetic generation
        self.fonts = self._load_fonts()
    
    def _load_fonts(self):
        """Load available fonts for text generation"""
        fonts = []
        try:
            from PIL import ImageFont
            
            # Common handwriting-like fonts
            font_paths = [
                "arial.ttf",
                "times.ttf", 
                "calibri.ttf",
                # Add more font paths as needed
            ]
            
            for font_path in font_paths:
                try:
                    for size in [16, 18, 20, 22, 24]:
                        fonts.append(ImageFont.truetype(font_path, size))
                except:
                    continue
            
            # Fallback to default font
            if not fonts:
                fonts = [ImageFont.load_default()]
        
        except ImportError:
            fonts = [None]  # Will use default PIL font
        
        return fonts
    
    def generate_synthetic_sample(self, text):
        """Generate a synthetic handwritten text image"""
        if not text:
            return None
        
        # Choose random font
        font = random.choice(self.fonts) if self.fonts else None
        
        # Create image with text
        # Estimate text size
        temp_img = Image.new('RGB', (1000, 100), color='white')
        draw = ImageDraw.Draw(temp_img)
        
        if font:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            bbox = draw.textbbox((0, 0), text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        
        # Create properly sized image
        margin = 10
        img_width = text_width + 2 * margin
        img_height = max(text_height + 2 * margin, self.target_height)
        
        image = Image.new('RGB', (img_width, img_height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Draw text
        text_x = margin
        text_y = (img_height - text_height) // 2
        
        if font:
            draw.text((text_x, text_y), text, fill='black', font=font)
        else:
            draw.text((text_x, text_y), text, fill='black')
        
        # Apply augmentations to make it look more natural
        augmenter = HTRDataAugmentation(target_height=self.target_height, augment_prob=0.8)
        image = augmenter(image)
        
        return image


def create_augmented_dataset(original_dataset, augmentation_factor=2):
    """Create an augmented version of the dataset"""
    
    class AugmentedDataset(torch.utils.data.Dataset):
        def __init__(self, original_dataset, augmentation_factor):
            self.original_dataset = original_dataset
            self.augmentation_factor = augmentation_factor
            self.augmenter = HTRDataAugmentation(augment_prob=0.7)
            
        def __len__(self):
            return len(self.original_dataset) * self.augmentation_factor
        
        def __getitem__(self, idx):
            # Get original sample
            original_idx = idx % len(self.original_dataset)
            image, label = self.original_dataset[original_idx]
            
            # Apply augmentation if not the first occurrence
            if idx >= len(self.original_dataset):
                image = self.augmenter(image)
            
            return image, label
        
        @property
        def vocab(self):
            return self.original_dataset.vocab
    
    return AugmentedDataset(original_dataset, augmentation_factor)


def test_augmentations():
    """Test the augmentation pipeline"""
    # Create a simple test image
    test_image = Image.new('RGB', (300, 40), color='white')
    draw = ImageDraw.Draw(test_image)
    draw.text((10, 10), "Hello World", fill='black')
    
    # Apply augmentations
    augmenter = HTRDataAugmentation(augment_prob=1.0)
    
    # Save original and augmented versions
    test_image.save('test_original.png')
    
    for i in range(5):
        augmented = augmenter(test_image.copy())
        augmented.save(f'test_augmented_{i}.png')
    
    print("Augmentation test completed. Check generated images.")


if __name__ == "__main__":
    test_augmentations()
