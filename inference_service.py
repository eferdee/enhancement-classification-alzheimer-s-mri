# inference_service.py
"""
InferenceService module for MRI image enhancement and classification.

This module provides a stateful service object that manages model loading,
preprocessing, and inference operations for Alzheimer's MRI classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import numpy as np
import io


# Import NAFNet architecture
from nafnet_mri import create_medical_nafnet


# ====================================================================
# PREPROCESSING UTILITIES
# ====================================================================
class PadToSquare:
    """
    Utility class untuk padding citra menjadi square dengan zero padding.
    Digunakan untuk memastikan input model memiliki dimensi yang konsisten.
    """
    def __call__(self, img_tensor):
        """
        Args:
            img_tensor: Tensor [1, H, W]
        
        Returns:
            Padded tensor [1, H_new, W_new] where H_new == W_new
        """
        _, h, w = img_tensor.shape
        if h == w:
            return img_tensor
        
        diff = abs(h - w)
        pad1 = diff // 2
        pad2 = diff - pad1
        
        if h < w:
            padding = (0, 0, pad1, pad2)
        else:
            padding = (pad1, pad2, 0, 0)
        
        return F.pad(img_tensor, padding, value=0)


def crop_to_original_size(padded_tensor, orig_h, orig_w):
    """
    Crop padded tensor kembali ke ukuran asli citra.
    
    Args:
        padded_tensor: Tensor [1, H, W] hasil padding
        orig_h: Tinggi asli citra
        orig_w: Lebar asli citra
    
    Returns:
        Cropped tensor [1, orig_h, orig_w]
    """
    curr_h, curr_w = padded_tensor.shape[1], padded_tensor.shape[2]
    
    if curr_h == orig_h and curr_w == orig_w:
        return padded_tensor
    
    diff = abs(orig_h - orig_w)
    pad1 = diff // 2
    
    if orig_h < orig_w:
        h_start = pad1
        h_end = pad1 + orig_h
        w_start = 0
        w_end = orig_w
    else:
        h_start = 0
        h_end = orig_h
        w_start = pad1
        w_end = pad1 + orig_w
    
    cropped = padded_tensor[:, h_start:h_end, w_start:w_end]
    return cropped


# ====================================================================
# ADAPTIVE RESIDUAL WRAPPER
# ====================================================================
class AdaptiveResidualNAFNet(nn.Module):
    """
    Wrapper untuk NAFNet dengan adaptive residual scaling.
    
    Model ini membungkus base NAFNet dan menerapkan residual scaling
    yang adaptif untuk enhancement citra MRI dengan preservasi struktur.
    
    Attributes:
        base_model: FlatNAFNet instance
        current_scale: Float, skala residual saat ini
    """
    def __init__(self, base_model, init_scale=0.05, peak_scale=0.15, min_scale=0.05):
        super().__init__()
        self.base_model = base_model
        self.init_scale = init_scale
        self.peak_scale = peak_scale
        self.min_scale = min_scale
        self.current_scale = init_scale
    
    def forward(self, x):
        """
        Forward pass dengan adaptive residual scaling.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Enhanced tensor [B, C, H, W]
        """
        out = self.base_model(x)
        return x + self.current_scale * (out - x)


# ====================================================================
# INFERENCE SERVICE (MAIN CLASS)
# ====================================================================
class InferenceService:
    """
    Stateful service object untuk mengelola model inference.
    
    Class ini bertindak sebagai mediator antara Flask controller dan
    model layer, mengelola lifecycle model dan menyediakan interface
    untuk operasi enhancement dan classification.
    
    Attributes:
        enhancer: AdaptiveResidualNAFNet untuk enhancement
        classifier: ResNet18 untuk classification
        device: PyTorch device (CPU/CUDA)
        image_size_nafnet: Target resolution untuk NAFNet (208)
        image_size_classifier: Target resolution untuk classifier (224)
        labels: List label klasifikasi
    
    Design Pattern:
        Service Object Pattern dengan explicit dependency management
    """
    
    def __init__(
        self,
        model_enhance_path='nafnet_best.pth',
        model_classifier_path='best_baseline_classifier.pth',
        device='cpu'
    ):
        """
        Inisialisasi InferenceService.
        
        Args:
            model_enhance_path: Path ke NAFNet weights
            model_classifier_path: Path ke ResNet18 weights
            device: Device untuk inference ('cpu' atau 'cuda')
        """
        # Configuration
        self.device = torch.device(device)
        self.image_size_nafnet = 208
        self.image_size_classifier = 224
        self.num_classes = 4
        self.labels = [
            "MildDemented",
            "ModerateDemented",
            "NonDemented",
            "VeryMildDemented"
        ]
        
        # Model paths
        self.model_enhance_path = model_enhance_path
        self.model_classifier_path = model_classifier_path
        
        # Model instances (akan diinisialisasi di load_models)
        self.enhancer = None
        self.classifier = None
        
        # Preprocessing utilities
        self.pad_to_square = PadToSquare()
    
    # ================================================================
    # MODEL LOADING
    # ================================================================
    
    def load_models(self):
        """
        Load kedua model (enhancer dan classifier) dari disk.
        
        Method ini harus dipanggil sebelum menggunakan service.
        Melakukan validasi loading dan menampilkan status.
        
        Raises:
            RuntimeError: Jika salah satu model gagal dimuat
        """
        print(f"[InferenceService] Initializing models...")
        print(f"Device: {self.device}")
        
        self._load_enhancer()
        self._load_classifier()
        
        # Validation
        if self.enhancer is None or self.classifier is None:
            raise RuntimeError("Failed to load models. Check paths and weights.")
        
        print(f"\n{'='*70}")
        print(f"InferenceService initialized successfully!")
        print(f"Enhancer:   {'✓ Ready' if self.enhancer else '✗ Failed'}")
        print(f"Classifier: {'✓ Ready' if self.classifier else '✗ Failed'}")
        print(f"{'='*70}\n")
    
    def _load_enhancer(self):
        """Load NAFNet enhancement model (private method)."""
        print(f"\n[InferenceService] Loading NAFNet from {self.model_enhance_path}...")
        
        try:
            # Create base NAFNet
            base_model = create_medical_nafnet(
                img_channel=1,
                width=32,
                num_blocks=6,
                residual_scale=1.0,
                use_unet=False
            )
            
            # Wrap dengan AdaptiveResidualNAFNet
            self.enhancer = AdaptiveResidualNAFNet(
                base_model,
                init_scale=0.05,
                peak_scale=0.15,
                min_scale=0.05
            ).to(self.device)
            
            # Load checkpoint
            checkpoint = torch.load(
                self.model_enhance_path,
                map_location=self.device,
                weights_only=False
            )
            self.enhancer.load_state_dict(checkpoint['model_state_dict'])
            
            # Restore residual scale
            if 'residual_scale' in checkpoint:
                self.enhancer.current_scale = checkpoint['residual_scale']
                print(f"✓ NAFNet loaded (Epoch {checkpoint.get('epoch', '?')}, "
                      f"Residual Scale: {self.enhancer.current_scale:.4f})")
            else:
                print(f"✓ NAFNet loaded (Default Scale: {self.enhancer.current_scale:.4f})")
            
            self.enhancer.eval()
            
        except Exception as e:
            print(f"❌ ERROR loading NAFNet: {e}")
            self.enhancer = None
    
    def _load_classifier(self):
        """Load ResNet18 classification model (private method)."""
        print(f"\n[InferenceService] Loading Classifier from {self.model_classifier_path}...")
        
        try:
            self.classifier = models.resnet18(weights=None)
            
            # Modify conv1 untuk 1-channel input
            self.classifier.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            
            # FC layer
            self.classifier.fc = nn.Linear(
                self.classifier.fc.in_features,
                self.num_classes
            )
            
            # Load weights
            checkpoint = torch.load(
                self.model_classifier_path,
                map_location=self.device,
                weights_only=False
            )
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.classifier.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Classifier loaded (Epoch {checkpoint.get('epoch', '?')}, "
                      f"Val Acc: {checkpoint.get('val_acc', 0):.4f})")
            else:
                self.classifier.load_state_dict(checkpoint)
                print(f"✓ Classifier loaded")
            
            self.classifier.to(self.device)
            self.classifier.eval()
            
        except Exception as e:
            print(f"❌ ERROR loading Classifier: {e}")
            self.classifier = None
    
    # ================================================================
    # PREPROCESSING METHODS
    # ================================================================
    
    def _preprocess_for_nafnet(self, img_pil):
        """
        Preprocessing pipeline untuk NAFNet.
        
        Pipeline: Grayscale → ToTensor → PadToSquare
        
        Args:
            img_pil: PIL Image object
        
        Returns:
            tuple: (tensor, (orig_h, orig_w))
        """
        orig_w, orig_h = img_pil.size
        
        img = img_pil.convert('L')
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        img_tensor = self.pad_to_square(img_tensor)
        
        return img_tensor, (orig_h, orig_w)
    
    def _preprocess_for_classifier(self, img_pil):
        """
        Preprocessing pipeline untuk classifier.
        
        Pipeline: Grayscale → ToTensor → PadToSquare → Resize → ZScore
        
        Args:
            img_pil: PIL Image object
        
        Returns:
            Tensor [1, 224, 224]
        """
        img = img_pil.convert('L')
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        img_tensor = self.pad_to_square(img_tensor)
        
        # Resize to 224
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0),
            size=(self.image_size_classifier, self.image_size_classifier),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # Z-score normalization
        mean = img_tensor.mean()
        std = img_tensor.std()
        img_tensor = (img_tensor - mean) / (std + 1e-8)
        
        return img_tensor
    
    # ================================================================
    # PUBLIC INFERENCE METHODS
    # ================================================================
    
    def enhance_image(self, image_file):
        """
        Enhancement citra MRI menggunakan NAFNet.
        
        Output dikembalikan ke ukuran asli citra (cropping setelah enhancement).
        
        Args:
            image_file: File object dari Flask request
        
        Returns:
            bytes: JPEG image dalam ukuran asli, atau None jika gagal
        """
        if self.enhancer is None:
            print("ERROR: Enhancer model not loaded")
            return None
        
        try:
            img_pil = Image.open(image_file).convert('L')
            
            # Preprocessing
            img_tensor, (orig_h, orig_w) = self._preprocess_for_nafnet(img_pil)
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # Enhancement
            with torch.no_grad():
                enhanced_tensor = self.enhancer(img_tensor)
                enhanced_tensor = torch.clamp(enhanced_tensor, 0.0, 1.0)
            
            # Crop back ke ukuran asli
            enhanced_tensor = enhanced_tensor.squeeze(0)
            enhanced_cropped = crop_to_original_size(enhanced_tensor, orig_h, orig_w)
            
            # Convert to PIL
            enhanced_np = enhanced_cropped.squeeze().cpu().numpy()
            enhanced_img = Image.fromarray((enhanced_np * 255).astype(np.uint8)).convert('L')
            
            # Verify size
            assert enhanced_img.size == (orig_w, orig_h), \
                f"Size mismatch! Expected {(orig_w, orig_h)}, got {enhanced_img.size}"
            
            # Save to bytes
            buffer = io.BytesIO()
            enhanced_img.save(buffer, format="JPEG", quality=95)
            buffer.seek(0)
            
            return buffer.read()
            
        except Exception as e:
            print(f"ERROR in enhance_image: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def classify_image(self, image_file):
        """
        Klasifikasi citra MRI mentah (baseline tanpa enhancement).
        
        Args:
            image_file: File object dari Flask request
        
        Returns:
            dict: {prediction, confidence, probabilities} atau None jika gagal
        """
        if self.classifier is None:
            print("ERROR: Classifier model not loaded")
            return None
        
        try:
            img_pil = Image.open(image_file).convert('L')
            
            # Preprocessing
            img_tensor = self._preprocess_for_classifier(img_pil)
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # Classification
            with torch.no_grad():
                output = self.classifier(img_tensor)
            
            # Process results
            probabilities = F.softmax(output, dim=1).squeeze(0).cpu().numpy()
            predicted_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_idx]
            predicted_label = self.labels[predicted_idx]
            
            probabilities_dict = {
                label: float(prob) for label, prob in zip(self.labels, probabilities)
            }
            
            return {
                'prediction': predicted_label,
                'confidence': float(confidence * 100),
                'probabilities': probabilities_dict
            }
            
        except Exception as e:
            print(f"ERROR in classify_image: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def classify_enhanced_image(self, image_file):
        """
        Klasifikasi citra MRI setelah enhancement.
        
        Pipeline: Enhancement → Classification
        
        Args:
            image_file: File object dari Flask request
        
        Returns:
            dict: {prediction, confidence, probabilities} atau None jika gagal
        """
        if self.enhancer is None or self.classifier is None:
            print("ERROR: Models not loaded")
            return None
        
        try:
            img_pil = Image.open(image_file).convert('L')
            
            # Step 1: Enhancement
            img_tensor_nafnet, (orig_h, orig_w) = self._preprocess_for_nafnet(img_pil)
            img_tensor_nafnet = img_tensor_nafnet.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                enhanced_tensor = self.enhancer(img_tensor_nafnet)
                enhanced_tensor = torch.clamp(enhanced_tensor, 0.0, 1.0)
            
            # Crop back to original size
            enhanced_tensor = enhanced_tensor.squeeze(0)
            enhanced_cropped = crop_to_original_size(enhanced_tensor, orig_h, orig_w)
            
            # Convert to PIL
            enhanced_np = enhanced_cropped.squeeze().cpu().numpy()
            enhanced_pil = Image.fromarray((enhanced_np * 255).astype(np.uint8)).convert('L')
            
            # Step 2: Classification
            img_tensor_clf = self._preprocess_for_classifier(enhanced_pil)
            img_tensor_clf = img_tensor_clf.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.classifier(img_tensor_clf)
            
            # Process results
            probabilities = F.softmax(output, dim=1).squeeze(0).cpu().numpy()
            predicted_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_idx]
            predicted_label = self.labels[predicted_idx]
            
            probabilities_dict = {
                label: float(prob) for label, prob in zip(self.labels, probabilities)
            }
            
            return {
                'prediction': predicted_label,
                'confidence': float(confidence * 100),
                'probabilities': probabilities_dict
            }
            
        except Exception as e:
            print(f"ERROR in classify_enhanced_image: {e}")
            import traceback
            traceback.print_exc()
            return None