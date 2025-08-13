import cv2
import numpy as np
import os
from pathlib import Path
import argparse
import time

class CUDAFaceExtractor:
    def __init__(self, use_cuda=True):
        """
        Initialize CUDA-accelerated face extractor
        
        Args:
            use_cuda (bool): Whether to use CUDA acceleration
        """
        self.use_cuda = use_cuda and cv2.cuda.getCudaEnabledDeviceCount() > 0
        
        if self.use_cuda:
            print(f"CUDA enabled. Found {cv2.cuda.getCudaEnabledDeviceCount()} CUDA device(s)")
            # Set CUDA device (use device 0 by default)
            cv2.cuda.setDevice(0)
            
            # Initialize CUDA cascade classifier
            self.face_cascade = cv2.cuda_CascadeClassifier.create(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        else:
            print("CUDA not available or disabled. Using CPU processing.")
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
    
    def _detect_faces_cuda(self, gpu_gray, min_face_size=(50, 50)):
        """CUDA-accelerated face detection"""
        # Create GPU buffer for detection results
        gpu_faces = cv2.cuda_GpuMat()
        
        # Detect faces on GPU
        num_faces = self.face_cascade.detectMultiScale(
            gpu_gray,
            gpu_faces,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=min_face_size
        )
        
        if num_faces > 0:
            # Download results from GPU to CPU
            faces_array = gpu_faces.download()
            # Reshape to get individual face rectangles
            faces = faces_array.reshape(num_faces, 4)
        else:
            faces = np.array([])
        
        return faces
    
    def _detect_faces_cpu(self, gray, min_face_size=(50, 50)):
        """CPU-based face detection (fallback)"""
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=min_face_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces
    
    def _resize_cuda(self, gpu_image, target_size):
        """CUDA-accelerated image resizing"""
        gpu_resized = cv2.cuda.resize(gpu_image, target_size, interpolation=cv2.INTER_AREA)
        return gpu_resized
    
    def _convert_color_cuda(self, gpu_image, conversion_code):
        """CUDA-accelerated color conversion"""
        gpu_converted = cv2.cuda.cvtColor(gpu_image, conversion_code)
        return gpu_converted
    
    def extract_faces(self, image_path, output_dir="extracted_faces", 
                     target_size=(224, 224), padding=0.2, min_face_size=(50, 50)):
        """
        Extract faces using CUDA acceleration
        
        Args:
            image_path (str): Path to input image
            output_dir (str): Directory to save extracted faces
            target_size (tuple): Target size for extracted faces (width, height)
            padding (float): Padding around detected face (0.0 to 1.0)
            min_face_size (tuple): Minimum face size to detect
        
        Returns:
            list: List of extracted face arrays
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        start_time = time.time()
        
        if self.use_cuda:
            # Upload image to GPU
            gpu_image = cv2.cuda_GpuMat()
            gpu_image.upload(image)
            
            # Convert to grayscale on GPU
            gpu_gray = self._convert_color_cuda(gpu_image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces on GPU
            faces = self._detect_faces_cuda(gpu_gray, min_face_size)
            
            # Convert to RGB on GPU for processing
            gpu_rgb = self._convert_color_cuda(gpu_image, cv2.COLOR_BGR2RGB)
            rgb_image = gpu_rgb.download()
        else:
            # CPU fallback
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self._detect_faces_cpu(gray, min_face_size)
        
        detection_time = time.time() - start_time
        
        extracted_faces = []
        base_name = Path(image_path).stem
        
        print(f"Found {len(faces)} face(s) in {image_path} (Detection time: {detection_time:.3f}s)")
        
        for i, (x, y, w, h) in enumerate(faces):
            # Calculate padding
            pad_x = int(w * padding)
            pad_y = int(h * padding)
            
            # Calculate expanded coordinates
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(rgb_image.shape[1], x + w + pad_x)
            y2 = min(rgb_image.shape[0], y + h + pad_y)
            
            # Extract face region
            face_region = rgb_image[y1:y2, x1:x2]
            
            # Resize using CUDA if available
            if target_size:
                if self.use_cuda:
                    gpu_face = cv2.cuda_GpuMat()
                    gpu_face.upload(face_region)
                    gpu_resized = self._resize_cuda(gpu_face, target_size)
                    face_region = gpu_resized.download()
                else:
                    face_region = cv2.resize(face_region, target_size, interpolation=cv2.INTER_AREA)
            
            # Save extracted face
            output_path = os.path.join(output_dir, f"{base_name}_face_{i+1}.jpg")
            face_bgr = cv2.cvtColor(face_region, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, face_bgr)
            
            extracted_faces.append(face_region)
            print(f"Saved face {i+1} to {output_path}")
        
        return extracted_faces
    
    def extract_faces_batch_cuda(self, input_dir, output_dir="extracted_faces", 
                                batch_size=32, **kwargs):
        """
        CUDA-accelerated batch processing with memory management
        
        Args:
            input_dir (str): Directory containing input images
            output_dir (str): Directory to save extracted faces
            batch_size (int): Number of images to process in each batch
            **kwargs: Additional arguments for extract_faces method
        """
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise ValueError(f"Input directory {input_dir} does not exist")
        
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in supported_formats]
        
        if not image_files:
            print(f"No supported image files found in {input_dir}")
            return
        
        print(f"Processing {len(image_files)} images with batch size {batch_size}...")
        
        total_faces = 0
        total_time = 0
        
        # Process in batches to manage GPU memory
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]
            batch_start = time.time()
            
            for image_file in batch_files:
                try:
                    faces = self.extract_faces(str(image_file), output_dir, **kwargs)
                    total_faces += len(faces)
                except Exception as e:
                    print(f"Error processing {image_file}: {str(e)}")
            
            batch_time = time.time() - batch_start
            total_time += batch_time
            
            print(f"Batch {i//batch_size + 1}/{(len(image_files) + batch_size - 1)//batch_size} "
                  f"completed in {batch_time:.2f}s")
            
            # Clear GPU memory between batches
            if self.use_cuda:
                cv2.cuda.deviceReset()
        
        print(f"Total faces extracted: {total_faces}")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Average time per image: {total_time/len(image_files):.3f}s")
    
    def extract_faces_for_model_cuda(self, image_path, target_size=(224, 224), 
                                    normalize=True, padding=0.2, dtype=np.float32):
        """
        CUDA-accelerated face extraction optimized for model input
        
        Args:
            image_path (str): Path to input image
            target_size (tuple): Target size for model input
            normalize (bool): Whether to normalize pixel values to [0, 1]
            padding (float): Padding around detected face
            dtype: Output data type
        
        Returns:
            np.ndarray: Preprocessed face arrays ready for model input
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        start_time = time.time()
        
        if self.use_cuda:
            # GPU processing
            gpu_image = cv2.cuda_GpuMat()
            gpu_image.upload(image)
            
            gpu_gray = self._convert_color_cuda(gpu_image, cv2.COLOR_BGR2GRAY)
            faces = self._detect_faces_cuda(gpu_gray)
            
            gpu_rgb = self._convert_color_cuda(gpu_image, cv2.COLOR_BGR2RGB)
            rgb_image = gpu_rgb.download()
        else:
            # CPU fallback
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self._detect_faces_cpu(gray)
        
        processed_faces = []
        
        for (x, y, w, h) in faces:
            # Add padding
            pad_x = int(w * padding)
            pad_y = int(h * padding)
            
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(rgb_image.shape[1], x + w + pad_x)
            y2 = min(rgb_image.shape[0], y + h + pad_y)
            
            # Extract face region
            face_region = rgb_image[y1:y2, x1:x2]
            
            # Resize using CUDA if available
            if self.use_cuda:
                gpu_face = cv2.cuda_GpuMat()
                gpu_face.upload(face_region)
                gpu_resized = self._resize_cuda(gpu_face, target_size)
                face_resized = gpu_resized.download()
            else:
                face_resized = cv2.resize(face_region, target_size, interpolation=cv2.INTER_AREA)
            
            # Convert to specified dtype and normalize if requested
            face_resized = face_resized.astype(dtype)
            if normalize:
                face_resized = face_resized / 255.0
            
            processed_faces.append(face_resized)
        
        processing_time = time.time() - start_time
        print(f"Processed {len(processed_faces)} faces in {processing_time:.3f}s")
        
        return np.array(processed_faces) if processed_faces else np.array([])
    
    def benchmark_performance(self, image_path, iterations=10):
        """
        Benchmark CUDA vs CPU performance
        
        Args:
            image_path (str): Path to test image
            iterations (int): Number of iterations for benchmarking
        """
        print("Benchmarking performance...")
        
        # Test with CUDA
        cuda_times = []
        if self.use_cuda:
            for i in range(iterations):
                start = time.time()
                self.extract_faces_for_model_cuda(image_path)
                cuda_times.append(time.time() - start)
            
            avg_cuda_time = np.mean(cuda_times)
            print(f"CUDA average time: {avg_cuda_time:.3f}s")
        
        # Test with CPU (temporarily disable CUDA)
        original_cuda_setting = self.use_cuda
        self.use_cuda = False
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        cpu_times = []
        for i in range(iterations):
            start = time.time()
            self.extract_faces_for_model_cuda(image_path)  # Will use CPU
            cpu_times.append(time.time() - start)
        
        avg_cpu_time = np.mean(cpu_times)
        print(f"CPU average time: {avg_cpu_time:.3f}s")
        
        # Restore original settings
        self.use_cuda = original_cuda_setting
        if self.use_cuda:
            self.face_cascade = cv2.cuda_CascadeClassifier.create(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            speedup = avg_cpu_time / avg_cuda_time
            print(f"CUDA speedup: {speedup:.2f}x")
    
    def get_gpu_memory_info(self):
        """Get GPU memory information"""
        if self.use_cuda:
            free_mem, total_mem = cv2.cuda.deviceInfo(0)
            print(f"GPU Memory - Total: {total_mem/1024**2:.0f}MB, "
                  f"Free: {free_mem/1024**2:.0f}MB, "
                  f"Used: {(total_mem-free_mem)/1024**2:.0f}MB")
        else:
            print("CUDA not available")

def main():
    parser = argparse.ArgumentParser(description='CUDA-accelerated face extraction')
    parser.add_argument('--input', '-i', required=True,
                       help='Input image path or directory')
    parser.add_argument('--output', '-o', default='extracted_faces',
                       help='Output directory for extracted faces')
    parser.add_argument('--size', '-s', nargs=2, type=int, default=[224, 224],
                       help='Target size for extracted faces (width height)')
    parser.add_argument('--padding', '-p', type=float, default=0.2,
                       help='Padding around face (0.0 to 1.0)')
    parser.add_argument('--batch', '-b', action='store_true',
                       help='Process all images in input directory')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for GPU processing')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA acceleration')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    parser.add_argument('--gpu-info', action='store_true',
                       help='Show GPU memory information')
    
    args = parser.parse_args()
    
    # Initialize CUDA face extractor
    extractor = CUDAFaceExtractor(use_cuda=not args.no_cuda)
    
    if args.gpu_info:
        extractor.get_gpu_memory_info()
        return
    
    if args.benchmark:
        if os.path.isfile(args.input):
            extractor.benchmark_performance(args.input)
        else:
            print("Benchmark requires a single image file")
        return
    
    try:
        if args.batch:
            extractor.extract_faces_batch_cuda(
                args.input,
                args.output,
                batch_size=args.batch_size,
                target_size=tuple(args.size),
                padding=args.padding
            )
        else:
            if os.path.isfile(args.input):
                faces = extractor.extract_faces(
                    args.input,
                    args.output,
                    target_size=tuple(args.size),
                    padding=args.padding
                )
            else:
                print("Single file mode requires an image file")
    
    except Exception as e:
        print(f"Error: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Uncomment to run examples directly
    
    # Example 1: CUDA-accelerated face extraction
    # extractor = CUDAFaceExtractor(use_cuda=True)
    # faces = extractor.extract_faces_for_model_cuda("image.jpg", 
    #                                                target_size=(224, 224),
    #                                                normalize=True)
    # print(f"Extracted {len(faces)} faces with shape: {faces.shape if len(faces) > 0 else 'None'}")
    
    # Example 2: Batch processing with CUDA
    # extractor.extract_faces_batch_cuda("images_directory/", batch_size=16)
    
    # Example 3: Performance benchmark
    # extractor.benchmark_performance("test_image.jpg", iterations=20)
    
    # Run command line interface
    main()