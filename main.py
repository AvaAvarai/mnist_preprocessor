import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random

class MNISTPreprocessor(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("MNIST Preprocessor")
        self.geometry("1200x800")
        
        # Load MNIST dataset
        self.load_mnist_data()
        
        # Initialize variables
        self.kernel_size = tk.IntVar(value=3)
        self.stride = tk.IntVar(value=1)
        self.kernel_elements = [[tk.DoubleVar(value=1.0 if i == j else 0.0) 
                                for j in range(5)] for i in range(5)]
        
        # Create the GUI
        self.create_widgets()
        
        # Generate initial samples
        self.refresh_samples()
    
    def load_mnist_data(self):
        """Load MNIST dataset"""
        try:
            transform = transforms.Compose([transforms.ToTensor()])
            self.mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
            self.mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
            
            # Group data by class
            self.class_data = {i: [] for i in range(10)}
            for img, label in self.mnist_train:
                # Handle label being either a tensor or already an int
                label_idx = label.item() if hasattr(label, 'item') else label
                self.class_data[label_idx].append(img)
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to load MNIST data: {str(e)}")
            raise
    
    def create_widgets(self):
        """Create all the widgets for the GUI"""
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Split the window horizontally between controls and display
        main_paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)
        
        # Controls section
        controls_frame = ttk.LabelFrame(main_paned, text="Controls", width=250)
        controls_frame.pack(fill=tk.Y, padx=5, pady=5)
        controls_frame.pack_propagate(False)  # Prevent frame from shrinking
        
        # Display section
        display_frame = ttk.Frame(main_paned)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add frames to the paned window with appropriate weights
        main_paned.add(controls_frame, weight=1)
        main_paned.add(display_frame, weight=4)
        
        # Controls section content
        # Refresh button
        ttk.Button(controls_frame, text="Refresh Samples", command=self.refresh_samples).pack(
            fill=tk.X, padx=5, pady=5)
        
        # Kernel size
        kernel_frame = ttk.LabelFrame(controls_frame, text="Kernel Settings")
        kernel_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(kernel_frame, text="Kernel Size:").pack(anchor=tk.W, padx=5, pady=2)
        kernel_size_spin = ttk.Spinbox(kernel_frame, from_=1, to=99, textvariable=self.kernel_size, width=10)
        kernel_size_spin.pack(fill=tk.X, padx=5, pady=2)
        kernel_size_spin.bind("<Return>", self.update_kernel_ui)
        kernel_size_spin.bind("<FocusOut>", self.update_kernel_ui)
        # Add trace to update immediately when value changes
        self.kernel_size.trace_add("write", lambda *args: self.update_kernel_ui())
        
        # Stride
        ttk.Label(kernel_frame, text="Stride:").pack(anchor=tk.W, padx=5, pady=2)
        stride_spin = ttk.Spinbox(kernel_frame, from_=1, to=99, textvariable=self.stride, width=10)
        stride_spin.pack(fill=tk.X, padx=5, pady=2)
        # Add trace to update immediately when value changes
        self.stride.trace_add("write", lambda *args: self.validate_stride())
        
        # Kernel elements
        self.kernel_elements_frame = ttk.LabelFrame(kernel_frame, text="Kernel Elements")
        self.kernel_elements_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.create_kernel_ui()
        
        # Apply button
        ttk.Button(controls_frame, text="Apply Fresh Preprocessing", command=self.update_processed_images).pack(
            fill=tk.X, padx=5, pady=5)
        
        # Create vertical paned window for original and processed images
        display_paned = ttk.PanedWindow(display_frame, orient=tk.VERTICAL)
        display_paned.pack(fill=tk.BOTH, expand=True)
        
        # Display section for original and processed images
        self.original_frame = ttk.LabelFrame(display_paned, text="Original MNIST Samples")
        self.processed_frame = ttk.LabelFrame(display_paned, text="Processed MNIST Samples") 
        
        # Add the frames to the paned window with equal weight
        display_paned.add(self.original_frame, weight=1)
        display_paned.add(self.processed_frame, weight=1)
        
        # Prepare matplotlib figures for display
        self.original_fig = plt.Figure(figsize=(12, 6), dpi=100)
        self.original_canvas = FigureCanvasTkAgg(self.original_fig, master=self.original_frame)
        self.original_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        self.processed_fig = plt.Figure(figsize=(12, 6), dpi=100)
        self.processed_canvas = FigureCanvasTkAgg(self.processed_fig, master=self.processed_frame)
        self.processed_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
    
    def create_kernel_ui(self):
        """Create the kernel matrix UI elements"""
        # Clear existing elements
        for widget in self.kernel_elements_frame.winfo_children():
            widget.destroy()
        
        size = self.kernel_size.get()
        
        # Create scrollable frame for large kernels
        canvas = tk.Canvas(self.kernel_elements_frame, width=200)
        scrollbar = ttk.Scrollbar(self.kernel_elements_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create matrix of entry widgets
        for i in range(size):
            for j in range(size):
                # Create additional kernel element variables if needed
                if i >= len(self.kernel_elements) or j >= len(self.kernel_elements[0]):
                    while i >= len(self.kernel_elements):
                        self.kernel_elements.append([tk.DoubleVar(value=0.0) for _ in range(max(5, size))])
                    while j >= len(self.kernel_elements[0]):
                        for row in self.kernel_elements:
                            row.append(tk.DoubleVar(value=0.0))
                
                entry = ttk.Entry(scrollable_frame, width=3, 
                                 textvariable=self.kernel_elements[i][j])
                entry.grid(row=i, column=j, padx=1, pady=1)
        
        # Add preset buttons in a more compact layout
        presets_frame = ttk.Frame(scrollable_frame)
        presets_frame.grid(row=size, column=0, columnspan=size, pady=5)
        
        # Create a grid of preset buttons for better space utilization
        ttk.Button(presets_frame, text="Identity", 
                  command=lambda: self.set_kernel_preset("identity")).grid(row=0, column=0, padx=1, pady=1)
        ttk.Button(presets_frame, text="Edge", 
                  command=lambda: self.set_kernel_preset("edge")).grid(row=0, column=1, padx=1, pady=1)
        ttk.Button(presets_frame, text="Blur", 
                  command=lambda: self.set_kernel_preset("blur")).grid(row=1, column=0, padx=1, pady=1)
        ttk.Button(presets_frame, text="Sharpen", 
                  command=lambda: self.set_kernel_preset("sharpen")).grid(row=1, column=1, padx=1, pady=1)
        
        # Pack the canvas and scrollbar
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        if size > 7:  # Only show scrollbar for larger kernels
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def update_kernel_ui(self, event=None):
        """Update kernel UI when size changes"""
        # Validate kernel size is a positive integer
        try:
            size = self.kernel_size.get()
            if size < 1:
                self.kernel_size.set(1)
            elif size > 11:  # Set a reasonable upper limit for UI purposes
                self.kernel_size.set(11)
        except:
            self.kernel_size.set(3)  # Default if invalid
        
        self.create_kernel_ui()
    
    def validate_stride(self):
        """Ensure stride is a positive integer"""
        try:
            stride = self.stride.get()
            if stride < 1:
                self.stride.set(1)
            elif stride > 10:  # Set a reasonable upper limit
                self.stride.set(10)
        except:
            self.stride.set(1)  # Default if invalid
    
    def set_kernel_preset(self, preset):
        """Set the kernel to a predefined preset"""
        size = self.kernel_size.get()
        
        # Reset all values to zero
        for i in range(size):
            for j in range(size):
                self.kernel_elements[i][j].set(0.0)
        
        if preset == "identity":
            # Identity matrix - only center element is 1
            center = size // 2
            self.kernel_elements[center][center].set(1.0)
            
        elif preset == "edge":
            # Edge detection for arbitrary sizes
            if size % 2 == 1:  # Only works with odd-sized kernels
                center = size // 2
                # Set all elements to -1
                for i in range(size):
                    for j in range(size):
                        self.kernel_elements[i][j].set(-1.0)
                # Set center to positive value: 8 for 3x3, scaled for other sizes
                center_value = size * size - 1
                self.kernel_elements[center][center].set(float(center_value))
            else:
                # For even sizes, use a simple approximation
                messagebox.showwarning("Warning", "Edge detection works best with odd-sized kernels. Using approximation.")
                for i in range(size):
                    for j in range(size):
                        self.kernel_elements[i][j].set(-1.0)
                self.kernel_elements[size//2][size//2].set(float(size*size))
            
        elif preset == "blur":
            # Box blur
            value = 1.0 / (size * size)
            for i in range(size):
                for j in range(size):
                    self.kernel_elements[i][j].set(value)
                    
        elif preset == "sharpen":
            # Sharpen for arbitrary sizes
            if size % 2 == 1:  # Only works with odd-sized kernels
                center = size // 2
                # Set center cross to -1
                for i in range(size):
                    for j in range(size):
                        if i == center or j == center:
                            self.kernel_elements[i][j].set(-1.0)
                # Set center element to positive value
                self.kernel_elements[center][center].set(float(size * 2 - 1))
            else:
                messagebox.showwarning("Warning", "Sharpen filter works best with odd-sized kernels. Using approximation.")
                self.kernel_elements[size//2][size//2].set(float(size))
                self.kernel_elements[size//2-1][size//2-1].set(-1.0)
                self.kernel_elements[size//2-1][size//2].set(-1.0)
                self.kernel_elements[size//2][size//2-1].set(-1.0)
                self.kernel_elements[size//2][size//2+1].set(-1.0)
                self.kernel_elements[size//2+1][size//2].set(-1.0)
    
    def refresh_samples(self):
        """Refresh the sample images"""
        try:
            # Select 10 random samples for each class
            self.current_samples = {}
            for cls in range(10):
                # Check if we have enough samples for this class
                if len(self.class_data[cls]) >= 10:
                    # Select random indices
                    indices = random.sample(range(len(self.class_data[cls])), 10)
                    self.current_samples[cls] = [self.class_data[cls][i] for i in indices]
                else:
                    # If not enough samples, use all available with possible repeats
                    samples = self.class_data[cls]
                    # Ensure we have 10 samples even if we need to repeat some
                    repeated_samples = samples * (10 // len(samples) + 1)
                    self.current_samples[cls] = repeated_samples[:10]
            
            # Display original samples
            self.display_image_atlas(self.current_samples, self.original_fig, self.original_canvas)
            
            # Update processed images based on current kernel
            self.update_processed_images()
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to refresh samples: {str(e)}")
    
    def display_image_atlas(self, samples_dict, figure, canvas):
        """Display an atlas of images from each class"""
        figure.clear()
        
        # Create a 10x10 grid (10 classes, 10 samples each)
        axes = figure.subplots(10, 10)
        figure.subplots_adjust(wspace=0.1, hspace=0.2)  # Reduce whitespace
        
        for row, (cls, samples) in enumerate(sorted(samples_dict.items())):
            for col, img in enumerate(samples):
                # Convert tensor to numpy array and remove channel dimension
                if isinstance(img, torch.Tensor):
                    img_np = img.squeeze().numpy()
                else:
                    img_np = img
                
                # Display the image with removed frames and border
                ax = axes[row, col]
                ax.imshow(img_np, cmap='gray', interpolation='nearest')
                
                # Remove axis ticks and frames
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                
                # Add class label for first column
                if col == 0:
                    ax.set_ylabel(f"Class {cls}", rotation=90, size='small')
        
        figure.tight_layout()
        canvas.draw()
    
    def get_kernel(self):
        """Get the current kernel as a numpy array"""
        size = self.kernel_size.get()
        kernel = np.zeros((size, size))
        
        for i in range(size):
            for j in range(size):
                kernel[i, j] = self.kernel_elements[i][j].get()
        
        return kernel
    
    def apply_convolution(self, image, kernel, stride):
        """Apply convolution to an image using the given kernel and stride"""
        try:
            if isinstance(image, torch.Tensor):
                # Convert to numpy if it's a tensor
                image = image.squeeze().numpy()
            
            # Get dimensions
            height, width = image.shape
            k_height, k_width = kernel.shape
            
            # Calculate padding needed to maintain original size
            pad_h = ((height - 1) * stride + k_height - height) // 2
            pad_w = ((width - 1) * stride + k_width - width) // 2
            
            # Apply padding to image
            padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
            
            # Calculate output dimensions - should match original image size
            out_height = height
            out_width = width
            
            # Initialize output image
            output = np.zeros((out_height, out_width))
            
            # Vectorized convolution using numpy for better performance
            for i in range(0, out_height):
                y_pos = i * stride
                for j in range(0, out_width):
                    x_pos = j * stride
                    # Extract window
                    window = padded_image[y_pos:y_pos+k_height, x_pos:x_pos+k_width]
                    # Apply kernel and sum
                    if window.shape == kernel.shape:  # Ensure window has correct size
                        output[i, j] = np.sum(window * kernel)
            
            # Normalize to [0, 1] range
            min_val = output.min()
            max_val = output.max()
            if max_val > min_val:
                output = (output - min_val) / (max_val - min_val)
            
            return output
        except Exception as e:
            print(f"Error in convolution: {str(e)}")
            # Return original image in case of error
            return image
    
    def update_processed_images(self):
        """Update the processed images display"""
        if not hasattr(self, 'current_samples'):
            return
        
        # Get current kernel and stride
        kernel = self.get_kernel()
        stride = self.stride.get()
        
        # Apply convolution to all samples
        processed_samples = {}
        for cls, samples in self.current_samples.items():
            processed_samples[cls] = []
            for img in samples:
                processed_img = self.apply_convolution(img, kernel, stride)
                processed_samples[cls].append(processed_img)
        
        # Display processed samples
        self.display_image_atlas(processed_samples, self.processed_fig, self.processed_canvas)

if __name__ == "__main__":
    app = MNISTPreprocessor()
    app.mainloop()
