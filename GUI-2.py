#!/usr/bin/python
import tkinter as tk
from tkinter import *
import SimpleITK as sitk
from PIL import Image, ImageTk
import numpy as np
from scipy.ndimage import binary_dilation

def resample_to_match(image, target_shape):
        original_spacing = np.array(image.GetSpacing())  
        original_size = np.array(image.GetSize())        

        target_size = target_shape[::-1]  
        new_spacing = original_spacing * (original_size / target_size)

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing.tolist())
        resampler.SetSize([int(sz) for sz in target_size])
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)  
        return resampler.Execute(image)

class MyApp:
    def __init__(self, parent):
        self.bg1 = '#717171'
        self.parent = parent
        self.parent.minsize(600, 450)
        self.is_loading_image = False

        # Bind the window resize event
        self.resize_event = self.parent.bind("<Configure>", self.on_resize)

        # Main container
        self.main_container = Frame(parent, background=self.bg1)
        self.main_container.grid(row=0, column=0, sticky="nsew")
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        # Top frame for image display
        self.top_frame = Frame(self.main_container, background=self.bg1)
        self.top_frame.grid(row=0, column=0, sticky="nsew")

        # Mid frame for controls
        self.mid_frame = Frame(self.main_container, background=self.bg1)
        self.mid_frame.grid(row=1, column=0, sticky="nsew")

        # Right frame for checkboxes
        self.right_frame = Frame(self.main_container, background=self.bg1)
        self.right_frame.grid(row=0, column=1, sticky="nsw")

        # Configure grid for canvas and scrollbars
        self.main_container.grid_rowconfigure(2, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)
        self.top_frame.grid_columnconfigure(0, weight=1)

        self.canvas = Canvas(self.top_frame)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.slice_scrollbar = Scale(self.top_frame, from_=0, to=0, orient="vertical", command=self.on_slice_scroll)
        self.slice_scrollbar.grid(row=0, column=1, sticky="ns")

        self.confidence_scrollbar = Scale(self.mid_frame, from_=0, to=255, orient="horizontal", label="Confidence Level", command=self.on_confidence_scroll)
        self.confidence_scrollbar.grid(row=0, column=0, sticky="ew")
        self.mid_frame.grid_columnconfigure(0, weight=1)

        self.load_button = Button(self.mid_frame, text="Show Wash", command=self.load_image, background=self.bg1, relief="groove")
        self.load_button.grid(row=1, column=0, sticky="ew", ipadx=20)

        self.image_array = None
        self.truth_array = None
        self.mask_arrays = {}  
        self.checked_masks = []  
        self.current_slice = 0

        self.masks = ['UNC', 'Physician A', 'B', 'C', 'D']
        self.checkbox_vars = {}
        for idx, mask in enumerate(self.masks):
            var = IntVar(value=0)
            self.checkbox_vars[mask] = var
            cb = Checkbutton(self.right_frame, text=mask, variable=var, command=self.on_checkbox_toggle, bg=self.bg1)
            cb.grid(row=idx, column=0, sticky="w")


    def load_image(self):
        """ Load a NIfTI image, ground truth mask, and five additional masks. """
        image_file_path = "Image.nii.gz"
        truth_file_path = "Truth.nii.gz"
        mask_paths = {
            "UNC": "0/Mask.nii.gz",
            "Physician A": "1/Mask.nii.gz",
            "B": "2/Mask.nii.gz",
            "C": "3/Mask.nii.gz",
            "D": "4/Mask.nii.gz",
        }

        try:
            img = sitk.ReadImage(image_file_path)
            self.image_array = sitk.GetArrayFromImage(img)
            #print(f"Image shape: {self.image_array.shape}")

            truth = sitk.ReadImage(truth_file_path)
            self.truth_array = sitk.GetArrayFromImage(truth)
            #print(f"Ground truth shape: {self.truth_array.shape}")

            self.mask_arrays = {}
            for key, path in mask_paths.items():
                mask = sitk.ReadImage(path)
                if mask.GetSize() != img.GetSize():  
                    #print(f"Resampling mask '{key}' from shape {mask.GetSize()} to {img.GetSize()}")
                    mask = resample_to_match(mask, self.image_array.shape)
                self.mask_arrays[key] = sitk.GetArrayFromImage(mask)
                #print(f"Mask '{key}' shape after resampling: {self.mask_arrays[key].shape}")

            
            shapes = [arr.shape for arr in self.mask_arrays.values()] + [self.image_array.shape, self.truth_array.shape]
            if not all(shape == shapes[0] for shape in shapes):
                raise ValueError("Shape mismatch between image and masks after resampling.")

            num_slices = self.image_array.shape[0]
            self.slice_scrollbar.config(from_=0, to=num_slices - 1)
            self.current_slice = 0
            self.display_slice(self.current_slice)

        except Exception as e:
            print(f"Error loading the data: {e}")


    def on_checkbox_toggle(self):
        self.checked_masks = [key for key, var in self.checkbox_vars.items() if var.get() == 1]
        self.confidence_scrollbar.config(to=255 * len(self.checked_masks) if self.checked_masks else 255)
        self.display_slice(self.current_slice) 


    def on_slice_scroll(self, value):
        if self.image_array is not None:
            self.current_slice = int(value)
            self.display_slice(self.current_slice)

    def on_confidence_scroll(self, value):
        self.display_slice(self.current_slice)

    def display_slice(self, slice_index):
        try:
            img_slice = self.image_array[slice_index, :, :]
            truth_slice = self.truth_array[slice_index, :, :]
            img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice)) * 255
            img_rgb = np.stack((img_slice, img_slice, img_slice), axis=-1).astype(np.uint8)

            mask_color = [0, 255, 0]  
            for mask_name in self.checked_masks:
                mask_slice = self.mask_arrays[mask_name][slice_index, :, :]
                img_rgb[mask_slice == 255] = mask_color

            truth_outline = binary_dilation(truth_slice > 0) & ~(truth_slice > 0)
            img_rgb[truth_outline] = [255, 0, 0] 

            window_width = min(self.parent.winfo_width() - 50, 512)  
            window_height = min(self.parent.winfo_height() - 100, 512)  
            pil_image = Image.fromarray(img_rgb).resize((window_width, window_height), Image.Resampling.LANCZOS)
            tk_image = ImageTk.PhotoImage(image=pil_image)

            self.canvas.config(width=window_width, height=window_height)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=tk_image)
            self.canvas.image = tk_image

        except Exception as e:
            print(f"Error displaying slice {slice_index}: {e}")

    def on_resize(self, event):
        if not self.is_loading_image and self.image_array is not None:
            self.display_slice(self.current_slice)


root = Tk()
root.configure(bg="#717171")
root.title("Confidence-Based Color Wash with Multiple Masks")
myapp = MyApp(root)
root.mainloop()
