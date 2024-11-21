#!/usr/bin/python
import tkinter as tk
from tkinter import *
import SimpleITK as sitk
from PIL import Image, ImageTk
import numpy as np
import matplotlib

class MyApp:
    def __init__(self, parent):
        self.bg1 = '#717171'
        self.parent = parent
        self.parent.minsize(600, 450)

        # Main container
        self.main_container = Frame(parent, background=self.bg1)
        self.main_container.grid(row=0, column=0, sticky="nsew")
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        # Top frame for image display
        self.top_frame = Frame(self.main_container, background=self.bg1)
        self.top_frame.grid(row=0, column=0, sticky="nsew")

        # Right panel for checkboxes
        self.right_panel = Frame(self.main_container, background=self.bg1)
        self.right_panel.grid(row=0, column=1, sticky="nsw")

        # Mid frame for controls (scrollbars)
        self.mid_frame = Frame(self.main_container, background=self.bg1)
        self.mid_frame.grid(row=1, column=0, sticky="nsew")

        # Configure grid for canvas and scrollbars
        self.main_container.grid_rowconfigure(2, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)
        self.top_frame.grid_columnconfigure(0, weight=1)

        # Canvas for image
        self.canvas = Canvas(self.top_frame)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.mask_labels = ['UNC', 'Physician A', 'B', 'C', 'D']
        self.check_vars = {}  

        for label in self.mask_labels:
            var = IntVar()
            checkbox = Checkbutton(self.right_panel, text=label, variable=var, background=self.bg1, command=self.update_masks)
            checkbox.pack(anchor="w", pady=2)
            self.check_vars[label] = var  

        self.slice_scrollbar = Scale(self.top_frame, from_=0, to=0, orient="vertical", command=self.on_slice_scroll)
        self.slice_scrollbar.grid(row=0, column=2, sticky="ns")

        self.confidence_scrollbar = Scale(self.mid_frame, from_=0, to=255, orient="horizontal", label="Confidence Level Threshold", command=self.on_confidence_scroll)
        self.confidence_scrollbar.grid(row=2, column=0, sticky="ew")
        self.mid_frame.grid_columnconfigure(0, weight=1)

        self.masks = {}  
        self.shown_prediction = None  
        self.current_slice = 0  
        self.load_sample_masks()

    def load_sample_masks(self):
        """Load binary masks from specified file paths."""
        base_path = "path to masks"
        mask_filenames = [f"{base_path}{i}/Mask.nii.gz" for i in range(5)]
        
        for label, file_path in zip(self.mask_labels, mask_filenames):
            try:
                mask = sitk.ReadImage(file_path)
                mask_array = sitk.GetArrayFromImage(mask)
                
                self.masks[label] = mask_array.astype(np.uint8)
                print(f"Loaded mask for {label} from {file_path}")
                self.slice_scrollbar.config(to=mask_array.shape[0] - 1)

            except Exception as e:
                print(f"Error loading mask for {label} from {file_path}: {e}")
                self.masks[label] = np.zeros((1, 320, 320), dtype=np.uint8)

    def update_masks(self):
        selected_masks = [self.masks[label][self.current_slice] for label, var in self.check_vars.items() if var.get() == 1]
        num_checked = len(selected_masks)
        
        if num_checked > 0:
            scaled_masks = [mask * (255 / num_checked) for mask in selected_masks]
            combined_mask = np.sum(scaled_masks, axis=0)
            self.shown_prediction = np.clip(combined_mask, 0, 255).astype(np.uint8)
        else:
            self.shown_prediction = np.zeros((320, 320), dtype=np.uint8)
        
        self.update_display()
    def update_display(self):
        try:
            if self.shown_prediction is None:
                return  

            display_array = self.shown_prediction.astype(np.uint8)
            display_array_rgb = np.stack((display_array,) * 3, axis=-1)  
            display_width, display_height = 750, 600  

            pil_image = Image.fromarray(display_array_rgb).resize((display_width, display_height), Image.Resampling.LANCZOS)
            tk_image = ImageTk.PhotoImage(image=pil_image)

            self.canvas.delete("all")
            self.canvas.config(width=display_width, height=display_height)
            self.canvas.create_image(0, 0, anchor="nw", image=tk_image)
            self.canvas.image = tk_image
           
        except Exception as e:
            print(f"Error updating display: {e}")

    def on_slice_scroll(self, value):
        """Handle slice scrolling to update the displayed slice."""
        self.current_slice = int(value)
        self.update_masks()  

    def on_confidence_scroll(self, value):
        self.confidence_level_threshold = int(value)
        self.update_display()

root = Tk()
root.configure(bg="#717171")
root.title("Show Multiple Mask Overlays")
myapp = MyApp(root)
root.mainloop()
