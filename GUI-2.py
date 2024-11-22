#!/usr/bin/python
import os
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
    def __init__(self, parent, base_path):
        self.bg1 = '#717171'
        self.base_path = base_path
        self.image_array = None
        self.truth_array = None
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

        self.mid_frame.grid_columnconfigure(0, weight=1)

        # self.load_button = Button(self.mid_frame, text="Show Wash", command=self.load_image, background=self.bg1,
        #                           relief="groove")
        # self.load_button.grid(row=1, column=0, sticky="ew", ipadx=20)
        self.mask_arrays = {}
        self.checked_masks = []
        self.current_slice = 0

        self.masks = [i for i in os.listdir(base_path) if i.endswith('.nii') and
                      i.find('Image') == -1]
        self.checkbox_vars = {}
        for idx, mask in enumerate(self.masks):
            mask = mask.split('CTV_Pelvis_')[1]
            var = IntVar(value=0)
            self.checkbox_vars[mask] = var
            cb = Checkbutton(self.right_frame, text=mask, variable=var, command=self.on_checkbox_toggle, bg=self.bg1)
            cb.grid(row=idx, column=0, sticky="w")
        self.load_image()

    def load_image(self):
        """ Load a NIfTI image, ground truth mask, and five additional masks. """
        image_file = "Image.nii"

        truth_file = [i for i in os.listdir(self.base_path) if i.endswith('.mhd')]

        try:
            img = sitk.ReadImage(os.path.join(self.base_path, image_file))
            self.image_array = sitk.GetArrayFromImage(img)
            self.truth_array = np.zeros(self.image_array.shape)
            # print(f"Image shape: {self.image_array.shape}")
            if truth_file:
                truth = sitk.ReadImage(os.path.join(self.base_path, truth_file[0]))
                self.truth_array = sitk.GetArrayFromImage(truth)
            # print(f"Ground truth shape: {self.truth_array.shape}")

            self.mask_arrays = {}
            for file_name in self.masks:
                key = file_name.split('CTV_Pelvis_')[1]
                mask = sitk.ReadImage(os.path.join(self.base_path, file_name))
                if mask.GetSize() != img.GetSize():
                    # print(f"Resampling mask '{key}' from shape {mask.GetSize()} to {img.GetSize()}")
                    mask = resample_to_match(mask, self.image_array.shape)
                mask_array = sitk.GetArrayFromImage(mask)
                self.mask_arrays[key] = mask_array
                # print(f"Mask '{key}' shape after resampling: {self.mask_arrays[key].shape}")

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
        self.display_slice(self.current_slice)

    def on_slice_scroll(self, value):
        if self.image_array is not None:
            self.current_slice = int(value)
            self.display_slice(self.current_slice)

    def on_confidence_scroll(self, value):
        self.display_slice(self.current_slice)

    def display_slice(self, slice_index):
        try:
            min_val, max_val = -200,  300
            dif = max_val - min_val if max_val != min_val else 1
            img_slice = self.image_array[slice_index, :, :]
            truth_slice = self.truth_array[slice_index, :, :]
            img_slice = (img_slice - min_val)/dif * 255
            img_slice = np.clip(img_slice, 0, 255)
            img_rgb = np.stack((img_slice, img_slice, img_slice), axis=-1).astype(np.uint8)

            green = [0, 255, 0]
            red = [255, 0, 0]
            blue = [0, 0, 255]
            total_mask = np.zeros(img_slice.shape)
            for mask_name in self.checked_masks:
                mask_slice = self.mask_arrays[mask_name][slice_index, :, :]
                total_mask += mask_slice
            pred_slice = (total_mask == len(self.checked_masks)).astype('int') if self.checked_masks else total_mask

            """
            Make a green outline where we have a prediction, but not the ground truth
            """
            green_outline = (pred_slice > 0) & ~(truth_slice > 0)
            green_outline = binary_dilation(green_outline) & ~green_outline
            """
            Make a blue outline where prediction AND ground truth agree
            """
            blue_outline = (pred_slice > 0) & (truth_slice == pred_slice) & (truth_slice > 0)
            blue_outline = binary_dilation(blue_outline) & ~blue_outline
            """
            Make a red outline where the ground truth exists but no prediction
            """
            red_outline = (truth_slice > 0) & ~(pred_slice > 0) if np.any(pred_slice) else truth_slice > 0
            red_outline = binary_dilation(red_outline) & ~red_outline

            img_rgb[green_outline] = green
            img_rgb[red_outline] = red
            # img_rgb[blue_outline] = blue


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


def run_model(path):
    root = Tk()
    root.configure(bg="#717171")
    root.title("Confidence-Based Color Wash with Multiple Masks")
    myapp = MyApp(root, path)
    root.mainloop()


if __name__ == '__main__':
    path = r'\\vscifs1\PhysicsQAdata\BMA\Predictions\ProstateNodes\Output\1.3.46.670589.33.1.63862355173814227200001.5286669292534571828'
    run_model(path)
