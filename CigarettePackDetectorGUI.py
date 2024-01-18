import glob
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from ultralytics import YOLO
import cv2

class YOLOGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("CIG Detection")

        self.model_path_label = tk.Label(master, text="YOLO Model Path:")
        self.model_path_label.pack()

        self.model_path_entry = tk.Entry(master)
        self.model_path_entry.pack()

        self.browse_model_button = tk.Button(master, text="Browse", command=self.browse_model)
        self.browse_model_button.pack()

        self.source_var = tk.IntVar()
        self.folder_radio = tk.Radiobutton(master, text="Folder", variable=self.source_var, value=0)
        self.folder_radio.pack()

        self.camera_radio = tk.Radiobutton(master, text="Camera", variable=self.source_var, value=1)
        self.camera_radio.pack()

        self.folder_path_label = tk.Label(master, text="Image Folder Path:")
        self.folder_path_label.pack()

        self.folder_path_entry = tk.Entry(master)
        self.folder_path_entry.pack()

        self.browse_folder_button = tk.Button(master, text="Browse", command=self.browse_folder)
        self.browse_folder_button.pack()

        self.clear_folder_button = tk.Button(master, text="Clear Folder", command=self.clear_folder_path)
        self.clear_folder_button.pack()

        self.confidence_label = tk.Label(master, text="Confidence Threshold:")
        self.confidence_label.pack()

        self.confidence_entry = tk.Entry(master)
        self.confidence_entry.pack()

        self.progress_bar = ttk.Progressbar(master, orient="horizontal", length=200, mode="determinate")
        self.progress_bar.pack()

        self.run_button = tk.Button(master, text="RUN", command=self.run_inference)
        self.run_button.pack()

    def browse_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("YOLO Model", "*.pt")])
        if model_path:
            self.model_path_entry.delete(0, tk.END)
            self.model_path_entry.insert(0, model_path)

    def browse_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.folder_path_entry.delete(0, tk.END)
            self.folder_path_entry.insert(0, folder_path)

    def clear_folder_path(self):
        self.folder_path_entry.delete(0, tk.END)

    def validate_confidence(self):
        try:
            confidence = float(self.confidence_entry.get())
            if 0.1 <= confidence <= 1.0:
                return True
            else:
                messagebox.showwarning("Invalid", "Confidence trebuie sa fie intre 0.1 si 1.")
                return False
        except ValueError:
            messagebox.showwarning("Invalid", "Confidence trebuie sa fie un numar.")
            return False

    def run_inference(self):
        model_path = self.model_path_entry.get()
        confidence_entry = self.confidence_entry.get()

        if not model_path or not confidence_entry:
            messagebox.showwarning("Informatie lipsa", "Verifica daca ai pus modelul si confidence.")
            return

        if not self.validate_confidence():
            return

        confidence_threshold = float(confidence_entry)

        model = YOLO(model_path)

        use_camera = self.source_var.get() == 1
        if use_camera:
            
            self.run_camera_inference(model, confidence_threshold)
        else:
            folder_path = self.folder_path_entry.get()
            if not folder_path:
                messagebox.showwarning("Informatie lipsa", "Lipseste path la imagini.")
                return

        
            self.run_folder_inference(model, folder_path, confidence_threshold)

        messagebox.showinfo("Succes", "Verificarea a fost facuta.")

    def run_folder_inference(self, model, folder_path, confidence_threshold):
        image_files = glob.glob(folder_path + "/*.jpg")  
        total_images = len(image_files)

        self.progress_bar["value"] = 0
        self.progress_bar["maximum"] = total_images

        for idx, image_path in enumerate(image_files, start=1):
            results = model(source=image_path, show=False, conf=confidence_threshold, save=True)

           
            self.progress_bar["value"] = idx
            self.master.update_idletasks()

    def run_camera_inference(self, model, confidence_threshold):
        cap = cv2.VideoCapture(0)  
        while True:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Camera Error", "Unable to capture frames from the camera.")
                break

            results = model(source=frame, show=True, conf=confidence_threshold)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOGUI(root)
    root.mainloop()