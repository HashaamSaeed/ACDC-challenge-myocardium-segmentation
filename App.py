###################################################################################################
'''
Code by : Adem Saglam and Syed Muhammad Hashaam Saeed


'''
###################################################################################################


from tkinter import *
import tkinter.filedialog
from tkinter.ttk import Style
import os
import numpy as np
import nibabel as nib
import cv2
import matplotlib
import matplotlib.pyplot as plt
import utility_functions as utils
from PIL import Image, ImageTk
import tensorflow as tf
from medpy.metric.binary import hd, dc

# class PatientScreening:

class App:

  def __init__(self, master):

    # All the files in selected path or selected by dialog
    self.selected_files = []
    self.processed_images = {}
    self.processed_masks = {}
    self.active_patient_id = ''
    self.slice_name = 'ni_z004'
    self.patient_info = None
    self.config_file = None
    self.ed_index = '01'
    self.es_index = '07'
    self.model_path = "./unet_model/"
    self.data_path = "./dataset/data"
    self.images_path = "./dataset/train"
    self.model = None
    self.predprob = 0.5
    self.loaded_images = { "ES": None, "ED": None }
    # Patient images in nibabel format full path
    self.patient_images = {}
    # Patient info in a dictionary
    self.image_order = 0
    # GUI related
    self.master = master
    self.create_ui()
    self.create_sidebar()
    self.load_model()

  def create_ui(self):
    self.master.geometry("1120x960")
    self.master.title("MRI Cardiac Segmentation")
    self.master.style = Style()
    self.master.style.theme_use("default")

    self.master.columnconfigure(0, weight=4)
    self.master.columnconfigure(1, weight=1)

    self.leftFrame = Frame(self.master, padx=10, pady=10)
    self.leftFrame.grid(row=0, column=0, sticky=N+S+E+W)

    self.rightFrame = Frame(self.master)
    self.rightFrame.columnconfigure(0, weight=1)
    self.rightFrame.grid(row=0, column=1, sticky=N+S+E+W)

    self.menuFrame = Frame(self.rightFrame, relief=RAISED, borderwidth=1, padx=10, pady=10)
    self.menuFrame.columnconfigure(0, weight=1)
    self.menuFrame.grid(row=0, column=0, sticky=N+S+E+W)

    self.metricsFrame = Frame(self.rightFrame, relief=RAISED, borderwidth=1)
    self.metricsFrame.grid(row=1, column=0, sticky=N+S+E+W)

    self.imageFrame = Frame(self.leftFrame, relief=RAISED, borderwidth=1, padx=5, pady=5)
    self.imageFrame.grid(row=0, column=0)

    self.imageControlsFrame = Frame(self.leftFrame, relief=RAISED, borderwidth=1, padx=5, pady=5)
    self.imageControlsFrame.grid(row=1, column=0)

    srcLbl = Label(self.imageFrame, text="ED Slice")
    srcLbl.grid(row=0, column=0)

    self.edimg = Label(self.imageFrame)
    self.edimg.grid(row=1, column=0, padx=5, pady=5)

    srcLbl = Label(self.imageFrame, text="ED Mask")
    srcLbl.grid(row=0, column=1)

    self.edmask = Label(self.imageFrame)
    self.edmask.grid(row=1, column=1, padx=5, pady=5)

    srcLbl = Label(self.imageFrame, text="ES Slice")
    srcLbl.grid(row=2, column=0)

    self.esimg = Label(self.imageFrame)
    self.esimg.grid(row=3, column=0, padx=5, pady=5)

    srcLbl = Label(self.imageFrame, text="ES Mask")
    srcLbl.grid(row=2, column=1)

    self.esmask = Label(self.imageFrame)
    self.esmask.grid(row=3, column=1, padx=5, pady=5)

    self.esimg = Label(self.imageFrame)
    self.esimg.grid(row=3, column=0, padx=5, pady=5)

    srcLbl = Label(self.imageFrame, text="ED Predicted Mask")
    srcLbl.grid(row=0, column=2)

    self.edpmask = Label(self.imageFrame)
    self.edpmask.grid(row=1, column=2, padx=5, pady=5)

    srcLbl = Label(self.imageFrame, text="ES Predicted Mask")
    srcLbl.grid(row=2, column=2)

    self.espmask = Label(self.imageFrame)
    self.espmask.grid(row=3, column=2, padx=5, pady=5)

    b = Button(self.imageControlsFrame, text=">", command=self.next_src_image)
    b.grid(row=1, column=1, sticky=E)
    b = Button(self.imageControlsFrame, text="<", command=self.prev_src_image)
    b.grid(row=1, column=0, sticky=W)

  def create_sidebar(self):
    fselect_button = Button(self.menuFrame, text="Load Data", command=self.open_path_dialog)
    fselect_button.grid(row=0, sticky=W+E)

    self.exit_button = Button(
      master=self.menuFrame,
      text="Exit",
      command=self.master.quit
    )

    self.exit_button.grid(row=1, sticky=W+E)

    self.patientNameLbl = Label(self.metricsFrame, text="patient_043", anchor="center")
    self.patientNameLbl.grid(row=5, column=0, padx=10, pady=10)

    tbl_fspace = Label(self.metricsFrame, text="")
    tbl_fspace.grid(row=0, column=0, padx=5, pady=5)

    lv_ed_lbl = Label(self.metricsFrame, text="LV")
    lv_ed_lbl.grid(row=1, column=0, padx=5, pady=5)

    dice_lbl = Label(self.metricsFrame, text="Dice coeff.")
    dice_lbl.grid(row=2, column=0, padx=5, pady=5)

    volume_lbl = Label(self.metricsFrame, text="Volume")
    volume_lbl.grid(row=3, column=0, padx=5, pady=5)

    volume_lbl = Label(self.metricsFrame, text="EF")
    volume_lbl.grid(row=4, column=0, padx=5, pady=5)

    ed_tbl_lbl = Label(self.metricsFrame, text="ED")
    ed_tbl_lbl.grid(row=0, column=1, padx=5, pady=5)
    es_tbl_lbl = Label(self.metricsFrame, text="ES")
    es_tbl_lbl.grid(row=0, column=2, padx=5, pady=5)

    self.lv_ed_val = Label(self.metricsFrame, text="")
    self.lv_ed_val.grid(row=1, column=1, padx=5, pady=5)

    self.lvl_es_val = Label(self.metricsFrame, text="")
    self.lvl_es_val.grid(row=1, column=2, padx=5, pady=5)

    self.dice_ed = Label(self.metricsFrame, text="")
    self.dice_ed.grid(row=2, column=1, padx=5, pady=5)
    self.dice_es = Label(self.metricsFrame, text="")
    self.dice_es.grid(row=2, column=2, padx=5, pady=5)

    self.vol_ed = Label(self.metricsFrame, text="")
    self.vol_ed.grid(row=3, column=1, padx=5, pady=5)
    self.vol_es = Label(self.metricsFrame, text="")
    self.vol_es.grid(row=3, column=2, padx=5, pady=5)

    self.ef_lbl = Label(self.metricsFrame, text="")
    self.ef_lbl.grid(row=4, column=1, padx=5, pady=5)

  def open_path_dialog(self):
    # path = tkinter.filedialog.askdirectory()
    path = self.images_path

    if not path:
      raise Exception("No Directory selected")

    self.path = path

    sub_path = os.path.join(path, 'images')

    for r, d, f in os.walk(sub_path):
      for file in f:
        full_path = os.path.join(sub_path, file)
        fname_split = os.path.splitext(file)
        second_ext = os.path.splitext(fname_split[0])
        if fname_split[1] == '.gz':
          self.selected_files.append(full_path)
        if fname_split[1] == '.cfg':
          self.config_file = full_path
        if fname_split[1] == '.png':
          img_full = { second_ext[1]: full_path }
          try:
            self.processed_images[second_ext[0]].update({second_ext[1][1:]: full_path})
          except KeyError:
            self.processed_images[second_ext[0]] = {second_ext[1][1:]: full_path}


    self.processed_images = self.merge_patient_data(self.processed_images)

    sub_path = os.path.join(path, 'masks')

    for r, d, f in os.walk(sub_path):
      for file in f:
        full_path = os.path.join(sub_path, file)
        fname_split = os.path.splitext(file)
        second_ext = os.path.splitext(fname_split[0])
        idx = second_ext[0].index('_frame')
        if fname_split[1] == '.gz':
          self.selected_files.append(full_path)
        if fname_split[1] == '.cfg':
          self.config_file = full_path
        if fname_split[1] == '.png':
          key = second_ext[0].replace('_gt', '')
          try:
            self.processed_masks[key].update({second_ext[1][1:]: full_path})
          except KeyError:
            self.processed_masks[key] = {second_ext[1][1:]: full_path}

    self.processed_masks = self.merge_patient_data(self.processed_masks)

    # for r, d, f in os.walk(path):
    #   for file in f:
    #     full_path = os.path.join(path, file)
    #     fname_split = os.path.splitext(file)
    #     if fname_split[1] == '.gz':
    #       self.selected_files.append(full_path)
    #     if fname_split[1] == '.cfg':
    #       self.config_file = full_path
    #     if fname_split[1] == '.png':
    #       self.processed_images.append(full_path)

    # if self.config_file == None or len(self.selected_files) < 1:
    #   raise Exception("Directory not valid")

    # self.process_files()
    self.set_active_patient()
    self.load_images()
    # self.load_model()
    self.predict_masks()
    self.calculate_metrics()
    self.place_ui_images()

  def metrics(self, img_gt, img_pred, voxel_size):
    """
    Function to compute the metrics between two segmentation maps given as input.

    Parameters
    ----------
    img_gt: np.array
    Array of the ground truth segmentation map.

    img_pred: np.array
    Array of the predicted segmentation map.

    voxel_size: list, tuple or np.array
    The size of a voxel of the images used to compute the volumes.

    Return
    ------
    A list of metrics in this order, [Dice LV, Volume LV, Err LV(ml)]
    """

    if img_gt.ndim  != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))

    res = []
    # Loop on each classes of the input images
    for c in [1]:
        # Copy the gt image to not alterate the input
        gt_c_i = np.copy(img_gt)
        #gt_c_i[gt_c_i != c] = 0

        # Copy the pred image to not alterate the input
        pred_c_i = np.copy(img_pred)
        #pred_c_i[pred_c_i != c] = 0

        # Clip the value to compute the volumes
        gt_c_i = np.clip(gt_c_i, 0, 1)
        pred_c_i = np.clip(pred_c_i, 0, 1)

        # Compute the Dice
        dice = dc(gt_c_i, pred_c_i)

        # Compute volume
        volpred = pred_c_i.sum() * np.prod(voxel_size) / 1000.
        volgt = gt_c_i.sum() * np.prod(voxel_size) / 1000.

        res += [dice, volpred, volgt]

    return res

  def load_datafile(self, fname):
    return nib.load(fname)

  def calculate_metrics(self):
    f_name = f"{self.active_patient_id}_frame{self.es_index}.nii.gz"
    self.nib_es_data = self.load_datafile(os.path.join(self.data_path, "train_set", f_name))

    f_name = f"{self.active_patient_id}_frame{self.ed_index}.nii.gz"
    self.nib_ed_data = self.load_datafile(os.path.join(self.data_path, "train_set", f_name))

    es_pixdim = self.nib_es_data.header['pixdim']
    ed_pixdim = self.nib_ed_data.header['pixdim']
    es_voxel1=es_pixdim[1]
    es_voxel2=es_pixdim[2]
    es_slicenum=int(es_pixdim[3])

    es_mask_filename = self.get_mask_filename(self.es_index)
    es_mask_frame = cv2.imread(es_mask_filename, 0)
    ed_mask_filename = self.get_mask_filename(self.ed_index)
    ed_mask_frame = cv2.imread(ed_mask_filename, 0)

    es_prediction = np.squeeze(self.es_prediction)
    ed_prediction = np.squeeze(self.ed_prediction)

    es_res = self.metrics(es_mask_frame, es_prediction, [es_pixdim[1], es_pixdim[2], int(es_pixdim[3])])
    ed_res = self.metrics(ed_mask_frame, ed_prediction, [ed_pixdim[1], ed_pixdim[2], int(ed_pixdim[3])])
    self.dice_es.config(text=f"{round(es_res[0]*100,2)}%")

    self.dice_ed.config(text=f"{round(ed_res[0]*100,2)}%")


  def load_images(self):
    patient_data = self.processed_images[self.active_patient_id]
    mask_data = self.processed_masks[self.active_patient_id]
    self.loaded_images["ED"] = cv2.imread(patient_data[self.ed_index][self.slice_name], 0)
    self.loaded_images["ES"] = cv2.imread(patient_data[self.es_index][self.slice_name], 0)

  def set_active_patient(self):
    keys = self.processed_images.keys()
    keys = list(keys)
    self.active_patient_id = keys[self.image_order]
    self.active_patient_id = 'patient043'

  def load_model(self):
    self.model = tf.keras.models.load_model(self.model_path)

  def predict_image(self, c_img):
    c_img  = np.expand_dims(c_img, axis=-1)
    c_img  = np.expand_dims(c_img, axis=0)
    prediction = self.model.predict(c_img, verbose=1)
    predimg = np.zeros((1,192,192,1),dtype=np.uint8)
    predimg = (prediction > self.predprob).astype(np.uint8)

    return predimg

  def predict_masks(self):
    self.ed_prediction = self.predict_image(self.loaded_images["ED"])
    predimg = np.squeeze(self.ed_prediction)*255

    im = Image.fromarray(predimg)
    im.save("ed_prediction.jpg")

    ed_image_comp = ImageTk.PhotoImage(im)
    self.set_image(self.edpmask, ed_image_comp)

    self.es_prediction = self.predict_image(self.loaded_images["ES"])
    predimg = np.squeeze(self.es_prediction)*255

    im = Image.fromarray(predimg)
    im.save("es_prediction.jpg")

    es_image_comp = ImageTk.PhotoImage(im)
    self.set_image(self.espmask, es_image_comp)

  # obsolete
  def open_file_dialog(self):
    filenames = tkinter.filedialog.askopenfiles()

    if filenames:
      self.selected_files = filenames
      self.read_files()
      self.process_files()
      self.create_ui_images()

  #arranges files in self.selected_files depending on their file extensions and creates a dictionary
  def arrange_files(self):
    for filename in self.selected_files:
      frame_idx = filename.find('_frame')
      qd_idx = filename.find('_4d')
      extension_idx = filename.find('.nii')

      if frame_idx is not -1:
        self.patient_images[filename[frame_idx+6:extension_idx]] = filename

      if qd_idx is not -1:
        self.patient_images[filename[qd_idx+1:extension_idx]] = filename

  def get_frame_file(self, frame_no):
    return self.patient_images.get(frame_no if len(frame_no) > 1 else "0" + frame_no)

  def process_files(self):
    self.patient_info = utils.UtilityFunctions.read_patient_cfg(self.config_file)
    self.arrange_files()

    ED_f = self.get_frame_file(self.patient_info['ED'])
    ES_f = self.get_frame_file(self.patient_info['ES'])
    D4_f = self.get_frame_file('4d')

    ED = nib.load(ED_f)
    ES = nib.load(ES_f)

    if (D4_f):
      D4 = nib.load(D4_f)

    ED_images = self.save_image(ED, 'ED', 0, ED.shape[2])
    EF_images = self.save_image(ED, 'ES', 0, ES.shape[2])

    self.processed_files =  { "ED": ED_images, "EF": EF_images }

  def save_image(self, nibFile, prefix, slice, multi=None):
    if multi is not None:
      output_files = []
      for s in range(slice, multi):
        output_files.append(self.save_image(nibFile, prefix, s)[0])

      return output_files

    fig, (ax1) = plt.subplots(1,1)
    ax1.imshow(nibFile.get_fdata()[:,:,slice])
    ax1.set_title('Image')

    output_file = ".temp/%s-slice_%s.png"%(prefix, slice)

    plt.savefig(output_file, transparent=True, pad_inches=0, bbox_inches='tight')
    plt.close()

    return [output_file]

  def read_files(self):
    for c_file in self.selected_files:
      nibFile = nib.load(c_file.name)
      print(nibFile.header.get_data_shape())

      fname_lbl = Label(self.frame, text=c_file.name)
      fname_lbl.pack(fill=X)

      utils.read_patient_cfg

  def prev_src_image(self):
    if (self.image_order < 1):
      return

    self.image_order -= 1
    load = Image.open(self.processed_files["ED"][self.image_order])
    resized = self.resize_image(load)
    render = ImageTk.PhotoImage(resized)
    self.set_image(self.imageFrame, render)

  def next_src_image(self):
    if (len(self.processed_files["ED"]) <= self.image_order+1):
      return

    self.image_order += 1
    load = Image.open(self.processed_files["ED"][self.image_order])
    resized = self.resize_image(load)
    render = ImageTk.PhotoImage(resized)
    self.set_image(self.imageFrame, render)

  # image = ImageTk component dstFrame = TkFrame
  def set_image(self, dstFrame, image):
    dstFrame.configure(image=image)
    dstFrame.image = image

  def create_image_component(self, data, index):
    load = Image.open(data[index][self.slice_name])
    return ImageTk.PhotoImage(load)

  def get_image_filename(self, index):
    return self.processed_images[self.active_patient_id][index][self.slice_name]

  def get_mask_filename(self, index):
    return self.processed_masks[self.active_patient_id][index][self.slice_name]

  def place_ui_images(self):
    patient_data = self.processed_images[self.active_patient_id]
    mask_data = self.processed_masks[self.active_patient_id]

    img = self.create_image_component(patient_data, self.ed_index)
    self.set_image(self.edimg, img)

    img = self.create_image_component(patient_data, self.es_index)
    self.set_image(self.esimg, img)

    img = self.create_image_component(mask_data, self.ed_index)
    self.set_image(self.edmask, img)

    img = self.create_image_component(mask_data, self.es_index)
    self.set_image(self.esmask, img)

  def _create_ui_images(self):
    load = Image.open(self.processed_files["ED"][self.image_order])
    resized = self.resize_image(load)
    render = ImageTk.PhotoImage(resized)
    self.set_image(self.imageFrame, render)

  def resize_image(self, pil_image):
    basewidth = 360
    wpercent = (basewidth/float(pil_image.size[0]))
    hsize = int((float(pil_image.size[1])*float(wpercent)))
    pil_image = pil_image.resize((basewidth,hsize), Image.ANTIALIAS)
    return pil_image

  def merge_patient_data(self, data):
    new_data = {}
    for old_key in data:
      patient_id = old_key[0:old_key.index('_frame')]
      frame_id = old_key[old_key.index('_frame')+6:len(old_key)]
      try:
        new_data[patient_id].update({ frame_id: data[old_key] })
      except KeyError:
        new_data[patient_id] = { frame_id: data[old_key] }

    return new_data


if __name__ == '__main__':
  matplotlib.use("Agg")
  root = Tk()
  app = App(root)

  root.mainloop()