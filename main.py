import matplotlib.pyplot as plt
import numpy as np
import pydicom
import math
import cv2
import hashlib
import struct
import reedsolo
#import os
#import pywt
#import reedsolo
#from PIL import Image

# FUNZIONI
def set_padding(rows, columns, image, dwt_levels = 2, block_size = 2):
    # block_size = 2 --> (blocchi 8x8)
    unit = pow(2, dwt_levels) * block_size

    # calcolo padding
    pad_rows = math.ceil(rows / unit) * unit - rows
    pad_columns = math.ceil(columns / unit) * unit - columns

    # aggiungo padding calcolato in modo simmetrico
    padded_image = np.pad(
        image,
        ((pad_rows // 2, pad_rows - (pad_rows // 2)), (pad_columns // 2, pad_columns - (pad_columns // 2))),
        mode='symmetric'
    )
    return padded_image

def unset_padding(rows, columns, padded_image, dwt_levels = 2, block_size = 2):
    unit = pow(2, dwt_levels) * block_size

    # calcolo padding
    pad_rows = math.ceil(rows / unit) * unit - rows
    pad_columns = math.ceil(columns / unit) * unit - columns

    top = pad_rows // 2
    bottom = top + rows
    left = pad_columns // 2
    right = left + columns

    # taglio immagine
    original_image_image = padded_image[top:bottom, left:right]

    return original_image

    # controllo che sia multiplo??

def generate_roni_mask(image):
    # normalizzazione a 8-bit e B/N
    # norm = np.zeros((800, 800))
    norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # riduzione rumore pre sogliatura (con filtro mediano)
    blurred = cv2.medianBlur(norm, 5)

    # thresholding
    thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)[1]

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        raise ValueError("Nessun contorno trovato.")

    # Cerco contorno con area maggiore
    main_cont = max(contours, key=cv2.contourArea)

    roi_mask = np.zeros_like(norm)
    cv2.drawContours(roi_mask, [main_cont], -1, 255, thickness=cv2.FILLED)

    roni_mask = cv2.bitwise_not(roi_mask)

    return roni_mask

def show_roni_mask(image, mask):
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    # 1. Immagine padding
    axes[0].imshow(image, cmap='bone')
    axes[0].set_title('Immagine con Padding')
    axes[0].axis('off')

    # 2. Maschera RONI
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Maschera RONI (bianco)')
    axes[1].axis('off')

    # 3. Marschera trasparente
    axes[2].imshow(image, cmap='bone')
    overlay = np.zeros((*mask.shape, 3), dtype=np.uint8)
    overlay[mask == 255] = [255, 0, 0]  # RONI = rosso
    overlay[mask == 0] = [0, 255, 0]    # ROI = verde
    axes[2].imshow(overlay, alpha=0.3)  # Semi-trasparente
    axes[2].set_title('Overlay: ROI (verde) + RONI (rosso)')
    axes[2].axis('off')

    # 4. Contorno ROI evidenziato

    # Disegna il contorno della ROI sull'immagine originale
    # Converti in 3 canali per colorare il contorno
    img_rgb = cv2.cvtColor(
        cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),
        cv2.COLOR_GRAY2RGB
    )

    # Trova contorni della ROI
    roi_mask = cv2.bitwise_not(mask)
    contours_roi, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Disegna contorno in rosso
    cv2.drawContours(img_rgb, contours_roi, -1, (255, 0, 0), 2)
    plt.imshow(img_rgb)

    axes[3].set_title('Contorno ROI')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()
    
def show_mask_perc(image, mask):
    total_px = image.size

    roni_px = np.sum(mask == 255)
    roi_px = np.sum(mask == 0)

    roni_perc = roni_px * 100 / total_px
    roi_perc = roi_px * 100 / total_px

    print("RONI: ", roni_px,
        "\nROI: ", roi_px,
        "\nPercentuale RONI: ", roni_perc,
        "\nPercentuale ROI: ", roi_perc
    )

# Non l'ID vero ma uno pseudonimo, sia x privacy che x risparmiare spazio
def create_patient_id_payload(patient_id_string, bits=64):      #bits = lunghezza payload
    # hash id
    hash_bytes = hashlib.sha256(patient_id_string.encode('utf-8')).digest()
    # prende i primi 64 bit 
    num_bytes = bits // 8                       # (trasforma in byte)
    return hash_bytes[:num_bytes]      # estrae solo i primi bytes

def add_ecc_to_payload(payload_bytes, ecc_syms=10):
    # Aggiungere Reed-Solomon ECC al payload (10 byte di parità)
    rs = reedsolo.RSCodec(ecc_syms)
    return bytes(rs.encode(payload_bytes))
    # ritorna lista di bytes con ECC

def bytes_to_bits(data_bytes):
    bits = []
    for byte in data_bytes:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & i)
    return bits

# ---------------------------------

# --------- PREPROCESSING --------- 
# Caricamento immagine
file_path = "images/005.DCM"
dataset = pydicom.dcmread(file_path)
print(f"ID Paziente: {dataset.PatientID}")
print(f"Modalita': {dataset.Modality}")      # US = UltraSound
# Creazione array
pixel_array = dataset.pixel_array
original_image = pixel_array[0]

# Visualizzazione immagine (frame 1/11 in questo caso)
""" plt.imshow(original_image, cmap=plt.cm.bone)
plt.title(f"DICOM: {dataset.Modality}")
plt.axis("off")  # Nasconde gli assi coordinati
plt.show() """

# Padding
padded_image = set_padding(dataset.Rows, dataset.Columns, original_image)

""" print(f"Dimensioni originali: {original_image.shape}")
print(f"Dimensioni con padding: {padded_image.shape}")

plt.imshow(padded_image, cmap=plt.cm.bone)
plt.axis("off")
plt.show() 

unpadded_image = unset_padding(dataset.Rows, dataset.Columns, padded_image)
print(f"Dimensioni dopo unset_padding: {unpadded_image.shape}") """

# divisione tra RONI e ROI
mask = generate_roni_mask(padded_image)

# show_roni_mask(padded_image, mask)
# show_mask_perc(padded_image, mask)

payload = create_patient_id_payload("1234-6789-5647")
print(payload)
print(add_ecc_to_payload(payload))

""" Da definire se intensità media RONI < intensità media ROI """

# --------- Livello 1: DWT-DCT-SVD nella RONI --------- 
# --------- Livello 2: IWT-DE con location map --------- 
# --------- Estrazione e block-chain --------- 
# --------- Validazione --------- 
