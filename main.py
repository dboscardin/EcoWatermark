import matplotlib.pyplot as plt
import numpy as np
import pydicom
import math
import cmath
import cv2
import hashlib
import struct
import reedsolo
import pywt
import time
import os, sys
from scipy.fft import dctn, idctn
from PIL import Image

# --------- SALVATAGGIO OUTPUT INTERMEDI ---------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pulisce il log ad ogni nuova esecuzione
log_path = os.path.join(OUTPUT_DIR, "log.txt")
open(log_path, "w", encoding="utf-8").close()

def save_intermediate(name, data, log_text=None, vmin=None, vmax=None):
    """
    Salva un output intermedio nella cartella outputs/.
    - name       : nome file senza estensione (es. "01_original_image")
    - data       : np.ndarray (immagine o array generico) oppure None
    - log_text   : stringa opzionale da aggiungere al log testuale
    - vmin, vmax : se forniti, usano scala fissa per il PNG (evita artefatti
                   di luminosità da normalizzazione indipendente). Se None,
                   usa il min/max dell'array corrente.
    """
    if data is not None:
        npy_path = os.path.join(OUTPUT_DIR, f"{name}.npy")
        np.save(npy_path, data)

        # Se è un'immagine 2D o 3D, salva anche come PNG leggibile
        if data.ndim in (2, 3):
            lo = float(vmin) if vmin is not None else float(data.min())
            hi = float(vmax) if vmax is not None else float(data.max())
            scale = 255.0 / (hi - lo) if hi > lo else 1.0
            img_norm = np.clip((data.astype(np.float32) - lo) * scale, 0, 255).astype(np.uint8)
            png_path = os.path.join(OUTPUT_DIR, f"{name}.png")
            cv2.imwrite(png_path, img_norm)
            print(f"[SAVE] {png_path}  |  {npy_path}  (scale [{lo:.1f}, {hi:.1f}])")
        else:
            print(f"[SAVE] {npy_path}")

    if log_text is not None:
        log_path = os.path.join(OUTPUT_DIR, "log.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"=== {name} ===\n{log_text}\n\n")
        print(f"[LOG]  {log_path}  ← '{name}'")

# FUNZIONI
def set_padding(rows, columns, image, dwt_levels = 1, block_size = 8):
    # block_size = 8 --> (blocchi 8x8)
    unit = pow(2, dwt_levels) * block_size      # 2 * 8 = 16

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

def unset_padding(rows, columns, padded_image, dwt_levels = 1, block_size = 8):
    unit = pow(2, dwt_levels) * block_size

    # calcolo padding
    pad_rows = math.ceil(rows / unit) * unit - rows
    pad_columns = math.ceil(columns / unit) * unit - columns

    top = pad_rows // 2
    bottom = top + rows
    left = pad_columns // 2
    right = left + columns

    return padded_image[top:bottom, left:right]

    # controllo che sia multiplo di unit o mi fido?

def generate_roni_mask(image):
    # normalizzazione a 8-bit e B/N
    # norm = np.zeros((800, 800))
    norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # riduzione rumore pre sogliatura (con filtro mediano)
    blurred = cv2.medianBlur(norm, 5)

    # thresholding (tutto sopra a 10 diventa bianco)
    thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)[1]

    # trovo contorni
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        raise ValueError("Nessun contorno trovato.")

    # Cerco contorno con area maggiore
    main_cont = max(contours, key=cv2.contourArea)

    # creo maschera roi
    roi_mask = np.zeros_like(norm)
    cv2.drawContours(roi_mask, [main_cont], -1, 255, thickness=cv2.FILLED)

    # inverto roi
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
    # hash id (SHA-256)
    hash_bytes = hashlib.sha256(patient_id_string.encode('utf-8')).digest()
    # prende i primi 64 bit 
    num_bytes = bits // 8                       # (trasforma in byte)
    return hash_bytes[:num_bytes]      # estrae solo i primi bytes

def add_ecc_to_payload(payload_bytes, ecc_syms=10):
    # Aggiungere Reed-Solomon ECC al payload (10 byte di parità x correzione errore) 
    rs = reedsolo.RSCodec(ecc_syms)
    return bytes(rs.encode(payload_bytes))
    # ritorna lista di bytes con ECC
    # ecc corregge fino a ecc_syms/2 byte corrotti

def bytes_to_bits(data_bytes):
    bits = []
    for byte in data_bytes:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits

def get_block(band, row, col, B=8):
    return band[row * B: (row + 1) * B, col * B: (col + 1) * B]

def set_block(band, row, col, block, B=8):
    band[row * B: (row + 1) * B, col * B: (col + 1) * B] = block

# Trasforma valori spaziali in coefficienti frequenziali
# k è il margine minimo tra i due coefficienti. + alto, + robusto ma + visibile
# USo un approccio a confronto di coefficienti, non SVD
def embed_bit(B, bit, k=20):
    #[4, 1] e [3, 2]: scelta di due coefficienti a media frequenza
    x, y = B[4, 1], B[3, 2]
    m =(x + y) / 2
    d = x - y
    want = 1 if bit else -1     # bit 1 -> B[4, 1] > B[3, 2], bit 0 viceversa
    if want * d < k:        # impone differenza di segno voluto e modulo >= k
        d = want * k
    B[4, 1], B[3, 2] = m + d / 2, m - d / 2

# Simmetrico
def extract_bit(B):
    return int(B[4, 1] > B[3, 2])

def iwt2(image_int):
    image = image_int.astype(np.int64)

    # Separare pixel pari e dispari

    # lift righe
    cols_pari = image[:, 0::2]     # a
    cols_dispari = image[:, 1::2]  # b
    
    high_rows = cols_dispari - cols_pari        # differenza
    low_rows = cols_pari + (high_rows >> 1)      # media

    # lift colonne di low_rows
    rows_pari_low = low_rows[0::2, :]
    rows_dispari_low = low_rows[1::2, :]
    HL = rows_dispari_low - rows_pari_low
    LL = rows_pari_low + (HL >> 1)

    # lift colonne di high_rows
    rows_pari_high = high_rows[0::2, :]
    rows_dispari_high = high_rows[1::2, :]
    HH = rows_dispari_high - rows_pari_high
    LH = rows_pari_high + (HH >> 1)

    return LL, LH, HL, HH

def haar_int_inverse(LL, LH, HL, HH):
    # Restituisce immagine ricostruita int64 — inverso esatto di iwt2
    LL = LL.astype(np.int64)
    LH = LH.astype(np.int64)
    HL = HL.astype(np.int64)
    HH = HH.astype(np.int64)

    # Inverso passo colonne di low_rows (da LL, HL)
    rows_pari_low    = LL - (HL >> 1)
    rows_dispari_low = HL + rows_pari_low
    low_rows = np.empty((rows_pari_low.shape[0] * 2, rows_pari_low.shape[1]), dtype=np.int64)
    low_rows[0::2, :] = rows_pari_low
    low_rows[1::2, :] = rows_dispari_low

    # Inverso passo colonne di high_rows (da LH, HH)
    rows_pari_high    = LH - (HH >> 1)
    rows_dispari_high = HH + rows_pari_high
    high_rows = np.empty((rows_pari_high.shape[0] * 2, rows_pari_high.shape[1]), dtype=np.int64)
    high_rows[0::2, :] = rows_pari_high
    high_rows[1::2, :] = rows_dispari_high

    # Inverso passo righe (da low_rows, high_rows)
    cols_pari    = low_rows - (high_rows >> 1)
    cols_dispari = high_rows + cols_pari
    image = np.empty((cols_pari.shape[0], cols_pari.shape[1] * 2), dtype=np.int64)
    image[:, 0::2] = cols_pari
    image[:, 1::2] = cols_dispari

    return image

def de_embed(band, bits, bit_depth=8):
    # DE — range check su coefficienti signed [-max_val, max_val]
    band_wm = band.copy()
    loc_map = []
    bit_idx = 0
    rows, cols = band.shape
    max_val = (1 << bit_depth) - 1  # 255 per 8 bit

    for r in range(rows):
        for c in range(0, cols - 1, 2):
            if bit_idx >= len(bits):
                loc_map.append(False)
                continue
            x = int(band_wm[r, c])
            y = int(band_wm[r, c + 1])
            l = (x + y) // 2
            h = x - y
            h_new = 2 * h + bits[bit_idx]
            x_new = l + (h_new + 1) // 2
            y_new = l - h_new // 2
            if -max_val <= x_new <= max_val and -max_val <= y_new <= max_val:
                band_wm[r, c]     = x_new
                band_wm[r, c + 1] = y_new
                loc_map.append(True)
                bit_idx += 1
            else:
                loc_map.append(False)

    return band_wm, loc_map, bit_idx

def de_extract(band_wm, loc_map):
    # Estrae i bit e ricostruisce la band originale dalla loc_map.
    band_orig = band_wm.copy()
    extracted = []
    rows, cols = band_wm.shape
    pair_idx = 0

    for r in range(rows):
        for c in range(0, cols - 1, 2):
            if pair_idx >= len(loc_map):
                break
            if loc_map[pair_idx]:
                x = int(band_wm[r, c])
                y = int(band_wm[r, c + 1])
                h_new = x - y
                bit = h_new % 2
                h   = (h_new - bit) // 2
                l   = x - (h_new + 1) // 2
                x_orig = l + (h + 1) // 2
                y_orig = l - h // 2
                band_orig[r, c]     = x_orig
                band_orig[r, c + 1] = y_orig
                extracted.append(bit)
            pair_idx += 1

    return band_orig, extracted

def compress_locmap(loc_map):
    """Converte la loc_map in bytes compressi (zlib). Ritorna (bytes_compressi, n_pairs)."""
    import zlib
    bits = np.array(loc_map, dtype=np.uint8)
    packed = np.packbits(bits).tobytes()
    return zlib.compress(packed, level=9), len(loc_map)

def decompress_locmap(data_bytes, n_pairs):
    """Ripristina la loc_map da bytes compressi."""
    import zlib
    packed = zlib.decompress(data_bytes)
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))
    return [bool(b) for b in bits[:n_pairs]]

# --------- FUNZIONI DI ATTACCO ---------
import random
import io

def jpeg2000_attack(image_np, compression_ratio=10):
    """
    Simula compressione JPEG2000 (supporta uint16) tramite OpenCV.
    compression_ratio : fattore di compressione (10 = 10:1, più alto = più perdita)
    Ritorna array dello stesso dtype dell'input.
    """
    encode_param = [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000,
                    int(1000 / compression_ratio)]
    _, encoded = cv2.imencode('.jp2', image_np, encode_param)
    attacked = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    return attacked.astype(image_np.dtype)

def crop_attack(image_np, crop_frac=0.1):
    """
    Ritaglia l'immagine rimuovendo crop_frac della dimensione su ogni lato,
    poi la riporta alle dimensioni originali paddando SOLO in basso e a destra.
    Così il contenuto rimasto è allineato in alto a sinistra come nell'originale:
    i blocchi embeddati mantengono la stessa posizione e l'estrazione funziona
    su tutto il contenuto non croppato.
    image_np  : numpy array
    crop_frac : frazione di pixel rimossi per lato (default 10%)
    """
    h, w = image_np.shape
    dy = max(1, int(h * crop_frac))
    dx = max(1, int(w * crop_frac))
    # Croppa solo dal basso e da destra: l'angolo in alto a sinistra resta intatto
    cropped = image_np[:h - dy, :w - dx]
    # Ripadda in basso e a destra per riportare alle dimensioni originali
    attacked = np.pad(cropped,
                      ((0, dy), (0, dx)),
                      mode='constant', constant_values=0)
    assert attacked.shape == image_np.shape
    return attacked

def noise_attack(image_np, std=5.0):
    """
    Aggiunge rumore gaussiano additivo.
    image_np : numpy array (qualsiasi dtype)
    std      : deviazione standard del rumore (in livelli di intensità)
    Ritorna array dello stesso dtype e shape, con clip al range originale.
    """
    noise = np.random.normal(0, std, image_np.shape)
    attacked = np.clip(image_np.astype(np.float64) + noise,
                       0, np.iinfo(image_np.dtype).max)
    return attacked.astype(image_np.dtype)

def extract_l1_ber(attacked_np, roni_blocks, payload_bits,
                   rows, cols, dwt_levels=1, block_size=8):
    """
    Estrae il watermark L1 dall'immagine attaccata e calcola il BER
    rispetto ai bit originali del payload.
    Ritorna (ber, n_bit_estratti, bit_estratti).
    """
    attacked_f = attacked_np.astype(np.float64)
    _, (LH_att, _, _) = pywt.dwt2(attacked_f, 'haar')
    extracted = [extract_bit(dctn(get_block(LH_att, r, c), norm='ortho'))
                 for r, c in roni_blocks[:len(payload_bits)]]
    ber = float(np.mean(np.array(extracted) != np.array(payload_bits)))
    return ber, len(extracted), extracted


def bits_to_bytes(bits):
    out = []
    for i in range(0, len(bits) - 7, 8):
        byte = 0
        for b in bits[i:i+8]:
            byte = (byte << 1) | b
        out.append(byte)
    return bytes(out)

def de_extract_blind(band_wm, n_bits, bit_depth=8):
    # Ricalcola location map ogni volta (blind, senza loc_map esterna)
    # Range check su coefficienti signed [-max_val, max_val]
    band_orig = band_wm.copy().astype(np.int64)
    extracted = []
    rows, cols = band_wm.shape
    max_val = (1 << bit_depth) - 1

    for r in range(rows):
        for c in range(0, cols - 1, 2):
            if len(extracted) >= n_bits:
                break
            x = int(band_wm[r, c])
            y = int(band_wm[r, c + 1])
            h_new = x - y
            bit    = h_new & 1              # ultimo bit, corretto anche per negativi
            h      = h_new >> 1             # shift aritmetico, inverso esatto di 2*h+bit
            l      = (x + y) // 2          # stessa formula del forward
            x_orig = l + (h + 1) // 2
            y_orig = l - h // 2
            if -max_val <= x_orig <= max_val and -max_val <= y_orig <= max_val:
                band_orig[r, c]     = x_orig
                band_orig[r, c + 1] = y_orig
                extracted.append(bit)

    return band_orig, extracted

# ---------------------------------

# --------- PREPROCESSING --------- 
# Caricamento immagine
file_path = "images/001.dcm"
dataset = pydicom.dcmread(file_path)
print(f"ID Paziente: {dataset.PatientID}")
print(f"Modalita': {dataset.Modality}")      # US = UltraSound
# Creazione array
pixel_array = dataset.pixel_array
original_image = pixel_array[0]

# Scala fissa per tutti i PNG di immagini pixel (evita artefatti di luminosità
# da normalizzazione indipendente nei confronti visivi)
IMG_VMIN = float(original_image.min())
IMG_VMAX = float(original_image.max())

save_intermediate("01_original_image", original_image,
    log_text=f"Shape: {original_image.shape}, dtype: {original_image.dtype}\n"
             f"Min: {original_image.min()}, Max: {original_image.max()}",
    vmin=IMG_VMIN, vmax=IMG_VMAX)

# Visualizzazione immagine (frame 1/11 in questo caso)
""" plt.imshow(original_image, cmap=plt.cm.bone)
plt.title(f"DICOM: {dataset.Modality}")
plt.axis("off")  # Nasconde gli assi coordinati
plt.show() """

# Padding
padded_image = set_padding(dataset.Rows, dataset.Columns, original_image)

save_intermediate("02_padded_image", padded_image,
    log_text=f"Shape originale: {original_image.shape} → Shape con padding: {padded_image.shape}",
    vmin=IMG_VMIN, vmax=IMG_VMAX)
# L'immagine dev'essere multiplo di 16 (2^1 * 8) perchè:
# La DWT dimezza ledimensioni (dev'essere divisibile per due)
# La DWT lavora su blocchi 8x8

""" print(f"Dimensioni originali: {original_image.shape}")
print(f"Dimensioni con padding: {padded_image.shape}")

plt.imshow(padded_image, cmap=plt.cm.bone)
plt.axis("off")
plt.show() 

unpadded_image = unset_padding(dataset.Rows, dataset.Columns, padded_image)
print(f"Dimensioni dopo unset_padding: {unpadded_image.shape}") """

# divisione tra RONI e ROI
mask = generate_roni_mask(padded_image)

save_intermediate("03_roni_mask", mask,
    log_text=f"Pixel RONI (255): {np.sum(mask==255)}  "
             f"({np.sum(mask==255)*100/mask.size:.1f}%)\n"
             f"Pixel ROI  (0)  : {np.sum(mask==0)}  "
             f"({np.sum(mask==0)*100/mask.size:.1f}%)")

# show_roni_mask(padded_image, mask)
# show_mask_perc(padded_image, mask)

payload = create_patient_id_payload("1234-6789-5647")
#print(payload)
print("Payload + ECC: ", add_ecc_to_payload(payload))

""" Da definire se intensità media RONI < intensità media ROI?? """

# --------- Livello 1: DWT-DCT-SVD nella RONI --------- 
#1. Divisione in quattro sottobande ✅
"""
LL  =  L×L  → approssimazione, bassa frequenza          --> robusta ma troppo percettibile
LH  =  L×H  → variazioni verticali, media frequenza     --> standard
HL  =  H×L  → variazioni orizzontali, media frequenza
HH  =  H×H  → dettagli diagonali, alta frequenza        --> troppo fragile
"""
image_float = padded_image.astype(np.float64)   # pywt usa float
coeffs = pywt.dwt2(image_float, 'haar')        #divide immagine in sottobande 
LL, (LH, HL, HH) = coeffs                       #coeffs resta in memoria x ricostruzione
subband = LH

#2. Applicazione mask alla sottobanda
""" La sottobanda ha dimensione dimezzata (512x512 -> 256x256) """
roni_bin = (mask == 255)        # Una maschera è RONI solo se tutti i 4px sono roni
h, w = roni_bin.shape
# Dimezzo RONI per adattarla alla nuova immagine dimezzata
roni_half = roni_bin.reshape(h // 2, 2, w // 2, 2).all(axis=(1, 3))      # Dimezzare mask
assert roni_half.shape == subband.shape

# Suddivisione in blocchi 8x8
block_size = 8
n_block_rows = LH.shape[0] // block_size
n_block_cols = LH.shape[1] // block_size
mask_blocks = roni_half.reshape(n_block_rows, block_size, n_block_cols, block_size).transpose(0, 2, 1, 3)

# Matrice booleana (n_block_rows, n_block_cols): True se il 100% dei pixel nel blocco è RONI 
roni_mask_2d = mask_blocks.all(axis=(2, 3))

# prende blocchi 8x8 e tiene solo quelli interamente in RONI
roni_blocks = np.argwhere(roni_mask_2d)

payload_bits = bytes_to_bits(add_ecc_to_payload(payload))

print("Blocchi RONI disponibili: ", len(roni_blocks))
print("Bit di payload (payload+ECC): ", len(payload_bits))
print("Fattore di ridondanza: ", len(roni_blocks) / len(payload_bits))

#3. applicazione DCT su ogni blocco
subband_wm = subband.copy()
for bit, (row, col) in zip(payload_bits, roni_blocks):
    B = dctn(get_block(subband_wm, row, col), norm='ortho')
    embed_bit(B, bit)
    set_block(subband_wm, row, col, idctn(B, norm='ortho'))

save_intermediate("04_L1_subband_LH_original", subband,
    log_text=f"Sottobanda LH prima dell'embedding L1\nShape: {subband.shape}, "
             f"Min: {subband.min():.2f}, Max: {subband.max():.2f}")
save_intermediate("05_L1_subband_LH_watermarked", subband_wm,
    log_text=f"Sottobanda LH dopo embedding L1 (DWT-DCT)\nShape: {subband_wm.shape}, "
             f"Bit embeddati: {len(payload_bits)}, Blocchi RONI usati: {len(roni_blocks)}\n"
             f"Differenza media |LH_wm - LH|: {np.mean(np.abs(subband_wm - subband)):.4f}")

# reinserimento sottobanda modificata e applicazione IDWT
wm_float = pywt.idwt2((LL, (subband_wm, HL, HH)), 'haar')
# Preserva il dtype originale: clip al range effettivo dell'immagine sorgente
_orig_max = int(original_image.max())
_orig_dtype = original_image.dtype
wm_image = np.clip(np.round(wm_float), 0, _orig_max).astype(_orig_dtype)

save_intermediate("06_L1_watermarked_image", wm_image,
    log_text=f"Immagine dopo watermarking L1 (IDWT)\nShape: {wm_image.shape}, "
             f"PSNR rispetto all'originale con padding: "
             f"{10*np.log10(_orig_max**2 / np.mean((wm_image.astype(float)-padded_image.astype(float))**2)):.2f} dB",
    vmin=IMG_VMIN, vmax=IMG_VMAX)

# Estrazione (blind: non serve l'originale)
_, (LH2, _, _) = pywt.dwt2(wm_image.astype(np.float64), 'haar')
extracted = [extract_bit(dctn(get_block(LH2, r, c), norm='ortho'))
             for r, c in roni_blocks[:len(payload_bits)]]
print("BER (Bit Error Rate):", np.mean(np.array(extracted) != np.array(payload_bits)))
# 0.0 = nessun errore senza attacchi
# DEBUG: verifica round-trip IWT su array piccolo
_test = np.array([[10, 20, 30, 40],
                  [50, 60, 70, 80],
                  [15, 25, 35, 45],
                  [55, 65, 75, 85]], dtype=np.uint8)
_LL, _LH, _HL, _HH = iwt2(_test)
_rec = haar_int_inverse(_LL, _LH, _HL, _HH)
_rec_clipped = np.clip(_rec, 0, 255).astype(np.uint8)
print(f"[DBG] Test array originale:\n{_test}")
print(f"[DBG] Test array ricostruito:\n{_rec_clipped}")
print(f"[DBG] Round-trip esatto: {np.array_equal(_test, _rec_clipped)}")

# --------- Livello 2: IWT-DE con location map --------- 
# IWT
LL_i, LH_i, HL_i, HH_i = iwt2(wm_image)

# ---- Costruzione payload multi-proprietario ----
# Struttura: 4 slot × 8 byte = 32 byte
#   ogni slot = owner_hash_4B (SHA-256[:4]) + timestamp_4B (Unix uint32)
#   slot vuoto = b'\x00' * 8
# In aggiunta: pid_hash_4B (SHA-256 del payload L1, troncato a 4 byte)
# Totale grezzo: 32 + 4 = 36 byte → + ECC 10 byte = 46 byte
MAX_OWNERS = 4
SLOT_BYTES = 8       # 4 byte hash + 4 byte timestamp per proprietario
PID_HASH_BYTES = 4   # link al paziente (da L1)

def build_chain_payload(owner_ids, timestamps, pid_hash_full):
    """
    Costruisce il payload per una catena di proprietari.
    owner_ids   : lista di stringhe (fino a MAX_OWNERS); slot non usati → zeri
    timestamps  : lista di int (epoch Unix); deve avere stessa lunghezza di owner_ids
    pid_hash_full: bytes (SHA-256 del payload L1)
    Ritorna 36 byte: 4×8 slot + 4 pid_hash
    """
    raw = b''
    for i in range(MAX_OWNERS):
        if i < len(owner_ids):
            h = hashlib.sha256(owner_ids[i].encode()).digest()[:4]
            ts = struct.pack('>I', timestamps[i] & 0xFFFFFFFF)  # uint32
        else:
            h = b'\x00' * 4
            ts = b'\x00' * 4
        raw += h + ts
    raw += pid_hash_full[:PID_HASH_BYTES]
    return raw  # 36 byte

def decode_chain_payload(raw_bytes):
    """Decodifica i 36 byte del payload e ritorna una lista di slot + pid_hash."""
    slots = []
    for i in range(MAX_OWNERS):
        offset = i * SLOT_BYTES
        owner_hash = raw_bytes[offset:offset + 4]
        ts_val = struct.unpack('>I', raw_bytes[offset + 4:offset + 8])[0]
        slots.append({'owner_hash': owner_hash.hex(), 'timestamp': ts_val})
    pid_h = raw_bytes[MAX_OWNERS * SLOT_BYTES:MAX_OWNERS * SLOT_BYTES + PID_HASH_BYTES]
    return slots, pid_h

# Primo proprietario nella catena; gli altri slot sono vuoti
owner_ids_chain = ["owner-001"]
timestamps_chain = [int(time.time())]

pid_hash_full = hashlib.sha256(payload).digest()  # lega L2 all'ID di L1

payload_l2_raw  = build_chain_payload(owner_ids_chain, timestamps_chain, pid_hash_full)  # 36 byte
payload_l2      = add_ecc_to_payload(payload_l2_raw, ecc_syms=10)  # 46 byte totali
payload_l2_bits = bytes_to_bits(payload_l2)

print(f"\n[L2] Payload grezzo : {len(payload_l2_raw)} byte  "
      f"({MAX_OWNERS} slot × {SLOT_BYTES}B + {PID_HASH_BYTES}B pid_hash)")
print(f"[L2] Payload + ECC  : {len(payload_l2)} byte  ({len(payload_l2_bits)} bit)")
print(f"[L2] Proprietari nella catena: {len(owner_ids_chain)} / {MAX_OWNERS}")

"""
# Stima capacità e generazione loc_map compressa
_band_test = HL_i.copy()
_, _loc_map_test, _ = de_embed(_band_test, [0] * len(payload_l2_bits))
_loc_bytes, _n_pairs = compress_locmap(_loc_map_test)
_loc_bits = bytes_to_bits(struct.pack('>I', len(_loc_bytes)) + _loc_bytes)

print(f"[L2] Coppie espandibili (pass 1)        : {sum(_loc_map_test)}")
print(f"[L2] Bit loc_map compressa (con header) : {len(_loc_bits)}")
print(f"[L2] Bit payload L2                     : {len(payload_l2_bits)}")
print(f"[L2] Capacità richiesta totale          : {len(_loc_bits) + len(payload_l2_bits)}")

if sum(_loc_map_test) < len(_loc_bits) + len(payload_l2_bits):
    raise RuntimeError("[L2] Capacità insufficiente nella banda HL.")

# 4. Pass 2: embedding reale — [header 4B | loc_map compressa | payload L2]
full_bits = bytes_to_bits(struct.pack('>I', len(_loc_bytes)) + _loc_bytes) + payload_l2_bits
HL_i_wm, loc_map_real, n_embedded = de_embed(HL_i, full_bits)

print(f"[L2] Bit embeddati: {n_embedded} / {len(full_bits)}")
assert n_embedded == len(full_bits), "[L2] Non tutti i bit sono stati embeddati!"

# 5. IWT inversa → immagine finale con entrambi i livelli
wm_final_int = haar_int_inverse(LL_i, LH_i, HL_i_wm, HH_i)
wm_final = np.clip(wm_final_int, 0, 255).astype(np.uint8)
print(f"[L2] Shape immagine finale: {wm_final.shape}")

# ---- Verifica estrazione L2 ----
LL_ex, LH_ex, HL_ex, HH_ex = iwt2(wm_final)
_, all_bits_ex = de_extract(HL_ex, loc_map_real)

header_bits  = all_bits_ex[:32]
loc_len_ex   = struct.unpack('>I', bits_to_bytes(header_bits))[0]
payload_bits_ex = all_bits_ex[32 + loc_len_ex * 8 : 32 + loc_len_ex * 8 + len(payload_l2_bits)]"""

# Pass unico: embedding diretto senza loc_map
HL_i_wm, loc_map_real, n_embedded = de_embed(HL_i, payload_l2_bits, bit_depth=16)

print(f"[L2] Coppie espandibili: {sum(loc_map_real)}")
print(f"[L2] Bit embeddati: {n_embedded} / {len(payload_l2_bits)}")
assert n_embedded == len(payload_l2_bits), "[L2] Capacità insufficiente nella banda HL."

save_intermediate("07_L2_band_HL_original", HL_i.astype(np.float32),
    log_text=f"Banda HL (IWT) prima dell'embedding L2\nShape: {HL_i.shape}, "
             f"Min: {HL_i.min()}, Max: {HL_i.max()}\n"
             f"Struttura payload: {MAX_OWNERS} slot × {SLOT_BYTES}B + {PID_HASH_BYTES}B pid_hash "
             f"= {len(payload_l2_raw)}B grezzo + ECC = {len(payload_l2)}B")
save_intermediate("08_L2_band_HL_watermarked", HL_i_wm.astype(np.float32),
    log_text=f"Banda HL dopo embedding L2 (DE)\nShape: {HL_i_wm.shape}, "
             f"Coppie usate: {sum(loc_map_real)}, Bit embeddati: {n_embedded}/{len(payload_l2_bits)}\n"
             f"Proprietari attivi: {len(owner_ids_chain)}/{MAX_OWNERS}")

# IWT inversa — non clippare prima dell'estrazione
wm_final_int = haar_int_inverse(LL_i, LH_i, HL_i_wm, HH_i)

# Estrazione blind su wm_final_int (senza clip, bit_depth=16 perché i coefficienti IWT non sono limitati a 8 bit)
LL_ex, LH_ex, HL_ex, HH_ex = iwt2(wm_final_int)

print(f"[DBG] HL range: min={HL_i.min()}, max={HL_i.max()}")

band_restored, all_bits_ex = de_extract_blind(HL_ex, len(payload_l2_bits), bit_depth=16)

# Solo per salvataggio/visualizzazione
wm_final = np.clip(wm_final_int, 0, _orig_max).astype(_orig_dtype)
print(f"[L2] Shape immagine finale: {wm_final.shape}")

save_intermediate("09_L2_final_watermarked_image", wm_final,
    log_text=f"Immagine finale dopo watermarking L1+L2\nShape: {wm_final.shape}, "
             f"PSNR rispetto all'originale con padding: "
             f"{10*np.log10(_orig_max**2 / np.mean((wm_final.astype(float)-padded_image.astype(float))**2)):.2f} dB\n"
             f"Differenza media assoluta: {np.mean(np.abs(wm_final.astype(float)-padded_image.astype(float))):.4f}",
    vmin=IMG_VMIN, vmax=IMG_VMAX)

try:
    rs = reedsolo.RSCodec(10)
    decoded = bytes(rs.decode(bits_to_bytes(all_bits_ex))[0])
    slots_ex, pid_h_ex = decode_chain_payload(decoded)

    log_lines = []
    print(f"\n[L2 estrazione] Catena di proprietà ({MAX_OWNERS} slot):")
    for i, slot in enumerate(slots_ex):
        if slot['owner_hash'] == '00000000' and slot['timestamp'] == 0:
            entry = f"  Slot {i+1}: [vuoto]"
        else:
            entry = (f"  Slot {i+1}: owner_hash={slot['owner_hash']}"
                     f"  timestamp={slot['timestamp']}")
        print(entry)
        log_lines.append(entry)

    pid_ok = (pid_h_ex == pid_hash_full[:PID_HASH_BYTES])
    pid_line = f"[L2 estrazione] pid_hash estratto: {pid_h_ex.hex()}  OK={pid_ok}"
    print(pid_line)
    log_lines.append(pid_line)

    save_intermediate("10_L2_extraction_result", None,
        log_text="\n".join(log_lines))
except reedsolo.ReedSolomonError as e:
    print(f"[L2 estrazione] Errore ECC: {e}")
    save_intermediate("10_L2_extraction_result", None,
        log_text=f"Errore ECC durante estrazione L2: {e}")
# --------- Estrazione e block-chain --------- 
# --------- Validazione --------- 

# --------- ATTACCHI ---------
print("\n" + "="*50)
print("TEST DI ROBUSTEZZA")
print("="*50)

attacks = {
    "JP2 ratio=5"  : jpeg2000_attack(wm_final, compression_ratio=5),
    "JP2 ratio=10" : jpeg2000_attack(wm_final, compression_ratio=10),
    "JP2 ratio=20" : jpeg2000_attack(wm_final, compression_ratio=20),
    "Crop 10%"     : crop_attack(wm_final, crop_frac=0.10),
    "Rumore σ=2"   : noise_attack(wm_final, std=2.0),
    "Rumore σ=5"   : noise_attack(wm_final, std=5.0),
    "Rumore σ=10"  : noise_attack(wm_final, std=10.0),
}

log_attack_lines = []
for name, attacked in attacks.items():
    ber, n_bits, extracted_bits = extract_l1_ber(attacked, roni_blocks, payload_bits,
                                                  dataset.Rows, dataset.Columns)
    # Verifica ECC sui bit già estratti (non ricalcola)
    try:
        rs = reedsolo.RSCodec(10)
        rs.decode(bits_to_bytes(extracted_bits))
        ecc_ok = True
    except reedsolo.ReedSolomonError:
        ecc_ok = False

    line = (f"  {name:<15} | BER={ber:.4f} ({int(ber*n_bits)}/{n_bits} bit errati)"
            f" | ECC recupero={'OK' if ecc_ok else 'FALLITO'}")
    print(line)
    log_attack_lines.append(line)

save_intermediate("11_robustness_attacks", None,
    log_text="Test di robustezza L1\n" + "\n".join(log_attack_lines))
