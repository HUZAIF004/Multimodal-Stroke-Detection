# --- FIX: Define JS/JSON literals globally for Python exec environment ---
# This resolves NameError: name 'null' is not defined (thread-safe)
import builtins
builtins.null = None
builtins.false = False
builtins.true = True
# ------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import joblib
import pickle
import torch
import torch.nn as nn
import tensorflow as tf
import io
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from monai.networks.nets import SegResNet
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer 
import tempfile
import nibabel as nib
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
    Spacingd,
)
from scipy import ndimage

# --- !! NEW IMPORTS FOR ZIP/DICOM PROCESSING !! ---
import zipfile
import pydicom
import glob
import warnings
# --------------------------------------------------

# --- Configuration & Initialization ---
print("✅ app.py has started running")
app = Flask(__name__)
CORS(app)

# --- 1. DEFINE FILE AND MODEL PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# --- !! FINAL HYBRID WEIGHTS !! ---
AUC_EHR = 0.9711 
AUC_GENETIC = 0.7251 

TOTAL_AUC = AUC_EHR + AUC_GENETIC

WEIGHTS = {
    "ehr": AUC_EHR / TOTAL_AUC, 
    "genetic": AUC_GENETIC / TOTAL_AUC 
}
print(f"✅ Hybrid weights calculated (EHR/Genetic only): {WEIGHTS}")


# --- 2. LOAD MODELS AND PREPROCESSORS ---
print("🧠 Loading models and preprocessors into memory...")
ehr_model_dnn = ehr_model_xgb = ehr_scaler = ehr_target_encoder = None
genetic_model_dnn = gwas_df = genetic_feature_map = None
mri_models = mri_transforms = None

try:
    # --- EHR Models (Ensemble) ---
    ehr_dnn_path = os.path.join(MODEL_DIR, "ehr", "dnn_model.h5")
    ehr_xgb_path = os.path.join(MODEL_DIR, "ehr", "xgboost_model.pkl")
    ehr_scaler_path = os.path.join(MODEL_DIR, "ehr", "feature_scaler.pkl")
    ehr_target_encoder_path = os.path.join(MODEL_DIR, "ehr", "target_encoder.pkl")

    ehr_model_dnn = tf.keras.models.load_model(ehr_dnn_path)
    ehr_model_xgb = joblib.load(ehr_xgb_path)
    ehr_scaler = joblib.load(ehr_scaler_path)
    ehr_target_encoder = joblib.load(ehr_target_encoder_path)
    
    score_dict = {
        'High': 1.0,
        'Medium': 0.66,
        'Low': 0.33,
        'No': 0.0
    }
    
    ehr_class_names = list(ehr_target_encoder.classes_)
    ehr_score_map = np.array([score_dict.get(class_name, 0.0) for class_name in ehr_class_names])
    
    print(f"EHR Class Names: {ehr_class_names}")
    print(f"EHR Score Map (Corrected): {ehr_score_map}")


    # --- Genetic Model (DNN) ---
    genetic_model_path = os.path.join(MODEL_DIR, "Genetic", "genetic_mlp_model_filtered.keras")
    genetic_model_dnn = tf.keras.models.load_model(genetic_model_path)
    gwas_path = os.path.join(DATA_DIR, "genetic", "gwas_results.assoc")
    
    gwas_df = pd.read_csv(gwas_path, sep=r'\s+')
    
    BEST_N_FEATURES = 1000 
    top_snps_df = gwas_df.sort_values('P').head(BEST_N_FEATURES)
    genetic_feature_map = {
        row['SNP']: (row['A1'], i)
        for i, row in top_snps_df.reset_index().iterrows()
    }
    genetic_feature_names = top_snps_df['SNP'].tolist()

    # --- EHR Encoders (FIXED to match training data) ---
    encoders = {
        'Gender': LabelEncoder().fit(['Male', 'Female', 'Other']),
        'Ever married': LabelEncoder().fit(['Yes', 'No']),
        'Work Type': LabelEncoder().fit(['Private', 'Self-employed', 'Govt_job']),
        'Residence type': LabelEncoder().fit(['Urban', 'Rural']),
        'Smoking Status': LabelEncoder().fit(['formerly smoked', 'never smoked', 'smokes', 'Unknown']),
        'Stress Levels': LabelEncoder().fit(['Yes', 'No']),
        'Alcohol Intake': LabelEncoder().fit(['never taken', 'formerly taken', 'yes']),
        'Family History': LabelEncoder().fit(['Yes', 'No'])
    }
    
    ehr_feature_cols = [
        'Age', 'Hyper tension', 'Heart disease', 'Avg glucose level', 'BMI',
        'HDL', 'LDL', 'Cholesterol Levels', 'Gender_encoded', 
        'Ever married_encoded', 'Work Type_encoded', 'Residence type_encoded',
        'Smoking Status_encoded', 'Stress Levels_encoded', 
        'Alcohol Intake_encoded', 'Family History_encoded'
    ]

    # --- MRI Models (unchanged) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # This is a special class to load a pickle saved on GPU onto a CPU
    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            else:
                return super().find_class(module, name)
                
    mri_models = []
    mri_model_dir = os.path.join(MODEL_DIR, "mri")
    for i in range(1, 6):
        path = os.path.join(mri_model_dir, f"MRI_best_model_fold_{i}.pkl")
        if not os.path.exists(path):
            print(f"Warning: Missing MRI fold model {i}")
            continue
        model_instance = SegResNet(spatial_dims=3, in_channels=2, out_channels=1, init_filters=16).to(device)
        
        try:
            state_dict = torch.load(path, map_location=device, weights_only=False)
        except (RuntimeError, pickle.UnpicklingError):
            print(f"Warning: torch.load failed. Attempting CPU_Unpickler for {path}...")
            with open(path, 'rb') as f:
                state_dict = CPU_Unpickler(f).load()

        model_instance.load_state_dict(state_dict)
        model_instance.eval()
        mri_models.append(model_instance)
    
    # --- !! Corrected Transform Pipeline !! ---
    mri_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=1500, b_min=0.0, b_max=1.0, clip=True),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")), 
        CropForegroundd(keys=["image"], source_key="image"),
        Resized(keys=["image"], spatial_size=[128, 128, 128]),
    ])
    
    MRI_MIN_SCORE = 0.0
    MRI_MAX_SCORE = 515445.26676757773
    print(f"✅ MRI Min/Max scores loaded: {MRI_MIN_SCORE}, {MRI_MAX_SCORE}")

    print("✅ All models and preprocessors loaded successfully!")
except Exception as e:
    print(f"❌ ERROR loading models: {e}")
    # Re-raising the error might be better in a production environment, 
    # but for development, we leave the variables as None and continue.


# --- 3. HELPER & PREDICTION FUNCTIONS ---
# (Helper functions remain unchanged as they were syntactically correct)

def get_ehr_risk_factors(form_data):
    """
    Checks the raw form data for common, high-impact risk factors.
    Returns a list of strings explaining the causes.
    """
    factors = []
    
    # Convert to numbers for safe comparison
    try:
        age = float(form_data.get('Age', 0))
        # Ensure boolean/int inputs are handled safely
        hypertension = int(form_data.get('Hyper tension', 0)) if str(form_data.get('Hyper tension', '0')).isdigit() else 0
        heart_disease = int(form_data.get('Heart disease', 0)) if str(form_data.get('Heart disease', '0')).isdigit() else 0
        glucose = float(form_data.get('Avg glucose level', 0))
        
        bmi_val = form_data.get('BMI', 0)
        bmi = float(bmi_val if (bmi_val and str(bmi_val).strip() and str(bmi_val).replace('.', '', 1).isdigit()) else 0)
        
    except (ValueError, TypeError) as e:
        return [f"Could not parse all EHR form data for explainability. Error: {e}"]

    if age > 60:
        factors.append(f"Major Risk: Age ({int(age)})")
    
    if hypertension == 1:
        factors.append("Major Risk: Active Hypertension")
        
    if heart_disease == 1:
        factors.append("Major Risk: Active Heart Disease")
        
    if glucose > 140:
        factors.append(f"Risk Factor: High Glucose ({glucose} mg/dL)")

    if bmi > 30:
        factors.append(f"Risk Factor: High BMI ({bmi})")
        
    smoking = form_data.get('Smoking Status', 'Unknown').lower()
    if smoking == 'smokes':
        factors.append("Risk Factor: Current Smoker")
    
    if not factors:
        factors.append("No major predefined risk factors identified in clinical data.")
        
    return factors

def synthesize_hdl_ldl(row):
    rng = np.random.default_rng(42)
    hdl = 55.0 + rng.normal(0, 8)
    ldl = 130.0 + rng.normal(0, 20)
    smk = str(row['Smoking Status']).lower()
    if 'smokes' in smk: hdl -= rng.uniform(2, 6)
    elif 'formerly' in smk: hdl -= rng.uniform(0, 3)
    bmi = row['BMI']
    ldl += (bmi - 25) * rng.uniform(0.8, 1.6)
    hdl -= max(0, (bmi - 28)) * rng.uniform(0.2, 0.6)
    glu = row['Avg glucose level']
    ldl += max(0, glu - 110) * rng.uniform(0.1, 0.3)
    hdl -= max(0, glu - 110) * rng.uniform(0.05, 0.12)
    if row['Hyper tension'] == 1: ldl += rng.uniform(5, 12)
    if row['Heart disease'] == 1:
        ldl += rng.uniform(5, 12)
        hdl -= rng.uniform(1, 4)
    row['HDL'] = float(np.clip(hdl, 30, 90))
    row['LDL'] = float(np.clip(ldl, 70, 220))
    row['Cholesterol Levels'] = round((0.6 * row['LDL'] - 0.4 * row['HDL']), 2)
    return row

def synthesize_stress(row):
    rng = np.random.default_rng(42)
    p = 0.25
    if str(row['Work Type']).lower() in ['private', 'self-employed', 'govt_job']: p += 0.15
    if str(row['Residence type']).lower() == 'urban': p += 0.05
    if row['Hyper tension'] == 1 or row['Heart disease'] == 1: p += 0.10
    if row['Age'] >= 60: p += 0.05
    if 'smokes' in str(row['Smoking Status']).lower(): p += 0.05
    row['Stress Levels'] = 'Yes' if rng.uniform() < min(max(p, 0.05), 0.85) else 'No'
    return row

def synthesize_alcohol(row):
    rng = np.random.default_rng(42)
    x = rng.uniform() + (0.05 if str(row['Work Type']).lower() in ['private', 'self-employed'] else 0) + (0.03 if str(row['Residence type']).lower() == 'urban' else 0)
    if x < 0.55: row['Alcohol Intake'] = 'never taken'
    elif x < 0.72: row['Alcohol Intake'] = 'formerly taken'
    elif x < 0.95: row['Alcohol Intake'] = 'yes'
    else: row['Alcohol Intake'] = 'unknown' 
    return row
    
def synthesize_family_history(row):
    rng = np.random.default_rng(42)
    p = 0.18
    if row['Hyper tension'] == 1: p += 0.07
    if row['Heart disease'] == 1: p += 0.05
    if row['Age'] >= 65: p += 0.02
    row['Family History'] = 'Yes' if rng.uniform() < min(p, 0.5) else 'No'
    return row

def calculate_mri_features(mask, voxel_spacing=[1.0, 1.0, 1.0]):
    lesion_volume = np.sum(mask) * np.prod(voxel_spacing)
    if lesion_volume < 1.0:
        return {'predicted_volume_mm3': 0, 'center_of_mass_x': 0, 'center_of_mass_y': 0, 'center_of_mass_z': 0}
    
    com_z, com_y, com_x = ndimage.center_of_mass(mask) 
    
    return {'predicted_volume_mm3': lesion_volume, 'center_of_mass_x': com_x, 'center_of_mass_y': com_y, 'center_of_mass_z': com_z}

def compute_mri_risk_score(volume, com_x, com_y, com_z, img_shape, alpha_z=0.5, alpha_xy=0.3):
    if volume <= 0: return 0.0
    
    z_dim, y_dim, x_dim = img_shape 
    x_center, y_center = x_dim / 2, y_dim / 2
    
    dist = np.sqrt((com_x - x_center)**2 + (com_y - y_center)**2)
    
    D_max = np.sqrt(x_center**2 + y_center**2)
    if D_max == 0: D_max = 1.0
    
    dist_weight = 1 + alpha_xy * (1 - dist / D_max)
    
    if z_dim == 0: z_dim = 1.0
    
    z_weight = 1 + alpha_z * (com_z / z_dim)
    location_weight = dist_weight * z_weight
    
    return volume * location_weight

def stack_dicom_slices(dicom_files):
    """
    Loads, sorts, and stacks DICOM files into a 3D NumPy array.
    """
    slices = []
    for f in dicom_files:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", pydicom.errors.InvalidDicomError)
                ds = pydicom.dcmread(f)
                if hasattr(ds, 'pixel_array') and hasattr(ds, 'InstanceNumber'):
                    slices.append(ds)
        except Exception:
            continue
            
    if not slices:
        return None, None

    slices.sort(key=lambda x: int(x.InstanceNumber))
    
    first = slices[0]
    affine = np.array([
        [first.ImageOrientationPatient[0], first.ImageOrientationPatient[3], 0, first.ImagePositionPatient[0]],
        [first.ImageOrientationPatient[1], first.ImageOrientationPatient[4], 0, first.ImagePositionPatient[1]],
        [0, 0, 1, first.ImagePositionPatient[2]],
        [0, 0, 0, 1]
    ])
    
    image_data = np.stack([s.pixel_array for s in slices], axis=-1)
    
    return image_data, affine

def convert_dicom_to_nifti(dwi_files, adc_files):
    """
    Converts two lists of DICOM files (DWI, ADC) into a single,
    two-channel NIFTI file saved to a temporary path.
    """
    dwi_data, dwi_affine = stack_dicom_slices(dwi_files)
    adc_data, _ = stack_dicom_slices(adc_files)

    if dwi_data is None or adc_data is None:
        return None, "Could not read valid DICOM slices for DWI or ADC."
        
    if dwi_data.shape != adc_data.shape:
        return None, f"DWI and ADC scans have mismatched shapes: {dwi_data.shape} vs {adc_data.shape}"

    combined_data = np.stack([dwi_data, adc_data], axis=-1)
    nifti_img = nib.Nifti1Image(combined_data, dwi_affine)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp:
        nib.save(nifti_img, tmp.name)
        return tmp.name, None

def process_dicom_zip(zip_file):
    """
    Extracts a .zip file, finds DWI and ADC scans, converts them
    to a 2-channel NIFTI file, and returns the path to that file.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(tmpdir)
        except Exception as e:
            return None, f"Could not extract .zip file: {e}"
        
        all_files = glob.glob(os.path.join(tmpdir, '**', '*.dcm'), recursive=True)
        if not all_files:
            return None, "No .dcm (DICOM) files found in the .zip archive."
            
        dwi_files = [f for f in all_files if 'dwi' in f.lower() or 'diffusion' in f.lower()]
        adc_files = [f for f in all_files if 'adc' in f.lower()]

        if not dwi_files:
            return None, "No DICOM files identified as DWI (e.g., 'dwi' in filename)."
        if not adc_files:
            return None, "No DICOM files identified as ADC (e.g., 'adc' in filename)."

        return convert_dicom_to_nifti(dwi_files, adc_files)
        
def process_nifti_files(dwi_file, adc_file):
    """
    Loads two NIFTI files (.nii or .nii.gz) and combines them
    into a single, two-channel NIFTI file saved to a temporary path.
    """
    tmp_dwi_path = None
    tmp_adc_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_dwi:
            dwi_file.save(tmp_dwi)
            tmp_dwi_path = tmp_dwi.name
            dwi_img = nib.load(tmp_dwi_path)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_adc:
            adc_file.save(tmp_adc)
            tmp_adc_path = tmp_adc.name
            adc_img = nib.load(tmp_adc_path)

        dwi_data = dwi_img.get_fdata()
        adc_data = adc_img.get_fdata()
        
        if dwi_data.shape != adc_data.shape:
            return None, f"DWI and ADC NIFTI files have mismatched shapes: {dwi_data.shape} vs {adc_data.shape}"

        combined_data = np.stack([dwi_data, adc_data], axis=-1)
        nifti_img = nib.Nifti1Image(combined_data, dwi_img.affine)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp:
            nib.save(nifti_img, tmp.name)
            
        os.remove(tmp_dwi_path)
        os.remove(tmp_adc_path)
            
        return tmp.name, None
        
    except Exception as e:
        if tmp_dwi_path and os.path.exists(tmp_dwi_path): os.remove(tmp_dwi_path)
        if tmp_adc_path and os.path.exists(tmp_adc_path): os.remove(tmp_adc_path)
        return None, f"Error processing NIFTI files: {e}"

def predict_ehr_risk(form_data):
    """
    Predicts stroke risk (0-1) from EHR data AND returns the causes.
    """
    global ehr_model_dnn, ehr_model_xgb, ehr_scaler, ehr_target_encoder, encoders, ehr_feature_cols, ehr_score_map
    
    ehr_causes = get_ehr_risk_factors(form_data)
    patient_df = pd.DataFrame([form_data.copy()])
    
    # Safely handle numeric conversions and missing values
    for col in ['Age', 'Hyper tension', 'Heart disease', 'Avg glucose level']:
        patient_df[col] = pd.to_numeric(patient_df[col], errors='coerce').fillna(0)
        
    patient_df['BMI'] = pd.to_numeric(patient_df['BMI'], errors='coerce').fillna(0)

    if 'Patient Name' in patient_df:
        patient_df = patient_df.drop(columns=['Patient Name'])

    # Synthesize missing/unknown non-numeric columns and encode them
    for col, le in encoders.items():
        if col not in patient_df or pd.isna(patient_df[col].iloc[0]) or str(patient_df[col].iloc[0]).strip() == '':
            if col in ['Gender', 'Ever married', 'Residence type', 'Stress Levels', 'Family History']:
                patient_df[col] = le.classes_[0] # Default to first class if missing
            elif col == 'Smoking Status':
                patient_df[col] = 'Unknown'
            elif col == 'Work Type':
                patient_df[col] = 'Private'
            elif col == 'Alcohol Intake':
                patient_df[col] = 'never taken'
        
        # Apply synthesis for derived fields if original was missing
        if col == 'Stress Levels' and patient_df['Stress Levels'].iloc[0] == 'Unknown':
             patient_df = patient_df.apply(synthesize_stress, axis=1)
        if col == 'Alcohol Intake' and patient_df['Alcohol Intake'].iloc[0] == 'Unknown':
             patient_df = patient_df.apply(synthesize_alcohol, axis=1)
        if col == 'Family History' and patient_df['Family History'].iloc[0] == 'Unknown':
             patient_df = patient_df.apply(synthesize_family_history, axis=1)
            
    # Synthesize HDL/LDL/Cholesterol (requires Age, HT, HD, Glucose, BMI, Smoking)
    patient_df = patient_df.apply(synthesize_hdl_ldl, axis=1)
    
    # Encoding non-numeric fields
    for col, le in encoders.items():
        if col in patient_df:
            valid_classes = le.classes_
            
            # Temporary fix for 'Unknown' not being in Smoking Status classes during initial fit
            if col == 'Smoking Status' and 'Unknown' not in valid_classes:
                 le.classes_ = np.append(le.classes_, 'Unknown')
                 valid_classes = le.classes_
            
            # Map values, defaulting to a known class if needed
            default_value = 'Unknown' if col == 'Smoking Status' else valid_classes[0]
            patient_df[col] = patient_df[col].apply(lambda x: x if x in valid_classes else default_value)
            patient_df[col + '_encoded'] = le.transform(patient_df[col].astype(str)) # Ensure string input for LE

    # Prepare final feature vector
    X_patient = patient_df[ehr_feature_cols]
    
    # Scaling and prediction
    X_patient_scaled = ehr_scaler.transform(X_patient)
    dnn_proba = ehr_model_dnn.predict(X_patient_scaled, verbose=0)[0]
    xgb_proba = ehr_model_xgb.predict_proba(X_patient_scaled)[0]
    ensemble_proba = (dnn_proba + xgb_proba) / 2.0
    
    final_risk_score = np.dot(ensemble_proba, ehr_score_map)
    
    return final_risk_score, ehr_causes

# The rest of the prediction helpers (predict_genetic_risk, predict_mri_risk) 
# were essentially correct and are included below without change.


def predict_genetic_risk(genetic_file):
    """
    Predicts stroke risk (0-1) from genetic file AND returns top 5 risk SNPs.
    """
    patient_risk_snps = [] 
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, mode='wb') as tmp:
            genetic_file.save(tmp)
            tmp_path = tmp.name
        
        report_df = pd.read_csv(
            tmp_path, comment='#', sep=r'\s+',
            header=0,
            low_memory=False
        )
        
        os.remove(tmp_path)
        report_df.columns = report_df.columns.str.lower()
        patient_genotypes = report_df.set_index('rsid')['genotype'].to_dict()
        
    except Exception as e:
        print(f"Error parsing genetic file: {e}")
        return 0.0, [f"Error parsing file: {e}"]

    feature_vector = np.zeros(BEST_N_FEATURES)
    
    for rsid, (risk_allele, index) in genetic_feature_map.items():
        genotype = patient_genotypes.get(rsid)
        
        if genotype and isinstance(genotype, str):
            count = genotype.count(risk_allele)
            feature_vector[index] = count
            
            if count == 2:
                patient_risk_snps.append(f"{rsid} (Homozygous, 2 risk alleles)")
            elif count == 1:
                patient_risk_snps.append(f"{rsid} (Heterozygous, 1 risk allele)")
                
    X_patient = feature_vector.reshape(1, -1)
    genetic_risk_score = genetic_model_dnn.predict(X_patient, verbose=0)[0, 0]
    
    if not patient_risk_snps:
        patient_risk_snps.append("No high-risk SNPs from the top 1000 list were found.")
    
    return genetic_risk_score, patient_risk_snps[:5]


def predict_mri_risk(nifti_file_path):
    """
    Predicts stroke risk (0-1) from a 2-channel NIFTI file PATH.
    """
    mri_causes = { 
        "lesion_volume_mm3": 0.0,
        "center_of_mass_x": 0.0,
        "center_of_mass_y": 0.0,
        "center_of_mass_z": 0.0,
        "raw_heuristic_score": 0.0
    }
    
    try:
        data_dict = mri_transforms({"image": nifti_file_path})
        image_tensor = data_dict["image"].unsqueeze(0).to(device)
        img_shape = image_tensor.shape[2:] # (z, y, x)

        with torch.no_grad():
            preds = [torch.sigmoid(model(image_tensor)) for model in mri_models]
            ensemble_pred = torch.stack(preds).mean(dim=0)
            predicted_mask = (ensemble_pred > 0.40).cpu().numpy().squeeze()
        
        features = calculate_mri_features(predicted_mask, voxel_spacing=[1.0, 1.0, 1.0])
        
        raw_score = compute_mri_risk_score(
            volume=features.get('predicted_volume_mm3', 0),
            com_x=features.get('center_of_mass_x', 0),
            com_y=features.get('center_of_mass_y', 0),
            com_z=features.get('center_of_mass_z', 0),
            img_shape=img_shape
        )
        
        mri_causes['lesion_volume_mm3'] = float(features.get('predicted_volume_mm3', 0))
        mri_causes['center_of_mass_x'] = float(features.get('center_of_mass_x', 0))
        mri_causes['center_of_mass_y'] = float(features.get('center_of_mass_y', 0))
        mri_causes['center_of_mass_z'] = float(features.get('center_of_mass_z', 0))
        mri_causes['raw_heuristic_score'] = float(raw_score)
        
        if (MRI_MAX_SCORE - MRI_MIN_SCORE) == 0:
            return 0.0, mri_causes
            
        mri_risk_score = (raw_score - MRI_MIN_SCORE) / (MRI_MAX_SCORE - MRI_MIN_SCORE)
        mri_risk_score = np.clip(mri_risk_score, 0.0, 1.0)
        
        return mri_risk_score, mri_causes

    except Exception as e:
        print(f"Error processing MRI file: {e}")
        mri_causes['error'] = f"Error processing file: {e}"
        return 0.0, mri_causes


# --- 4. DEFINE API ROUTE (UPDATED FOR FLEXIBILITY) ---
@app.route('/api/predict', methods=['POST'])
def predict():
    # Initialize outside of try block for cleanup in exception handler
    nifti_path_to_predict = None 
    
    # --- DYNAMIC LOGIC START ---
    
    # Determine which data modalities are present
    form_data = request.form.to_dict()
    genetic_file = request.files.get('geneticReport')
    dicom_zip_file = request.files.get('dicomZip')
    dwi_scan_file = request.files.get('dwiScan')
    adc_scan_file = request.files.get('adcScan')
    
    is_ehr_present = bool(form_data.get('Age')) # Check for a key EHR field
    is_genetic_present = bool(genetic_file)
    is_mri_present = bool(dicom_zip_file or (dwi_scan_file and adc_scan_file))

    # Check if ANY data was submitted
    if not (is_ehr_present or is_genetic_present or is_mri_present):
          return jsonify({"error": "Missing all data. Please submit at least one of EHR, Genetic, or MRI data."}), 400

    try:
        # Ensure models are loaded for the present data
        if is_ehr_present and not all([ehr_model_dnn, ehr_model_xgb]):
            return jsonify({"error": "EHR Models are not loaded on the server. Check server logs."}), 500
        if is_genetic_present and not genetic_model_dnn:
            return jsonify({"error": "Genetic Model is not loaded on the server. Check server logs."}), 500
        if is_mri_present and not mri_models:
            return jsonify({"error": "MRI Models are not loaded on the server. Check server logs."}), 500

        # 1. MRI Processing and Prediction
        mri_score, mri_causes, mri_error = 0.0, {"info": "No MRI data submitted."}, None
        if is_mri_present:
            if dicom_zip_file:
                print("Processing DICOM .zip file...")
                nifti_path_to_predict, mri_error = process_dicom_zip(dicom_zip_file)
            elif dwi_scan_file and adc_scan_file:
                print("Processing NIFTI files...")
                nifti_path_to_predict, mri_error = process_nifti_files(dwi_scan_file, adc_scan_file)
                
            if mri_error:
                # Clean up if processing fails
                if nifti_path_to_predict and os.path.exists(nifti_path_to_predict): os.remove(nifti_path_to_predict)
                return jsonify({"error": f"MRI Processing Error: {mri_error}"}), 500
                
            mri_score, mri_causes = predict_mri_risk(nifti_path_to_predict)
            
        # 2. EHR Prediction
        ehr_score, ehr_causes = 0.0, ["No EHR data submitted."]
        if is_ehr_present:
            ehr_score, ehr_causes = predict_ehr_risk(form_data)
            
        # 3. Genetic Prediction
        genetic_score, genetic_causes = 0.0, ["No genetic data submitted."]
        if is_genetic_present:
            genetic_score, genetic_causes = predict_genetic_risk(genetic_file)

        # Clean up temporary NIFTI file
        if nifti_path_to_predict and os.path.exists(nifti_path_to_predict):
              os.remove(nifti_path_to_predict)

        # 4. Overall Risk Calculation (Dynamic Weighting)
        
        total_chronic_weight = 0.0
        weighted_chronic_sum = 0.0
        
        if is_ehr_present:
            total_chronic_weight += WEIGHTS["ehr"]
            weighted_chronic_sum += ehr_score * WEIGHTS["ehr"]
            
        if is_genetic_present:
            total_chronic_weight += WEIGHTS["genetic"]
            weighted_chronic_sum += genetic_score * WEIGHTS["genetic"]

        long_term_risk = 0.0
        if total_chronic_weight > 0:
            long_term_risk = weighted_chronic_sum / total_chronic_weight
        
        # Define the new weighting logic
        LONG_TERM_WEIGHT = 0.7  
        ACUTE_WEIGHT = 0.3      
        MRI_OVERRIDE_THRESHOLD = 0.8 
        
        decision_reason = ""
        overall_risk = 0.0
        
        if is_mri_present:
            if mri_score > MRI_OVERRIDE_THRESHOLD:
                # HARD OVERRIDE
                overall_risk = mri_score
                decision_reason = (f"Final risk ({overall_risk*100:.1f}%) **determined by a severe acute MRI finding.** "
                                  f"The high MRI score ({mri_score*100:.1f}%) overrides the long-term risk ({long_term_risk*100:.1f}%) due to its urgency.")
            else:
                # SOFT COMBINATION
                if total_chronic_weight > 0:
                    overall_risk = (LONG_TERM_WEIGHT * long_term_risk) + (ACUTE_WEIGHT * mri_score)
                    decision_reason = (f"Final risk ({overall_risk*100:.1f}%) is a **weighted combination** of "
                                      f"long-term risk ({long_term_risk*100:.1f}% weighted at 70%) and "
                                      f"non-severe acute MRI findings ({mri_score*100:.1f}% weighted at 30%).")
                else:
                    # Only MRI data is present
                    overall_risk = mri_score
                    decision_reason = (f"Final risk ({overall_risk*100:.1f}%) is based **only on the acute MRI score** "
                                       f"({mri_score*100:.1f}%) as no chronic data was provided.")
        else:
            # Only chronic data is present
            overall_risk = long_term_risk
            decision_reason = (f"Final risk ({overall_risk*100:.1f}%) is based **only on the long-term chronic risk** "
                               f"({long_term_risk*100:.1f}%) as no MRI data was provided.")
            
        overall_risk = np.clip(overall_risk, 0.0, 1.0)
        
        # --- DYNAMIC LOGIC END ---

        response_data = {
            "ehrRisk": float(ehr_score * 100),
            "geneticRisk": float(genetic_score * 100),
            "mriRisk": float(mri_score * 100),
            "overallRisk": float(overall_risk * 100),
            
            "causes": {
                "ehr": ehr_causes,
                "genetic": genetic_causes,
                "mri": mri_causes, 
                "final_decision_logic": decision_reason
            },
            "submitted_modalities": {
                "EHR": is_ehr_present,
                "Genetic": is_genetic_present,
                "MRI": is_mri_present
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Server Error during prediction: {e}")
        # Cleanup code for nifti_path_to_predict moved outside the try block
        if nifti_path_to_predict and os.path.exists(nifti_path_to_predict):
             os.remove(nifti_path_to_predict)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# --- 5. ROOT ROUTE ---
@app.route('/')
def index():
    try:
        # --- Use the name of your HTML file ---
        with open('index.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return render_template_string(html_content)
    except FileNotFoundError:
        return "<h1>✅ Flask Backend is Running</h1><p>index.html not found. (Make sure it's in the same folder as app.py)</p>"


# --- 6. RUN THE APP (UNCHANGED) ---
if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    app.run(host='127.0.0.1', port=8080, debug=True)