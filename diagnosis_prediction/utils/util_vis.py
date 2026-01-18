import re


Hospital_Code_Dict = {
    'Nanfang': 'H1',
    'NanFang': 'H1',
    'ZJ1': 'H2',
    'YNFirst': 'H3',
    'YNCH': 'H4',
    'Qianfoshan': 'H5',
    'Hebeisiyuan': 'H6',
    'PWH': 'H7',
    'argo': 'H8', # 中山六院colon
    'SDPH': 'H9',

    'test': 'internal',
    'Prospective': 'Prospective',
}

Task_Name_Dict_with_organ = {
    # lung
    'Nanfang_primary_metastatic': '[Lung] Primary/Metastatic Classification',
    # 'Nanfang_cohort1': '[Lung] Primary Site Prediction',
    'Nanfang_lung_finegrained': '[Lung] Cancer Subtyping',
    'Nanfang-Lung-NSCLC': '[Lung] NSCLC Subtyping',
    'Nanfang-Lung-Frozen-LymphNodeMetastasis': '[Lung] Lymph Node Metastasis Prediction',
    'Nanfang_lung_P63': '[Lung] IHC Status Prediction - P63',

    # Breast
    'ZJ1-C1_Breast_TNM-N': '[Breast] TNM-N Staging (N0/N+)',
    'ZJ1-C1_Breast_pTNM': '[Breast] pTNM Overall Staging (I/II/III)',
    # --
    'ZJ1-C1_Breast_MolSubtype': '[Breast] Molecular Subtyping',
    'ZJ1-C1_Breast_IHC-AR': '[Breast] IHC Status Prediction - AR',  # "IHC Expression Prediction - AR" 或 "AR IHC Status Prediction"
    'ZJ1-C1_Breast_IHC-ER': '[Breast] IHC Status Prediction - ER',
    'ZJ1-C1_Breast_IHC-PR': '[Breast] IHC Status Prediction - PR',
    'ZJ1-C1_Breast_IHC-HER2': '[Breast] IHC Status Prediction - HER2',
    'ZJ1-C1_Breast_IHC-CK5': '[Breast] IHC Status Prediction - CK5',

    # Brain
    'NanFang_Glioma_Pathological_Subtype': '[Brain] Pathological Subtyping',
    'NanFang_Glioma_WHO_Grading': '[Brain] WHO Grading',
    # --
    'NanFang_Glioma_IDH_Mutation': '[Brain] Biomarker Prediction - IDH Mutation',

    # Gastric
    'NanFang_Gastric_Grade': '[Gastric] Cancer Grading',
    'NanFang_Gastric_Lauren': '[Gastric] Lauren Subtyping',
    'NanFang_Gastric_PathSubtype': '[Gastric] Pathological Subtyping',
    'NanFang_Gastric_TNM-N': '[Gastric] TNM-N Staging (N0/N+)',
    'NanFang_Gastric_TNM-T': '[Gastric] TNM-T Staging (T1/T2/T3/T4)',
    'PWH_Stomach_Biopsy_Normal_Abnormal': '[Gastric] Normal/Abnormal Classification',
    'PWH_Stomach_Biopsy_Intestinal_metaplasia': '[Gastric] Intestinal Metaplasia Classification',
    'NanFang_Gastric_Perineural': '[Gastric] Perineural Invasion Detection',
    'NanFang_Gastric_Vascular': '[Gastric] Vascular Invasion Detection',
    'PWH_Stomach_Abnormal_is_ACGxHP': '[Gastric] Abnormal Tissue Subtyping - ACGxHP',
    'PWH_Stomach_Abnormal_is_HPACG': '[Gastric] Abnormal Tissue Subtyping - HPACG',
    'PWH_Stomach_Abnormal_is_Polyp': '[Gastric] Abnormal Tissue Subtyping - Polyp',
    'PWH_Stomach_Abnormal_is_Ulcer': '[Gastric] Abnormal Tissue Subtyping - Ulcer',
    # --
    'NanFang_Gastric_IHC-HER-2': '[Gastric] IHC Status Prediction - HER2',
    'NanFang_Gastric_IHC-S-100': '[Gastric] IHC Status Prediction - S-100',

    # Colorectal Cancer
    'ARGO-TNM_N0_N+': '[Colorectal] TNM-N Staging (N0/N+)',
    'ARGO-TNM_T1+T2_T3+T4': '[Colorectal] TNM-T Staging (T1+T2/T3+T4)',
    'ARGO-TNM_T1_T4': '[Colorectal] TNM-T Staging (T1/T2/T3/T4)',
    'argo_colon_staging': '[Colorectal] TNM Overall Staging (I/II/III/IV)',
    # --
    'argo_colon_deep_cms': '[Colorectal] Consensus Molecular Subtyping',
}

Task_Name_Dict = {
    # lung
    'Nanfang_primary_metastatic': 'Primary/Metastatic Classification',
    # 'Nanfang_cohort1': 'Primary Site Prediction',
    'Nanfang_lung_finegrained': 'Cancer Subtyping',
    'Nanfang-Lung-NSCLC': 'NSCLC Subtyping',
    'Nanfang-Lung-Frozen-LymphNodeMetastasis': 'Lymph Node Metastasis Prediction',
    'Nanfang_lung_P63': 'IHC Status Prediction - P63',

    # Breast
    'ZJ1-C1_Breast_TNM-N': 'TNM-N Staging (N0/N+)',
    'ZJ1-C1_Breast_pTNM': 'pTNM Overall Staging (I/II/III)',
    # --
    'ZJ1-C1_Breast_MolSubtype': 'Molecular Subtyping',
    'ZJ1-C1_Breast_IHC-AR': 'IHC Status Prediction - AR',  # "IHC Expression Prediction - AR" 或 "AR IHC Status Prediction"
    'ZJ1-C1_Breast_IHC-ER': 'IHC Status Prediction - ER',
    'ZJ1-C1_Breast_IHC-PR': 'IHC Status Prediction - PR',
    'ZJ1-C1_Breast_IHC-HER2': 'IHC Status Prediction - HER2',
    'ZJ1-C1_Breast_IHC-CK5': 'IHC Status Prediction - CK5',

    # Brain
    'NanFang_Glioma_Pathological_Subtype': 'Pathological Subtyping',
    'NanFang_Glioma_WHO_Grading': 'WHO Grading',
    # --
    'NanFang_Glioma_IDH_Mutation': 'Biomarker Prediction - IDH Mutation',

    # Gastric
    'NanFang_Gastric_Grade': 'Cancer Grading',
    'NanFang_Gastric_Lauren': 'Lauren Subtyping',
    'NanFang_Gastric_PathSubtype': 'Pathological Subtyping',
    'NanFang_Gastric_TNM-N': 'TNM-N Staging (N0/N+)',
    'NanFang_Gastric_TNM-T': 'TNM-T Staging (T1/T2/T3/T4)',
    'PWH_Stomach_Biopsy_Normal_Abnormal': 'Normal/Abnormal Classification',
    'PWH_Stomach_Biopsy_Intestinal_metaplasia': 'Intestinal Metaplasia Classification',
    'NanFang_Gastric_Perineural': 'Perineural Invasion Detection',
    'NanFang_Gastric_Vascular': 'Vascular Invasion Detection',
    'PWH_Stomach_Abnormal_is_ACGxHP': 'Abnormal Tissue Subtyping - ACGxHP',
    'PWH_Stomach_Abnormal_is_HPACG': 'Abnormal Tissue Subtyping - HPACG',
    'PWH_Stomach_Abnormal_is_Polyp': 'Abnormal Tissue Subtyping - Polyp',
    'PWH_Stomach_Abnormal_is_Ulcer': 'Abnormal Tissue Subtyping - Ulcer',
    # --
    'NanFang_Gastric_IHC-HER-2': 'IHC Status Prediction - HER2',
    'NanFang_Gastric_IHC-S-100': 'IHC Status Prediction - S-100',

    # Colorectal Cancer
    'ARGO-TNM_N0_N+': 'TNM-N Staging (N0/N+)',
    'ARGO-TNM_T1+T2_T3+T4': 'TNM-T Staging (T1+T2/T3+T4)',
    'ARGO-TNM_T1_T4': 'TNM-T Staging (T1/T2/T3/T4)',
    'argo_colon_staging': 'TNM Overall Staging (I/II/III/IV)',
    # --
    'argo_colon_deep_cms': 'Consensus Molecular Subtyping',
}

task_order = [
    # lung
    'Nanfang_primary_metastatic',
    'Nanfang_cohort1',
    # Breast
    'ZJ1-C1_Breast_TNM-N',
    'ZJ1-C1_Breast_pTNM',
    # Brain
    'NanFang_Glioma_Pathological_Subtype',
    'NanFang_Glioma_WHO_Grading',
    # Gastric
    'NanFang_Gastric_Grade',
    'NanFang_Gastric_Lauren',
    'NanFang_Gastric_PathSubtype',
    'NanFang_Gastric_TNM-N',
    'NanFang_Gastric_TNM-T',
    'PWH_Stomach_Biopsy_Normal_Abnormal',
    'PWH_Stomach_Abnormal_is_ACGxHP',
    'PWH_Stomach_Biopsy_Intestinal_metaplasia',
    'NanFang_Gastric_Perineural',
    'NanFang_Gastric_Vascular',
    'PWH_Stomach_Abnormal_is_HPACG',
    'PWH_Stomach_Abnormal_is_Polyp',
    'PWH_Stomach_Abnormal_is_Ulcer',
    # Colorectal Cancer
    'ARGO-TNM_N0_N+',
    'ARGO-TNM_T1+T2_T3+T4',
    'ARGO-TNM_T1_T4',
    'argo_colon_staging',

    # --
    'ZJ1-C1_Breast_MolSubtype',
    'ZJ1-C1_Breast_IHC-AR',
    'ZJ1-C1_Breast_IHC-ER',
    'ZJ1-C1_Breast_IHC-PR',
    'ZJ1-C1_Breast_IHC-HER2',
    'ZJ1-C1_Breast_IHC-CK5',
    # --
    'NanFang_Glioma_IDH_Mutation',
    # --
    'NanFang_Gastric_IHC-HER-2',
    'NanFang_Gastric_IHC-S-100',
    # -- 
    'argo_colon_deep_cms',
]

# 任务→器官的映射，可按需要补充/调整
organ_map = {
    # 肺
    'Nanfang_primary_metastatic': 'Lung',
    # 'Nanfang_cohort1': 'Lung',
    'Nanfang-Lung-NSCLC': 'Lung',
    'Nanfang_lung_finegrained': 'Lung',
    'Nanfang-Lung-Frozen-LymphNodeMetastasis': 'Lung',
    'Nanfang_lung_P63': 'Lung',

    # 乳腺
    'ZJ1-C1_Breast_TNM-N': 'Breast',
    'ZJ1-C1_Breast_pTNM': 'Breast',
    'ZJ1-C1_Breast_MolSubtype': 'Breast',
    'ZJ1-C1_Breast_IHC-AR': 'Breast',
    'ZJ1-C1_Breast_IHC-ER': 'Breast',
    'ZJ1-C1_Breast_IHC-PR': 'Breast',
    'ZJ1-C1_Breast_IHC-HER2': 'Breast',
    'ZJ1-C1_Breast_IHC-CK5': 'Breast',
    # 脑
    'NanFang_Glioma_Pathological_Subtype': 'Brain',
    'NanFang_Glioma_WHO_Grading': 'Brain',
    'NanFang_Glioma_IDH_Mutation': 'Brain',
    # 胃/胃镜
    'NanFang_Gastric_Grade': 'Gastric',
    'NanFang_Gastric_Lauren': 'Gastric',
    'NanFang_Gastric_PathSubtype': 'Gastric',
    'NanFang_Gastric_TNM-N': 'Gastric',
    'NanFang_Gastric_TNM-T': 'Gastric',
    'NanFang_Gastric_Perineural': 'Gastric',
    'NanFang_Gastric_Vascular': 'Gastric',
    'NanFang_Gastric_IHC-HER-2': 'Gastric',
    'NanFang_Gastric_IHC-S-100': 'Gastric',
    'PWH_Stomach_Biopsy_Normal_Abnormal': 'Gastric',
    'PWH_Stomach_Biopsy_Intestinal_metaplasia': 'Gastric',
    'PWH_Stomach_Abnormal_is_ACGxHP': 'Gastric',
    'PWH_Stomach_Abnormal_is_Ulcer': 'Gastric',
    'PWH_Stomach_Abnormal_is_HPACG': 'Gastric',
    'PWH_Stomach_Abnormal_is_Polyp': 'Gastric',
    # 结直肠
    'ARGO-TNM_N0_N+': 'Colorectal',
    'ARGO-TNM_T1+T2_T3+T4': 'Colorectal',
    'ARGO-TNM_T1_T4': 'Colorectal',
    'argo_colon_staging': 'Colorectal',
    'argo_colon_deep_cms': 'Colorectal',
}

task_type_map = {
    # 肺
    'Nanfang_primary_metastatic': 'Histological Diagnosis',
    # 'Nanfang_cohort1': 'Histological Diagnosis',
    'Nanfang_lung_finegrained': 'Histological Diagnosis',
    'Nanfang-Lung-NSCLC': 'Histological Diagnosis',
    'Nanfang-Lung-Frozen-LymphNodeMetastasis': 'Histological Diagnosis',
    'Nanfang_lung_P63': 'Molecular Prediction',
    # 乳腺
    'ZJ1-C1_Breast_TNM-N': 'Histological Diagnosis',
    'ZJ1-C1_Breast_pTNM': 'Histological Diagnosis',
    'ZJ1-C1_Breast_MolSubtype': 'Molecular Prediction',
    'ZJ1-C1_Breast_IHC-AR': 'Molecular Prediction',
    'ZJ1-C1_Breast_IHC-ER': 'Molecular Prediction',
    'ZJ1-C1_Breast_IHC-PR': 'Molecular Prediction',
    'ZJ1-C1_Breast_IHC-HER2': 'Molecular Prediction',
    'ZJ1-C1_Breast_IHC-CK5': 'Molecular Prediction',
    # 脑
    'NanFang_Glioma_Pathological_Subtype': 'Histological Diagnosis',
    'NanFang_Glioma_WHO_Grading': 'Histological Diagnosis',
    'NanFang_Glioma_IDH_Mutation': 'Molecular Prediction',
    # 胃/胃镜
    'NanFang_Gastric_Grade': 'Histological Diagnosis',
    'NanFang_Gastric_Lauren': 'Histological Diagnosis',
    'NanFang_Gastric_PathSubtype': 'Histological Diagnosis',
    'NanFang_Gastric_TNM-N': 'Histological Diagnosis',
    'NanFang_Gastric_TNM-T': 'Histological Diagnosis',
    'NanFang_Gastric_Perineural': 'Histological Diagnosis',
    'NanFang_Gastric_Vascular': 'Histological Diagnosis',
    'NanFang_Gastric_IHC-HER-2': 'Molecular Prediction',
    'NanFang_Gastric_IHC-S-100': 'Molecular Prediction',
    'PWH_Stomach_Biopsy_Normal_Abnormal': 'Histological Diagnosis',
    'PWH_Stomach_Biopsy_Intestinal_metaplasia': 'Histological Diagnosis',
    'PWH_Stomach_Abnormal_is_ACGxHP': 'Histological Diagnosis',
    'PWH_Stomach_Abnormal_is_Ulcer': 'Histological Diagnosis',
    'PWH_Stomach_Abnormal_is_HPACG': 'Histological Diagnosis',
    'PWH_Stomach_Abnormal_is_Polyp': 'Histological Diagnosis',
    # 结直肠
    'ARGO-TNM_N0_N+': 'Histological Diagnosis',
    'ARGO-TNM_T1+T2_T3+T4': 'Histological Diagnosis',
    'ARGO-TNM_T1_T4': 'Histological Diagnosis',
    'argo_colon_staging': 'Histological Diagnosis',
    'argo_colon_deep_cms': 'Molecular Prediction',
}


organ_task = {
    'Lung': ['Nanfang_primary_metastatic', 'Nanfang-Lung-NSCLC', 'Nanfang_lung_finegrained', 
             'Nanfang-Lung-Frozen-LymphNodeMetastasis', 'Nanfang_lung_P63'],
    'Breast': ['ZJ1-C1_Breast_TNM-N', 'ZJ1-C1_Breast_pTNM', 'ZJ1-C1_Breast_MolSubtype', 'ZJ1-C1_Breast_IHC-AR', 'ZJ1-C1_Breast_IHC-ER', 
               'ZJ1-C1_Breast_IHC-PR', 'ZJ1-C1_Breast_IHC-HER2', 'ZJ1-C1_Breast_IHC-CK5'],
    'Brain': ['NanFang_Glioma_IDH_Mutation', 'NanFang_Glioma_Pathological_Subtype', 'NanFang_Glioma_WHO_Grading'],
    'Gastric': ['PWH_Stomach_Biopsy_Normal_Abnormal', 'PWH_Stomach_Abnormal_is_ACGxHP', 'PWH_Stomach_Biopsy_Intestinal_metaplasia', 
                'NanFang_Gastric_Grade', 'NanFang_Gastric_IHC-HER-2', 'NanFang_Gastric_IHC-S-100', 'NanFang_Gastric_Lauren', 
                'NanFang_Gastric_PathSubtype', 'NanFang_Gastric_Perineural', 'NanFang_Gastric_Vascular', 'NanFang_Gastric_TNM-N', 
                'NanFang_Gastric_TNM-T',  'PWH_Stomach_Abnormal_is_Ulcer',
                # 'PWH_Stomach_Abnormal_is_HPACG', 'PWH_Stomach_Abnormal_is_Polyp',
                ],
    'Colon': ['ARGO-TNM_N0_N+', 'ARGO-TNM_T1+T2_T3+T4', 'ARGO-TNM_T1_T4', 'argo_colon_staging', 'argo_colon_deep_cms'],
}

def task_organ(sheet):
    for organ in organ_task.keys():
        for item in organ_task[organ]:
            if item in sheet:
                return organ
    return None


model_camel_dict = {
    # 'litepath': 'LitePath-S',
    # 'litepath-h': 'LitePath-L',
    # 'litepath-l': 'LitePath',  # Note: present only litepath-l, and renmame to "litepath" (as the main model of paper)
    'litepath': 'LiteFM-S',
    'litepath-h': 'LiteFM-L',
    'litepath-l': 'LiteFM',
    'litepath-l-APS': 'LitePath',
    'litepath-l-virchow2': 'LiteVirchow2',
    'virchow2': 'Virchow2',
    'h-optimus-1': 'H-Optimus-1',
    'uni2': 'UNI2',
    'mstar': 'mSTAR',
    'gpfm': 'GPFM',
    'uni': 'UNI',
    'gigapath': 'Prov-GigaPath',
    'conch15': 'CONCH1.5',
    'conch': 'CONCH',
    'phikon2': 'Phikon2',
    'phikon': 'Phikon',
    'virchow': 'Virchow',
    'ctranspath': 'CTransPath',
    'chief': 'CHIEF',
    'musk': 'MUSK',
    'hibou-l': 'Hibou-L',
    'plip': 'PLIP',
    'resnet50': 'Resnet50'
}

SPLIT_CSV = {
    # lung
    'Nanfang_primary_metastatic': '/jhcnas4/Pathology/code/PathTasks/data/lung/primary_meta/data/Nanfang_primary_metastatic.xlsx',
    # 'Nanfang_cohort1': '/jhcnas4/Pathology/code/PathTasks/data/lung/primary_site_prediction/data/Nanfang_cohort1.xlsx',
    # 'IHC_NF1_C-met': '/jhcnas4/Pathology/code/PathTasks/data/lung/IHC_NF_C-met/data/IHC_NF1_C-met_VALID_CASE.xlsx',  # Delete task
    # 'IHC_NF1_CK7': '/jhcnas4/Pathology/code/PathTasks/data/lung/IHC_NF_CK7/data/Nanfang_lung_CK7_merged_VALID_CASE.xlsx',
    # 'IHC_NF1_TTF-1': '/jhcnas4/Pathology/code/PathTasks/data/lung/IHC_NF_TTF-1/data/Nanfang_lung_TTF1_merged_VALID_CASE.xlsx',
    # 'IHC_NF1_NapsinA': '/jhcnas4/Pathology/code/PathTasks/data/lung/IHC_NF_Napsin-A/data/Nanfang_lung_NapsinA_merged_VALID_CASE.xlsx',
    # 'Nanfang_lung_cancerVSbenign': '/jhcnas4/Pathology/code/PathTasks/data/lung/cancer_vs_benign/data/Nanfang_lung_cancerVSbenign.xlsx',
    'Nanfang-Lung-NSCLC': '/jhcnas4/Pathology/code/PathTasks/data/lung/NSCLC/data/Nanfang_lung_NSCLC_VALID.xlsx',
    'Nanfang_lung_P63': '/jhcnas4/Pathology/code/PathTasks/data/lung/IHC_NF_P63/data/Nanfang_lung_P63_merged_VALID_CASE.xlsx',
    'Nanfang_lung_finegrained': '/jhcnas4/Pathology/code/PathTasks/data/lung/finegrained_classification/data/Nanfang_lung_finegrained_cleaned.xlsx',
    'Nanfang-Lung-Frozen-LymphNodeMetastasis': '/jhcnas4/Pathology/code/PathTasks/data/lung/NF_Lymph_Metastasis_Frozen/data/Nanfang_lung_LymphNodeMetastasis_Frozen.xlsx',
    
    # breast
    'ZJ1-C1_Breast_TNM-N': '/jhcnas4/Pathology/code/PathTasks/data/breast/TNM-N/output/ZJ1-C1_Breast_TNM-N.xlsx',
    'ZJ1-C1_Breast_pTNM': '/jhcnas4/Pathology/code/PathTasks/data/breast/pTNM/data/ZJ1-C1_Breast_pTNM.xlsx',
    'ZJ1-C1_Breast_MolSubtype': '/jhcnas4/Pathology/code/PathTasks/data/breast/Molecular_Subtype/data/ZJ1-C1_Breast_MolSubtype.xlsx',
    'ZJ1-C1_Breast_IHC-AR': '/jhcnas4/Pathology/code/PathTasks/data/breast/IHC-AR/output/ZJ1-C1_Breast_IHC-AR.xlsx',
    'ZJ1-C1_Breast_IHC-ER': '/jhcnas4/Pathology/code/PathTasks/data/breast/IHC-ER/output/ZJ1-C1_Breast_IHC-ER.xlsx',
    'ZJ1-C1_Breast_IHC-PR': '/jhcnas4/Pathology/code/PathTasks/data/breast/IHC-PR/data/ZJ1-C1_Breast_IHC-PR.xlsx',
    'ZJ1-C1_Breast_IHC-HER2': '/jhcnas4/Pathology/code/PathTasks/data/breast/IHC-HER2/output/ZJ1-C1_Breast_IHC-HER2.xlsx',
    'ZJ1-C1_Breast_IHC-CK5': '/jhcnas4/Pathology/code/PathTasks/data/breast/IHC-CK5/data/ZJ1-C1_Breast_IHC-CK5.xlsx',
    
    # brain
    'NanFang_Glioma_IDH_Mutation': '/jhcnas4/Pathology/code/PathTasks/data/brain/IDH_Mutation/data/NanFang_Glioma_IDH_Mutation.xlsx',
    'NanFang_Glioma_Pathological_Subtype': '/jhcnas4/Pathology/code/PathTasks/data/brain/Pathological_Subtype/data/NanFang_Glioma_Pathological_Subtype.xlsx',
    'NanFang_Glioma_WHO_Grading': '/jhcnas4/Pathology/code/PathTasks/data/brain/WHO_Grading/data/NanFang_Glioma_WHO_Grading.xlsx',
    
    # Gastric
    'PWH_Stomach_Biopsy_Normal_Abnormal': '/jhcnas4/Pathology/code/PathTasks/data/gastric/Normal_or_Abnormal/data/PWH_Stomach_Biopsy_Normal_Abnormal.xlsx',
    'PWH_Stomach_Abnormal_is_ACGxHP': '/jhcnas4/Pathology/code/PathTasks/data/gastric/Biopsy_ACGxHP/data/PWH_Stomach_Abnormal_is_ACGxHP.xlsx',
    'PWH_Stomach_Biopsy_Intestinal_metaplasia': '/jhcnas4/Pathology/code/PathTasks/data/gastric/Intestinal_metaplasia/data/PWH_Stomach_Biopsy_Intestinal_metaplasia.xlsx',
    'NanFang_Gastric_Grade': '/jhcnas4/Pathology/code/PathTasks/data/gastric/Grade/output/NanFang_Gastric_Grade.xlsx',
    'NanFang_Gastric_IHC-HER-2': '/jhcnas4/Pathology/code/PathTasks/data/gastric/IHC-HER-2/output/NanFang_Gastric_IHC-HER-2.xlsx',
    'NanFang_Gastric_IHC-S-100': '/jhcnas4/Pathology/code/PathTasks/data/gastric/IHC-S-100/data/NanFang_Gastric_IHC-S-100.xlsx',
    'NanFang_Gastric_Lauren': '/jhcnas4/Pathology/code/PathTasks/data/gastric/Lauren/data/NanFang_Gastric_Lauren.xlsx',
    'NanFang_Gastric_PathSubtype': '/jhcnas4/Pathology/code/PathTasks/data/gastric/PathSubtype/data/NanFang_Gastric_PathSubtype.xlsx',
    'NanFang_Gastric_Perineural': '/jhcnas4/Pathology/code/PathTasks/data/gastric/Perineural/data/NanFang_Gastric_Perineural.xlsx',
    'NanFang_Gastric_Vascular': '/jhcnas4/Pathology/code/PathTasks/data/gastric/Vascular/data/NanFang_Gastric_Vascular.xlsx',
    'NanFang_Gastric_TNM-N': '/jhcnas4/Pathology/code/PathTasks/data/gastric/TNM-N/data/NanFang_Gastric_TNM-N.xlsx',
    'NanFang_Gastric_TNM-T': '/jhcnas4/Pathology/code/PathTasks/data/gastric/TNM-T/data/NanFang_Gastric_TNM-T.xlsx',
    'PWH_Stomach_Abnormal_is_Ulcer': '/jhcnas4/Pathology/code/PathTasks/data/gastric/Biopsy_Ulcer/data/PWH_NanFang-GC_23-25_Biopsy_Ulcer-TEMP-ZFT.xlsx', # TODO
    
    # Colon
    'ARGO-TNM_N0_N+': '/jhcnas4/Pathology/code/PathTasks/data/colon/TNM_N/data/ARGO-TNM_N0_N+.xlsx',
    'ARGO-TNM_T1+T2_T3+T4': '/jhcnas4/Pathology/code/PathTasks/data/colon/TNM_T/data/ARGO-TNM_T1+T2_T3+T4.xlsx',
    'ARGO-TNM_T1_T4': '/jhcnas4/Pathology/code/PathTasks/data/colon/TNM_T/data/ARGO-TNM_T1_T4.xlsx',
    'argo_colon_staging': '/jhcnas4/Pathology/code/PathTasks/data/colon/TNM_Staging/data/argo_colon_staging.xlsx',
    'argo_colon_deep_cms': '/jhcnas4/Pathology/code/PathTasks/data/colon/Deep_CMS/data/argo_colon_deep_cms.xlsx',
}


model_order = ['resnet50', 'ctranspath', 'conch', 'phikon', 'plip', 'uni', 'virchow', 'gigapath', 'hibou-l', 'mstar', 'gpfm', 'virchow2', 'chief',
               'phikon2', 'conch15', 'musk', 'uni2', 'h-optimus-1', 'litepath-l']

def get_title_info(task_with_split, refine_task_name=False):
    study, hospital = extract_from_title(task_with_split)

    internal_hospital = get_hospital_code(study)
    eval_hospital = get_hospital_code(hospital)
    
    if eval_hospital == 'internal':
        eval_hospital = f"Internal-{internal_hospital}"
    elif eval_hospital == 'Prospective':
        eval_hospital = f'Prospective-{internal_hospital}'
    else:
        eval_hospital = f"External-{eval_hospital}"
    # eval_hospital = f"Internal-{internal_hospital}" if eval_hospital == 'internal' else f"External-{eval_hospital}" if eval_hospital != 'Prospective' else f'Prospective-{internal_hospital}'

    if refine_task_name:
        task_name = get_task_name(study)
    else:
        task_name = study
    return task_name, eval_hospital


def extract_from_title(title):
    match = re.match(r'([^(]+)\(([^)]+)\)', title)
    assert match is not None, f"format not match for {title}"
    study = match.group(1).strip()   # 括号前
    hospital = match.group(2).strip()   # 括号里
    return study, hospital


# def get_hospital_code(info):
#     # study can be task name or hospital name
#     if info == 'test':
#         return 'internal'

#     for key, value in Hospital_Code_Dict.items():
#         if key in info or key.upper() in info:
#             return value
#     return None
def get_hospital_code(info):
    # study can be task name or hospital name
    if info == 'test':
        return 'internal'

    if info == 'Prospective':
        return 'Prospective'

    if info == 'NanFang_Gastric_IHC-HER-2' or info == 'NanFang_Gastric_IHC-S-100':
        return 'H1+H3+H4'
    
    result = []
    for key, value in Hospital_Code_Dict.items():
        if key == 'test':
            continue
        if key in info or key.upper() in info:
            result.append(value)
    return "+".join(result) if result else None


def get_task_name(study):
    return Task_Name_Dict[study]
