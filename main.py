import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import gzip
import shutil

app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://ddos-point-fe.vercel.app",
    "https://ddos-point-8j91y58vv-sidratul-muntaha-tubas-projects.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pickleFileSpecific = './Models/ddos-detecting-ensemble-model-50000-all-specific-0.9.pkl.gz'
pickleFileGeneral = './Models/ddos-detecting-ensemble-model-100000-top60-general-0.85.pkl.gz'

def unzip_model(file_path):
    if file_path.endswith('.gz'):
        unzipped_file_path = file_path[:-3]
        with gzip.open(file_path, 'rb') as f_in:
            with open(unzipped_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return unzipped_file_path
    return file_path

try:
    specific_model_path = unzip_model(pickleFileSpecific)
    modelForSpecific = joblib.load(specific_model_path)
    print(modelForSpecific)
    
    general_model_path = unzip_model(pickleFileGeneral)
    modelForGeneral = joblib.load(general_model_path)
    print(modelForGeneral)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

impFeatureSet = ['Protocol_17', 'Idle Max', 'Flow IAT Max', 'Flow Duration',
'Protocol_6', 'Fwd IAT Max', 'Fwd IAT Total', 'Idle Mean',
'ACK Flag Count', 'Fwd IAT Std', 'Fwd Packet Length Mean',
'Avg Fwd Segment Size', 'Packet Length Min', 'Fwd Packet Length Min',
'Idle Min', 'Flow IAT Std', 'Packet Length Mean', 'Fwd IAT Mean',
'Avg Packet Size', 'Bwd IAT Max', 'Bwd IAT Total', 'Flow IAT Mean',
'Fwd Packet Length Max', 'Bwd IAT Std', 'Packet Length Max', 'Idle Std',
'Bwd Packet Length Max', 'Bwd IAT Mean', 'Bwd Packet Length Std',
'Packet Length Std', 'Init Fwd Win Bytes', 'Avg Bwd Segment Size',
'Bwd Packet Length Mean', 'Down/Up Ratio', 'Active Max',
'Fwd Packet Length Std', 'URG Flag Count', 'Flow Bytes/s',
'Packet Length Variance', 'Active Mean', 'Active Std',
'Subflow Bwd Packets', 'Total Backward Packets', 'Flow Packets/s',
'Fwd Packets/s', 'Fwd Act Data Packets', 'CWE Flag Count',
'Subflow Bwd Bytes', 'Bwd Packets Length Total', 'Subflow Fwd Bytes',
'Fwd Packets Length Total', 'Bwd Packet Length Min', 'Fwd PSH Flags',
'RST Flag Count', 'Active Min', 'Init Bwd Win Bytes', 'Bwd IAT Min',
'Fwd Seg Size Min', 'Total Fwd Packets', 'Subflow Fwd Packets']

allFeatureSet = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Fwd Packets Length Total', 'Bwd Packets Length Total', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Packet Length Min', 'Packet Length Max', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'SYN Flag Count', 'RST Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'Down/Up Ratio', 'Avg Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init Fwd Win Bytes', 'Init Bwd Win Bytes', 'Fwd Act Data Packets', 'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Protocol_0', 'Protocol_6', 'Protocol_17']

specificAttackEncoding = { 'UDP': 14, 'MSSQL': 9, 'Benign': 0, 'Portmap': 11, 'Syn': 12, 'NetBIOS': 10, 'UDPLag': 16, 'LDAP': 8, 'DrDoS_DNS': 1, 'UDP-lag': 15, 'WebDDoS': 17, 'TFTP': 13, 'DrDoS_UDP': 7, 'DrDoS_SNMP': 6, 'DrDoS_NetBIOS': 5, 'DrDoS_LDAP': 2, 'DrDoS_MSSQL': 3, 'DrDoS_NTP': 4}

genarelAttackEncoding = {'Attack': 0, 'Benign': 1}

def formatString(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

class Data(BaseModel):
    data: List[dict]
    param: int = 0

@app.get("/")
async def root():
    return {"data": "DDoS Point BE Running"}

@app.post("/predict")
async def sendDataToBE(data: Data):
    df = pd.DataFrame(data.data)

    if 'Protocol' in df.columns:
        df = pd.get_dummies(df, prefix='Protocol', drop_first=True, columns=['Protocol'])

    protocol_columns = [col for col in df.columns if col.startswith('Protocol_')]
    all_protocols = {'Protocol_17', 'Protocol_6', 'Protocol_0'}
    missing_protocols = all_protocols - set(protocol_columns)

    for missing_col in missing_protocols:
        df[missing_col] = 0

    resultArray = []
    for _, row in df.iterrows():
        p1 = [formatString(row[feature]) for feature in impFeatureSet]
        p2 = [formatString(row[feature]) for feature in allFeatureSet]
        row_df = pd.DataFrame([p2], columns=allFeatureSet)
        imp_row_df = pd.DataFrame([p1], columns=impFeatureSet)
        resultForSpecific = 'Unknown'
        resultForGeneral = 'Unknown'
        specificPrediction = modelForSpecific.predict(row_df)[0]
        generalPrediction = modelForGeneral.predict(imp_row_df)[0]

        for k, v in specificAttackEncoding.items():
            if v == int(specificPrediction):
                resultForSpecific = k
                break
            
        for k, v in genarelAttackEncoding.items():
            if v == int(generalPrediction):
                resultForGeneral = k
                break
            
        if resultForSpecific == 'Benign' and resultForGeneral == 'Attack':
            resultForSpecific = 'Unknown'
        
        resultArray.append({
            "specific": resultForSpecific,
            "general": resultForGeneral
        })

    return resultArray
