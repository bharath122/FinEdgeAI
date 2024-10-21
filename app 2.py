from flask import Flask, request, jsonify
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
pd.pandas.set_option('display.max_columns',None)
pd.pandas.set_option('display.max_rows',None)
import datetime as datetime
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import plotly.io as pio
import traceback
from flask_cors import CORS
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report,confusion_matrix,ConfusionMatrixDisplay
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor 


app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 5000 * 1024 * 1024  # Increase limit to 100MB

pio.orca.config.executable = 'C:/Users/naveen.singaravelan/AppData/Local/anaconda3/orca_app/orca.exe'

#C:\Users\naveen.singaravelan\AppData\Local\anaconda3\orca_app
pio.orca.config.save()


with open(r'CASA_Lib/training_dataMerchant City.pkl', 'rb') as file:
    Merchant_City_label_encoder = pickle.load(file)

with open(r'CASA_Lib/training_dataMerchant State.pkl', 'rb') as file:
    merchant_state_label_encoder = pickle.load(file)

with open(r'CASA_Lib/training_dataCard Type.pkl', 'rb') as file:
    merchant_state_label_encoder = pickle.load(file)

scaler1 = pickle.load(open(r'CASA_Lib/scaled_data.sav', 'rb'))

rfc=pickle.load(open(r'CASA_Lib/casa_prediction.sav','rb'))

def base_64_processing(fig):
    buffer = BytesIO()
    fig.write_image(buffer, format='png', engine="orca")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return image_base64

def base_64_processing_plotly(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_str = base64.b64encode(buf.read())
    return img_str.decode("utf-8")


def visualization(data):   
    output_dict = {}
    try:
        # Credit Card transaction Data Distribution
        labels = data['Is Fraud?'].value_counts().index
        values = data['Is Fraud?'].value_counts().values
        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
        ax.set_title("TRANSACTION DATA DISTRIBUTION")
        
        file_path = os.path.join('static', 'transaction Distribution.png')
        plt.savefig(file_path)
        transaction_distribution_base_64 = base_64_processing_plotly(fig)
        output_dict['transaction Distribution'] = transaction_distribution_base_64
        plt.close(fig)

        # Analysis of chip in card
        labels = data['Has Chip'].value_counts().index
        values = data['Has Chip'].value_counts().values
        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
        ax.set_title("ANALYSIS OF CHIP IN CARD")
        
        file_path = os.path.join('static', 'Analysis of chip in card.png')
        plt.savefig(file_path)
        transaction_distribution_base_64 = base_64_processing_plotly(fig)
        output_dict['Analysis of chip in card'] = transaction_distribution_base_64
        plt.close(fig)

        # Use Chip Vs Count
        df = data.groupby('Use Chip')['Use Chip'].agg(['count']).sort_values(by='count', ascending=False)[:10]
        df = df.reset_index()
        fig, ax = plt.subplots()
        sns.barplot(data=df, x='Use Chip', y='count', ax=ax)
        ax.set_title('ANALYSIS OF TYPES OF TRANSACTIONS')
        
        file_path = os.path.join('static', 'chip vs count.png')
        plt.savefig(file_path)
        transaction_distribution_base_64 = base_64_processing_plotly(fig)
        output_dict['chip vs count'] = transaction_distribution_base_64
        plt.close(fig)

        # Use_Chip_vs_Number_of_Transaction
        df1 = data.groupby(by=['Use Chip', 'Is Fraud?']).size().reset_index(name='No. of Transactions')
        fig, ax = plt.subplots()
        sns.barplot(data=df1, x='Use Chip', y='No. of Transactions', hue='Is Fraud?', ax=ax)
        ax.set_title('TRANS.TYPES/IS_FRAUD WISE DATA DETECTION')
        
        file_path = os.path.join('static', 'Use Chip vs Number of Transaction.png')
        plt.savefig(file_path)
        transaction_distribution_base_64 = base_64_processing_plotly(fig)
        output_dict['Use Chip vs Number of Transaction'] = transaction_distribution_base_64
        plt.close(fig)

        # MCC vs Count
        df = data.groupby('MCC')['MCC'].agg(['count']).sort_values(by='count', ascending=False)[:10]
        df = df.reset_index()
        fig, ax = plt.subplots()
        sns.barplot(data=df, x='MCC', y='count', ax=ax)
        ax.set_title('CATEGORY CODE CARD WISE FRAUD DETECTION')
        
        file_path = os.path.join('static', 'MCC vs Count.png')
        plt.savefig(file_path)
        transaction_distribution_base_64 = base_64_processing_plotly(fig)
        output_dict['MCC vs Count'] = transaction_distribution_base_64
        plt.close(fig)

        # Amount Range vs Transaction
        df1 = data.copy()
        df1.loc[df1['Amount'].between(-1000, 0), 'amount_range'] = '-1k-0k'
        df1.loc[df1['Amount'].between(0, 200), 'amount_range'] = '0-0.2k'
        df1.loc[df1['Amount'].between(200, 400), 'amount_range'] = '0.2k-0.4k'
        df1.loc[df1['Amount'].between(400, 600), 'amount_range'] = '0.4k-0.6k'
        df1.loc[df1['Amount'].between(600, 800), 'amount_range'] = '0.6k-0.8k'
        df1.loc[df1['Amount'].between(800, 1000), 'amount_range'] = '0.8k-1k'
        df1.loc[df1['Amount'].between(1000, 1200), 'amount_range'] = '1k-1.2k'
        df1.loc[df1['Amount'].between(1200, 1400), 'amount_range'] = '1.2k-1.4k'
        df1.loc[df1['Amount'].between(1400, 1600), 'amount_range'] = '1.4k-1.6k'
        df1.loc[df1['Amount'].between(1600, 1800), 'amount_range'] = '1.6k-1.8k'
        df1.loc[df1['Amount'].between(1800, 2000), 'amount_range'] = '1.8k-2k'
        df1.loc[df1['Amount'] > 2000, 'amount_range'] = '>2k'
        df = df1.groupby(by=['amount_range', 'Is Fraud?']).size().reset_index(name='No. of Transactions')
        fig, ax = plt.subplots()
        sns.barplot(data=df, x='amount_range', y='No. of Transactions', hue='Is Fraud?', ax=ax)
        ax.set_title('AMOUNT_RANGE/IS_FRAUD WISE DATA DETECTION')
        plt.xticks(rotation=45, ha='right')
        
        file_path = os.path.join('static', 'Amount Range vs Transaction.png')
        plt.savefig(file_path)
        transaction_distribution_base_64 = base_64_processing_plotly(fig)
        output_dict['Amount Range vs Transaction'] = transaction_distribution_base_64
        plt.close(fig)

        # Amount Range vs Number of Incidents
        df = df1.groupby(by=['Use Chip', 'amount_range']).size().reset_index(name='No. of Incidents')
        fig, ax = plt.subplots()
        sns.barplot(data=df, x='amount_range', y='No. of Incidents', hue='Use Chip', ax=ax)
        ax.set_title('TYPES OF TRANSACTIONS/IS_FRAUD WISE FRAUD DETECTION')
        plt.xticks(rotation=45, ha='right')


        file_path = os.path.join('static', 'AmountRange vs Number_of_Incident.png')
        plt.savefig(file_path)
        transaction_distribution_base_64 = base_64_processing_plotly(fig)
        output_dict['AmountRange vs Number_of_Incident'] = transaction_distribution_base_64
        plt.close(fig)

        response = {
            "status": "success",
            "images": output_dict
        }
        return response

    except Exception as e:
        print('--------error-----------', e)
        return 'Connection Error'

    
@app.route('/submit', methods=['POST'])
def submit():
    #try:
    data = request.json

    Card_Number=np.array([data['Card_Number']])
    FICO_Score = np.array([data['FICO_Score']])
    Yearly_Income = np.array([data['Yearly_Income']])
    Current_Age = np.array([data['Current_Age']])
    Credit_Limit = np.array([data['Credit_Limit']])
    mcc = np.array([data['mcc']])
    amount = np.array([data['amount']])
    
    use_chip_map = {'Swipe Transaction': [1], 'Chip Transaction': [2], 'Online Transaction': [3]}
    use_chip = use_chip_map.get(data['use_chip'], "Invalid input")
    print("**************----->",data['Card_Brand'])
    Card_Brand_map = {'Visa': [1], 'Amex': [2], 'Mastercard': [3], 'Discover': [4]}


    Card_Brand = Card_Brand_map.get(data['Card_Brand'], "Invalid input")

    Card_Type_map={'Debit':[0], 'Credit':[1], 'Debit (Prepaid)':[2]}
    Card_Type = Card_Type_map.get(data['Card_Type'], "Invalid input")


    
    merchant_state = merchant_state_label_encoder.transform([data['merchant_state'].upper().strip()])
    Merchant_City = Merchant_City_label_encoder.transform([data['Merchant_City'].strip()])
    
    opt_map = {'YES': [1], 'NO': [0]}
    Has_Chip = opt_map.get(data['Has_Chip'], "Invalid input")
    
    user_Latitude = np.array([data['user_Latitude']])
    user_Longitude = np.array([data['user_Longitude']])
    print("---------",amount,use_chip,Merchant_City,merchant_state,mcc,Card_Brand,Card_Type,Card_Number,Has_Chip,Credit_Limit,Current_Age,user_Latitude,user_Longitude,Yearly_Income,FICO_Score)
    print("**************88",Card_Brand)
    input_d=np.concatenate((amount,use_chip,Merchant_City,merchant_state,mcc,Card_Brand,Card_Type,Card_Number,Has_Chip,Credit_Limit,Current_Age,user_Latitude,user_Longitude,Yearly_Income,FICO_Score))
    

    input1 = input_d.astype(np.float64)
    input_data = input1.reshape(1, -1)

    X_test= scaler1.transform(input_data)
    x=rfc.predict(X_test)
    
    if x[0]==0:
        response='The transaction is not a fraudlent transaction'
        return jsonify(response)
    else:
        response='Sorry! The transaction may be fraudlent transaction'
        return jsonify(response)
    # except Exception as e:
    #     return ""

class TimeoutException(Exception):
    pass

def datapreprocessing(dd):

    dd['Use Chip']=dd['Use Chip'].map({'Swipe Transaction':1, 'Chip Transaction':2, 'Online Transaction':3})
    dd['Card Brand']=dd['Card Brand'].map({'Visa':1, 'Amex':2, 'Mastercard':3, 'Discover':4})
    try:
        dd= dd[['Amount','Use Chip','Merchant City','Merchant State','MCC','Is Fraud?','Card Brand','Card Number','Has Chip','Credit Limit','Current Age','user_Latitude','user_Longitude','Yearly Income - Person','FICO Score']]
    except:
        dd= dd[['Amount','Use Chip','Merchant City','Merchant State','MCC','Card Brand','Card Number','Has Chip','Credit Limit','Current Age','user_Latitude','user_Longitude','Yearly Income - Person','FICO Score']]
    
    columns_to_convert_in_string_to_int = ['Merchant City','Merchant State']
    for col in columns_to_convert_in_string_to_int:
        if col == 'Merchant State':
            dd [col] = merchant_state_label_encoder.transform(dd [col])
        else:
            dd [col] = Merchant_City_label_encoder.transform(dd [col])
    
    return dd


@app.route('/data_preprocessing', methods=['POST'])
def data_preprocessing():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    print("-------------------")
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:

        dd=pd.DataFrame()
        chunk_size = 100000
        for chunk in pd.read_csv(file, chunksize=chunk_size,encoding='ISO_8859_1'):
            dd=pd.concat([dd,chunk])
            break

        description = dd.describe().to_json()

        dd=datapreprocessing(dd)
        
        correlation_matrix=dd.corr(method = 'pearson')
        correlation_matrix = correlation_matrix.drop(columns=['Card Brand'], index=['Card Brand'])
        correlation_matrix_json=correlation_matrix.to_dict()

        numerical_features = [feature for feature in dd.columns if dd[feature].dtypes != 'O']
        vif1=dd[dd[numerical_features].columns[dd[numerical_features].columns!='Is Fraud?']]
        vif_data = pd.DataFrame()
        vif_data["feature"] = vif1.columns
        
        # calculating VIF for each feature

        vif_data["VIF"] = [variance_inflation_factor(vif1.values, i)
                                for i in range(len(vif1.columns))]
        
        response={'description':description,'VIF':vif_data.to_json(),'correlation_matrix':correlation_matrix_json}
        return jsonify(response)
    except Exception as e:
        print('errror',e)
        response={'description':'fail'}
        traceback.print_exc()  # This will print the traceback, including the line number
        return jsonify(response)

 
@app.route('/EDA_Process', methods=['POST'])
def eda_process():
    try:
        file = request.files['file']
        chunk_size = 1000000  # Adjust based on your memory constraints
        dd = pd.DataFrame()
        for chunk in pd.read_csv(file, chunksize=chunk_size, encoding='ISO_8859_1'):
            dd = pd.concat([dd, chunk])
            break
        
        data = dd.copy()
        data['Date'] = pd.to_datetime(data['Date'])
        data['year'] = data['Date'].dt.year
        data['Card Number'] = data['Card Number'].astype(np.int64)
        data['year'] = data['year'].astype('str')
        data['MCC'] = data['MCC'].astype('str')
        data['Is Fraud?'] = data['Is Fraud?'].map({0: 'Non_Fraudulent', 1: 'Fraudulent'})
        data['Has Chip'] = data['Has Chip'].map({0: 'NO', 1: 'Yes'})
        response=visualization(data)
        print('*****************************type oof ',type(response))
        return jsonify(response)
    except TimeoutException as e:
        return jsonify({"Connection Error": str(e)}), 504  # HTTP 504 Gateway Timeout
    except Exception as e:
        return jsonify({"Connection Error": str(e)}), 500


def evaluate_model(model, X_test, y_test,file_name):
    print(X_test)
    print(y_test.head())
    array = y_test.values
    
    print(type(array))
    y_pred = model.predict(X_test)
    f1 = f1_score(array, y_pred)

    Target_Response = {
        'Fraud': 1,
        'Legitimate': 0
    }

    # Compute confusion matrix
    conf_matrix = confusion_matrix(array, y_pred, labels=list(Target_Response.values()))
    conf_matrix_df = pd.DataFrame(conf_matrix, index=list(Target_Response.keys()), columns=list(Target_Response.keys()))
    print(conf_matrix_df)
    
    color_maps = [
    'Blues', 'Greens', 'Oranges', 'Reds', 'Purples', 'YlGnBu', 'BuPu', 
    'GnBu', 'PuBu', 'YlOrRd', 'PuRd', 'RdPu', 'BuGn', 'YlGn', 'OrRd', 
    'PuBuGn', 'YlOrBr', 'RdYlBu', 'Spectral', 'coolwarm', 'viridis', 
    'plasma', 'inferno', 'magma', 'cividis'
    ]

    color=random.choice(color_maps)

    fig1 = plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap=color,cbar=True, linewidths=1, linecolor='black')

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f'plots/{file_name}')
    plt.close() 

    plots=[]

    #------------------------
    fig, ax = plt.subplots(figsize=(7, 6))  # Adjust figsize as needed

    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=Target_Response)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'Confusion Matrix')

    # Rotate x-axis labels and set y-axis labels
    ax.set_xticklabels(disp.display_labels, rotation=45, ha='right')
    ax.set_yticklabels(disp.display_labels)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f'plots/{file_name}')  # Save the figure as a PNG file
    plt.close()
    #------------------------

    with open(f'plots/{file_name}', "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode('utf-8')
        plots.append(b64_string)

    return plots,f1


@app.route('/model_validation', methods=['POST'])
def model_validation():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.csv'):
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file)
        
        #---------------------Sample 1------------------------#
        subset1 = df.sample(frac=0.5, random_state=1)
        #print("----------subset1----------->",subset1.shape)
        subset1_x=subset1[subset1.columns[subset1.columns!='Is Fraud?']]
        y=subset1['Is Fraud?']
        #print("*****************************",y.shape)
        subset1_x = datapreprocessing(subset1_x)
        x=['Use Chip','MCC','Card Brand','Card Number','Has Chip','Credit Limit','Current Age','Yearly Income - Person','FICO Score']
        for i in x:
            subset1_x [i] = subset1_x [i].astype('float64')

        subset1_x = subset1_x[['Amount','Use Chip','Merchant City','Merchant State', 'MCC','Card Brand','Card Type', 'Card Number', 'Has Chip','Credit Limit', 
                       'Current Age','user_Latitude', 'user_Longitude',
                       'Yearly Income - Person','FICO Score']]
        scaler1_data = scaler1.transform(subset1_x)
        #print(scaler1_data)
        #print(type(scaler1_data))
        file_name = "sample_1.png"
        conf_matrix_subset1_base64,f1_score_sample_1 = evaluate_model(rfc,scaler1_data,y,file_name)

        #---------------------Sample 2------------------------#
        subset1 = df.sample(frac=0.4, random_state=1)
        #print("----------subset1----------->",subset1.shape)
        subset1_x=subset1[subset1.columns[subset1.columns!='Is Fraud?']]
        y=subset1['Is Fraud?']
        subset1_x = datapreprocessing(subset1_x)
        x=['Use Chip','MCC','Card Brand','Card Number','Has Chip','Credit Limit','Current Age','Yearly Income - Person','FICO Score']
        for i in x:
            subset1_x [i] = subset1_x [i].astype('float64')

        subset1_x = subset1_x[['Amount','Use Chip','Merchant City','Merchant State', 'MCC','Card Brand', 'Card Type', 'Card Number', 'Has Chip','Credit Limit', 
                       'Current Age','user_Latitude', 'user_Longitude',
                       'Yearly Income - Person','FICO Score']]
        scaler1_data = scaler1.transform(subset1_x)
        file_name = "sample_2.png"
        conf_matrix_subset2_base64,f1_score_sample_2 = evaluate_model(rfc,scaler1_data,y,file_name)

        #---------------------Sample 3------------------------#
        subset1 = df.sample(frac=0.3, random_state=1)
        #print("----------subset1----------->",subset1.shape)
        subset1_x=subset1[subset1.columns[subset1.columns!='Is Fraud?']]
        y=subset1['Is Fraud?']
        subset1_x = datapreprocessing(subset1_x)
        x=['Use Chip','MCC','Card Brand','Card Number','Has Chip','Credit Limit','Current Age','Yearly Income - Person','FICO Score']
        for i in x:
            subset1_x [i] = subset1_x [i].astype('float64')

        subset1_x = subset1_x[['Amount','Use Chip','Merchant City','Merchant State', 'MCC','Card Brand', 'Card Type', 'Card Number', 'Has Chip','Credit Limit', 
                       'Current Age','user_Latitude', 'user_Longitude',
                       'Yearly Income - Person','FICO Score']]
        scaler1_data = scaler1.transform(subset1_x)
        file_name = "sample_3.png"
        conf_matrix_subset3_base64,f1_score_sample_3 = evaluate_model(rfc,scaler1_data,y,file_name)

        return jsonify({
            'confusion matrix 1': conf_matrix_subset1_base64,
            'f1_score1':f1_score_sample_1,
            'confusion matrix 2': conf_matrix_subset2_base64,
            'f1_score2':f1_score_sample_2,
            'confusion matrix 3': conf_matrix_subset3_base64,
            'f1_score3':f1_score_sample_3,
        }), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5004,debug=True)
