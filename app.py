from dotenv import load_dotenv
from flask import Flask,render_template,request,redirect,url_for
from flask_cors import CORS
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import base64
from modulation import *
from binascii import unhexlify
import logging
import json

import firebase_admin
from firebase_admin import credentials, db

logger = logging.getLogger('waitress') # A logger is used to record events and messages during the program's execution
logger.setLevel(logging.INFO) # This line sets the logging level of the logger to INFO. The logging level determines the importance of messages to be logged

matplotlib.use('WebAgg')# WebAgg is a backend that's suitable for rendering plots in web applications.

load_dotenv()#  This line calls a function to load environment variables from a file named .env in the current directory.

app=Flask(__name__,static_url_path='/static') # creates a Flask web application
CORS(app)


@app.template_filter('decode_hex') # function to decode hex
def decode_hex(s):
    return unhexlify(s)


@app.template_filter('b64encode') # Base64 is a binary-to-text encoding scheme
def b64encode(s):
    return base64.b64encode(s).decode('utf-8') # first we encode in base64 format and then decode it in unistring format so that it can be displayed in text fields,email etc

@app.route('/',methods=['GET'])
def home():
    f = open('./data.json')# opens data.json
    data = json.load(f)["Home"] # loads data named HOME from data.json
    return render_template('home.html',data=data) # This renders the home.html file and passes data to home.html

@app.route("/subbb", methods=["POST"])
def submit_form():
    if request.method == "POST":
        name = request.form["name"]
        subject = request.form["subject"]

        # Push form data to the database
        ref = db.reference("form_submissions")
        new_data_ref = ref.push()
        new_data_ref.set({"name": name, "subject": subject})

        return redirect("/")

@app.route('/references',methods=['GET'])
def references():
    return render_template('references.html')

@app.route('/theory/<modulation_type>',methods=["GET"])
def theory(modulation_type): # this function takes modulation type as input renders appropriate pages
    return render_template(f'theory/{modulation_type}.html')




# ---------- Analog Modulation -------------------
@app.route('/AM',methods=['GET'])
def AM_page():
    f = open('./data.json')
    data = json.load(f)["AM"] # loads AM form data.json
    return render_template('Analog_Modulation.html',data=data) # renders analog_modulation.html and passed data to it

@app.route('/AM/<am_type>',methods=['GET','POST']) # get and post methods for specific am type
def Amplitutde_Modulation(am_type):  
    
    title = {"MAIN":"Amplitutde Modulation","SSB":"SSB Modulation","DSBSC":"DSB Modulation","QAM":"QAM Modulation"} # title depends on the am type
    plots = [] # This is an empty list that will hold the generated plots for the AM scenario
    x_message = [] # for holding message signal data for plotting
    x_carrier = [] # for holding carrier signal data for plotting
    errMsg = "" # initializes error message
    if (request.method=='POST'): # if the method is post
        content = request.form  # the form contents are stored in content
        fm=int(content['fm'])
        fc=int(content['fc'])
        Am=int(content['Am'])     # all the contents are stored in seperate variables
        Ac=int(content['Ac'])
        message_signal = str(content['message_signal']) # stores message_signale with the message_signal string passed in the form content
        if(Am>Ac or fm>fc): # conditions which are not possible
            errMsg = "Given graph is Not possible as Fc <Fm or Ac<Am."

        inputs = {"Am":Am,"Ac":Ac,"fm":fm,"fc":fc,"message_signal":message_signal} # stores the inputs in seperate variables inside the 'inputs'
        
        if am_type == "MAIN":
            plots = AM_main_graph(inputs)
        elif am_type == "SSB":
            plots = AM_ssb_modulation(inputs)     # plots differnet graphs according to the input
        elif am_type == "DSBSC":
            plots = AM_double_sideband_modulation(inputs)
        elif am_type == "QAM":
            message_signal_2 = request.form['message_signal_2']
            inputs["message_signal_2"] = message_signal_2
            plots = AM_QAM(inputs) 
        # return render_template('AM_graphs.html',am_type=am_type.upper(),title=title[am_type],plots = plots,inputs=inputs,errMsg=errMsg)
    return render_template('AM_graphs.html',am_type=am_type.upper(),title=title[am_type],plots = plots,errMsg=errMsg)

@app.route('/FM/<index>',methods=['GET','POST']) # this is the get and post method for fm type
def FM(index):
    images = [] # for storing images
    index = int(index) # index can be 1 or to for fm or pm
    inputs = {} # for storing input values
    errMsg = ""
    title={1:"Frequency modulation",2:"Phase modulation"} # if index is 1 fm or if index is 2 pm
    if request.method == 'POST':
        fm=int (request.form['fm'])
        fc=int (request.form['fc'])
        Am=int (request.form['Am'])
        Ac=int (request.form['Ac'])
        message_signal = str(request.form['message_signal']) # gets message signal value
        K = float(request.form['K'])
        if(fc<fm or Ac<Am):
            errMsg = "Given graph is Not possible as Fc <Fm or Ac<Am." 

        inputs = {"Am":Am,"Ac":Ac,"fm":fm,"fc":fc,"message_signal":message_signal,"K":K}
        # = np.linspace(-200,200,10000)  #domain for the modulated_wave
        #s = [1 for i in x] # looks like its an impulse
        if(index==1):   # if index is 1 then fm is called and result is stored in images
            images = FM_MAIN(inputs)           
        elif(index==2):
            images = PHASE_MAIN(inputs) 
      
    return render_template('fm_graphs.html',title=title[index],index=index,plots=images,inputs=inputs,errMsg=errMsg) #renders appropriate images in fm_graphs.html

# ---------- End of Analog Modulation ------------


# ---------- Digital Modulation ---------------------

@app.route('/DM',methods=['GET'])
def DM_page():
    f = open('./data.json')
    data = json.load(f)["DM"]
    return render_template('Digital_Modulation.html',data=data)


@app.route('/DM/<dmtype>', methods=['GET','POST'])  # get and post for dm
def DigitalModulation(dmtype):
    errMsg=''
    title = {"BPSK":"BPSK Modulation","BFSK":"BFSK Modulation","BASK":"BASK Modulation","QPSK":"QPSK Modulation","DPSK":"DPSK Modulation"}
    plots = []
    inputs = {}

    if (request.method=='POST'):
        try:
            Tb=float (request.form['Tb'])

            if(dmtype=='BFSK'):
                Ac=int (request.form['Ac'])
                fc1=int (request.form['fc1'])
                fc2=int (request.form['fc2'])
                inputs['fc1'] = fc1
                inputs['fc2'] = fc2
                inputs['Ac'] = Ac

            elif(dmtype=='BASK'):
                fc=int (request.form['fc'])
                Ac1=int (request.form['Ac1'])
                Ac2=int (request.form['Ac2'])
                inputs['Ac1'] = Ac1
                inputs['Ac2'] = Ac2
                inputs['fc'] = fc

            else:    
                fc=int (request.form['fc'])
                Ac=int (request.form['Ac'])
                inputs['fc'] = fc
                inputs['Ac'] = Ac


            binaryInput = str(request.form['inputBinarySeq'])
            inputs = {"Tb":Tb,"binaryInput":binaryInput}
            #   fc2=1


            # Change Binary string to array
            inputBinarySeq = np.array(list(binaryInput), dtype=int) # converts binary input into array


            if dmtype.upper() == 'BASK':
                plots = BASK(Tb, fc,Ac1,Ac2, inputBinarySeq)
            elif dmtype.upper() == 'BFSK':
                plots = BFSK(Tb,Ac, fc1, fc2, inputBinarySeq)

            elif dmtype.upper() == 'BPSK':
                plots = BPSK(Tb,Ac, fc, inputBinarySeq)

            elif dmtype.upper() == 'QPSK':
                plots = QPSK(Tb,Ac, fc, inputBinarySeq)

            elif dmtype.upper() == 'DPSK':
                plots = DPSK(Tb,Ac, fc, inputBinarySeq)

        except Exception:
            errMsg = 'Unknown error occured. Make sure that the input values are correct'   

    return render_template('DM_graphs.html',dmtype=dmtype.upper(),title=title[dmtype], plots=plots,inputs=inputs, errMsg=errMsg)



@app.route('/DM3/<dmtype>', methods=['GET','POST']) # get and post for dpsk
def DPSK_Modulation(dmtype):
    title = {"DPSK":"DPSK Modulation"}
    plots = []
    inputs = {}
    if (request.method=='POST'):
        Ac= int (request.form['Ac'])
        fc= int (request.form['fc'])
        inputs = {'fm':fm,'Am':Am,'phi_m':phi_m,'phi_c':phi_c}
        if dmtype.upper() == 'DPSK':
            plots = DPSK(fm, Am, phi_m, fc, Ac, phi_c)
    return render_template('DPSK_graphs.html',dmtype=dmtype.upper(),title=title[dmtype], plots=plots,inputs=inputs)    


# ------------ End of Digital Modulation -------------

# ---------- Pulse Modulation ---------------------

@app.route('/PM',methods=['GET'])
def PM_page():
    f = open('./data.json')
    data = json.load(f)["PM"]
    return render_template('Pulse_Modulation.html',data=data)


@app.route('/PM/<pmtype>', methods=['GET','POST'])
def PulseModulation(pmtype):
    encoded=''
    #quantized_value = ''
    title = {"SAMPLING":"Sampling",
             "QUANTIZATION":"Quantization",
             "PAM":"Pulse Amplitude Modulation",
             "PPM":"Pulse Phase Modulation",
             "PCM":"Pulse Code Modulation",
             "PWM":"Pulse Width Modulation"
             }
    plots = []
    inputs = []
    print(request.form)
    if (request.method=='POST'): 
        fm = int (request.form['fm'])
        am = int (request.form['am'])
        message_type = str(request.form["message_signal"])

        inputs = [fm,am,message_type]

      # Change Binary string to array
        print(pmtype)
        if pmtype.upper() == 'PPM':
            inputs.append(int(request.form['fs']))
            ppm_ratio = float(request.form['ppm_ratio'])        
            inputs.append(ppm_ratio)
            plots = PPM(inputs)
        elif pmtype.upper() == 'PAM':
          inputs.append(int(request.form['fs']))
          plots = PAM(inputs)
        elif pmtype.upper() == 'PWM':
          inputs.append(int(request.form['fs']))
          plots = PWM(inputs)
        elif pmtype.upper() == 'PCM':
          inputs.append(int(request.form['ql']))
          inputs.append(int(request.form['nb']))
          a,b,c,encoded= PCM(inputs)
          plots = [a,b,c]
        elif pmtype.upper() == 'SAMPLING':
          inputs.append(int(request.form['fs']))
          plots = SAMPLING(inputs)
        elif pmtype.upper() == 'QUANTIZATION':
          inputs.append(int(request.form['ql']))
          plots = QUANTIZATION(inputs)


    if(pmtype != "PCM"):
        return render_template('PM_graphs.html',pmtype=pmtype.upper(),title=title[pmtype], plots=plots)
    else:
        return render_template('PM_graphs.html',pmtype=pmtype.upper(),title=title[pmtype], plots=plots, encoded=encoded)
 

def create_app():
    # PORT = int(os.environ.get("PORT",8000))
    port = int(os.environ.get("PORT", 5000))  # Use 5000 as default
    app.run(debug=True, host='0.0.0.0', port=port)
    # app.run(debug=False,host='0.0.0.0')

if __name__ == "__main__":
    create_app()