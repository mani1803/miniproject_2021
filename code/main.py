# Elaine Assistent source code!!!!!!

from gtts import gTTS
import pyttsx3
import speech_recognition as sr
import time
import pygame
import random
import playsound
import datetime
import os
from bs4 import BeautifulSoup
import bs4 
import requests
import re
import webbrowser
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
import subprocess
import pyttsx3
import json
import pafy
import pyjokes
from num2words import num2words
from python_utils import *
from urllib.request import urlopen
import urllib.request 
from urllib.parse import * 
import datetime
from datetime import date
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from googleapiclient.discovery import build
import smtplib
import pyaudio
import platform
import sys
import email
import imaplib
import pyglet
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
import ecapture as ec
import operator
import wolframalpha 
import cv2
import psutil
import numpy as np
import yagmail
from time import strftime
import pyautogui
import ctypes
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QMainWindow, QPushButton, QApplication, QWidget, QLabel
from PyQt5.uic import loadUiType
from elaineUi import Ui_MainWindow



def talk(audio):
    engine = pyttsx3.init() 
    voices = engine.getProperty('voices') 
    engine.setProperty('voice', voices[1].id)  
    engine.say(audio)   
    engine.runAndWait() 

def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour>= 0 and hour<12:
        talk("Good Morning !")
  
    elif hour>= 12 and hour<18:
        talk("Good Afternoon !")   
  
    else:
        talk("Good Evening!")  
  
    assname =("Elaine 1 point o")
    talk("I am your Assistant")
    talk(assname)
    talk('How can I help you?')  


def myCommand():    

        r = sr.Recognizer()

        with sr.Microphone() as source:
            print('Elaine is Listening...')
            r.pause_threshold = 0.8
            r.adjust_for_ambient_noise(source, duration=0.7)       
            audio = r.listen(source, timeout=8)

        try:
            command = r.recognize_google(audio).lower()
            print('You said: ' + command + '\n')

        
        except sr.UnknownValueError:
        
            talk("Your last command couldn\'t be heard")
            command = myCommand()
        return command

def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers

def start_webcam():
    cap = cv2.VideoCapture(0)
    return cap

def display_blob(blob):
    for b in blob:
        for n, imgb in enumerate(b):
            cv2.imshow(str(n), imgb)
           

def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids

def draw_labels(boxes, confs, colors, class_ids, classes, img):
    engine = pyttsx3.init()
    r = sr.Recognizer()
    detected_obj={}
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
            if label in detected_obj:
                pass
            else:
                detected_obj[label]= 1
                voices = engine.getProperty('voices') 
                engine.setProperty('voice', voices[1].id)  
                engine.say("the detected image is")
                engine.say(label)
                engine.runAndWait()
       
                
    cv2.imshow("Image", img)
    

    
def load_image(img_path):
    img = cv2.imread(img_path)
    #img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    return img, height, width, channels

def image_detect(img_path):
    model, classes, colors, output_layers = load_yolo()
    image, height, width, channels = load_image(img_path)
    blob, outputs = detect_objects(image, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    draw_labels(boxes, confs, colors, class_ids, classes, image)
    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break
            
            
def webcam_detect():
    r = sr.Recognizer()
    model, classes, colors, output_layers = load_yolo()
    cap = start_webcam()
    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, colors, class_ids, classes, frame)
        
        key = cv2.waitKey(1)
        if key == 27:
            break
        with sr.Microphone() as source:
            print('listening')
            talk("shall i continue to detect the images")
            ch = myCommand()
            if 'stop' in ch or "end" in ch:                     
                cv2.destroyAllWindows()           
                return
            else:                
                continue         
    
            
    cap.release()

def my_location():
    ip_add = requests.get('https://api.ipify.org').text
    url = 'https://get.geojs.io/v1/ip/geo/' + ip_add + '.json'
    geo_requests = requests.get(url)
    geo_data = geo_requests.json()
    city = geo_data['city']
    state = geo_data['region']
    country = geo_data['country']

    return city, state,country



def computational_intelligence(question):
    try:
        client = wolframalpha.Client('37XXAW-U5Y43UTLQR')
        answer = client.query(question)
        answer = next(answer.results).text
        print(answer)
        return answer
    except:
        talk("Sorry I couldn't fetch your question's answer. Please try again ")
        return None


def Elaine(command):
        errors=[
            "I don't know what you mean",        
            "Excuse me?",
            "Can you repeat it please?",
        ]

        

        if 'hello' in command:
            talk('Hello! I am Elaine . How can I help you?')

        if 'start image detection' in command:
            webcam_detect()
            talk("do you aything else,  want me too do?")
        
        if 'open google and search' in command:
            reg_ex = re.search('open google and search (.*)', command)
            search_for = command.split("search",1)[1] 
            print(search_for)
            url = 'https://www.google.com/'
            if reg_ex:
                subgoogle = reg_ex.group(1)
                url = url + 'r/' + subgoogle
            talk('Okay!')
            driver = webdriver.Chrome(ChromeDriverManager().install())
            driver.get('http://www.google.com')
            search = driver.find_element_by_name('q')
            search.send_keys(str(search_for))
            search.send_keys(Keys.RETURN) 
            url = 'https://www.google.com/search?q='+search_for
            
            api_key="AIzaSyB7Whiyh69g80Sv6UqU95fAJrCDl7b6EFk"
            resource=build("customsearch","v1",developerKey=api_key).cse()
            result = resource.list(q=search_for, cx='7052e161dff01cf72').execute()
            print(result['items'][0]['link'])
            m_url=result['items'][0]['link']
            
            response = requests.get(m_url)
            if response is not None:
                    html = bs4.BeautifulSoup(response.text, 'html.parser')
                    #title = html.select("#firstHeading")[0].text
                    paragraphs = html.select("p")
                    for para in paragraphs:
                        print (para.text)
                    intro = '\n'.join([ para.text for para in paragraphs[0:3]])
                    print (intro)
                    mp3name = 'speech1.mp3'
                    language = 'en'
                    myobj = gTTS(text=intro, lang=language, slow=False)   
                    myobj.save(mp3name)
                    playsound.playsound(mp3name,True)  
            driver.close()  
            os.remove(mp3name) 
            talk("i hope you got it. what else you want me to do ..Im all ears to listen to you")       
            return         
            

        elif 'wikipedia' in command:
            reg_ex = re.search('wikipedia (.+)', command)
            if reg_ex: 
                query = command.split("wikipedia",1)[1] 
                response = requests.get("https://en.wikipedia.org/wiki/" + query)
                if response is not None:
                    html = bs4.BeautifulSoup(response.text, 'html.parser')
                    title = html.select("#firstHeading")[0].text
                    paragraphs = html.select("p")
                    for para in paragraphs:
                        print (para.text)
                    intro = '\n'.join([ para.text for para in paragraphs[0:3]])
                    print (intro)
                    mp3name = 'speech.mp3'
                    language = 'en'
                    myobj = gTTS(text=intro, lang=language, slow=False)   
                    myobj.save(mp3name)
                    playsound.playsound(mp3name,True)
                    os.remove(mp3name)
                    talk("you got the details right? what else can be helpful to you?")
                    return
                
        elif 'youtube' in command:
            talk('Ok!') 
            search_for = command[22:]   
            api="AIzaSyBARoYHUsnYnt4FwtmHUCRKGRusjGNrCeI"    

            api_service_name = "youtube"
            api_version = "v3"
            youtube = build(api_service_name, api_version, developerKey=api)
            

            request = youtube.search().list(
                part="id",
                maxResults=10,
                q=search_for
                
            )
            response = request.execute()
            ids=response['items'][2]['id']['videoId']
            print(response['items'][2]['id']['videoId'])
            url="https://www.youtube.com/watch?v="+ids
            print(url)
            video=pafy.new(url)
            t=video.length   
            subprocess.Popen(["chrome",url])        
            time.sleep(t)
            subprocess.call("taskkill /IM chrome.exe")    
            talk("i hope you enjoyed the video, do you have anything else ")
            return
            

        elif "weather" in command:
            user_api = "f5d2914b2e9f620ce79c323dd4d525d1"
            talk("the city name")
            location=myCommand()

            complete_api_link = "https://api.openweathermap.org/data/2.5/weather?q="+location+"&appid="+user_api
            api_link = requests.get(complete_api_link)
            api_data = api_link.json()

            #create variables to store and display data
            temp_city = ((api_data['main']['temp']) - 273.15)
            weather_desc = api_data['weather'][0]['description']
            hmdt = api_data['main']['humidity']
            wind_spd = api_data['wind']['speed']
            date_time = datetime.datetime.now().strftime("%d %b %Y | %I:%M:%S %p")

            print ("-------------------------------------------------------------")
            print ("Weather Stats for - {}  || {}".format(location.upper(), date_time))
            print ("-------------------------------------------------------------")

            print ("Current temperature is: {:.2f} deg C".format(temp_city))
            print ("Current weather desc  :",weather_desc)
            print ("Current Humidity      :",hmdt, '%')
            print ("Current wind speed    :",wind_spd ,'kmph')
            talk("Currently, in {}, the temperature is  {:.2f} degree celcius ".format(location, temp_city, ))
            talk(" with {}".format(weather_desc))
            talk("with {} percentage humudity".format(hmdt))
            talk("and {} kilometer per hour wind speed".format(wind_spd))
            talk("i truly hope the weather details of you city is accurate. anything else you want me to check?")
            return

        elif 'locate to mail' in command or 'locate mail' in command or 'mail' in command:
            login = os.getlogin
            print ("You are logged In from : "+login())
            talk("You are logged In from : "+login())
            print ("Do you want to compose mail or check inbox?")     
            talk("Do you want to compose mail or check inbox?")
            choice = myCommand()
            if "compose mail" in choice:
                recognizer=sr.Recognizer()
                with sr.Microphone() as source:
                    print('Clearing noise...')
                    recognizer.adjust_for_ambient_noise(source,duration=1)
                    print("Please tell the Name of receiver mail ID") 
                    talk("Please tell the Name of receiver mail ID")
                    r_id=recognizer.listen(source)
                    time.sleep(2)
                    talk("What is the subject")
                    sub=myCommand()
                    talk("what's your message?")
                    print('listening the message....')
                    recordedaudio=recognizer.listen(source)
                print("ok!")
                talk("okay!")
                try:
                    print('printing the message....')
                    text=recognizer.recognize_google(recordedaudio,language='en-US')

                    print('Your message:{}'.format(text))
                except Exception as ex:
                    print(ex)
                try:
                    em=recognizer.recognize_google(r_id,language='en-US')
                    em=em+'@gmail.com'
                    em=em.replace(" ","")
                    em=em.lower()

                    print('Receiver mail id:{}'.format(em))
                except Exception as ex:
                    print(ex)
                
                r=em
                msg='Subject: {}\n\n{}'.format(sub,text)  
                mail = smtplib.SMTP('smtp.gmail.com',587)                                               #host and port area
                mail.ehlo()                                                                             #Hostname to send for this command defaults to the FQDN of the local host.
                mail.starttls()     
                try:                                                                    #security connection
                    mail.login("assistelaine@gmail.com",'abdimaro')
                    mail.sendmail('assistelaine@gmail.com',r,msg)                                                 #send section
                    print ("Congrates! Your mail has been send. ")
                    talk("Congrates! Your mail has been send. ")
                    mail.close() 
                except Exception as e:
                    print(e)
                    talk("I am not able to send this email")
 
                time.sleep(2)
                talk("may i help you with any other things?")
                return
            
            elif "inbox" in choice:
                mail = imaplib.IMAP4_SSL('imap.gmail.com',993) #this is host and port area.... ssl security
                unm = ('assistelaine@gmail.com')  #username
                psw = ('abdimaro')  #password
                mail.login(unm,psw)  #login
                stat, total = mail.select('Inbox') 
                total_mail=int(total[0])     
                 #total number of mails in inbox
                print ("Number of mails in your inbox :"+str(total))
                talk("Total mails are {}:".format(total_mail))                        
               
                _, search_data=mail.search(None,'UNSEEN')
                n=0
                print(search_data)
                for i in range(len(search_data)):
                    n=n+1          
                print("Your Unseen mail :{}".format(n))
                talk("Your Unseen mail :{}".format(n))
                
                for num in search_data[0].split():                    
                    _, data=mail.fetch(num, '(RFC822)')
                    #print(data[0])
                    _ ,b=data[0]
                    email_message=email.message_from_bytes(b)
                    
                    for header in['subject', 'to' , 'from', 'date']:
                        print("{}: {}".format(header,email_message[header]))
                    ts = gTTS(text="From: "+email_message['from']+"Date :"+email_message['date']+" And Your subject: "+str(email_message['subject']), lang='en')
                    tsname=("mail.mp3")
                    ts.save(tsname)
                    music = pyglet.media.load(tsname, streaming = False)
                    music.play()
                    time.sleep(music.duration)
                    os.remove(tsname)
                    for part in email_message.walk():
                        if part.get_content_type() == 'text/plain':
                            body=part.get_payload(decode=True)
                            print(body.decode())

                            ts = gTTS(text="Body: "+str(body.decode()), lang='en')
                            tsname=("body.mp3")
                            ts.save(tsname)
                            music = pyglet.media.load(tsname, streaming = False)
                            music.play()
                            time.sleep(music.duration)
                            os.remove(tsname)

                talk("i think you got your details of your inbox")
                talk("what further you want me to do")
                return

        elif 'location' in command or 'directions' in command:
            api="AIzaSyC0zYuJireg_8phiD2l75kipPSqpM7EvqQ"
            gmaps= googlemaps.Client(key=api)
            
            myloc = geocoder.ip('me')
            print(myloc)
            print(myloc.latlng)       
            talk('Whats your destination?')        
            #address = myCommand()
            #print(address)
            #talk(address)        
            talk("is your destination right?")
            #yn=myCommand()
            yn='yes'
            if 'yes' in yn or 'correct' in yn or 'yeah' in yn:
                #geocode_result = gmaps.geocode(address)
                talk("what mode of travelling r u in? bicycling or transit or walking or driving")
                #mode=myCommand()
                #talk("your mode is",mode)
                start = "Bridgewater, Sa, Australia"
                finish = "Stirling, SA, Australia"

                
                url="https://maps.googleapis.com/maps/api/directions/json?origin=start&destination=finish&mode=transit&region=location_region&key=AIzaSyC0zYuJireg_8phiD2l75kipPSqpM7EvqQ"
        
                with urllib.request.urlopen(url) as ur:
                    result = json.load(ur)
                print(result)

                for i in range (0, len (result['routes'][0]['legs'][0]['steps'])):
                    j = result['routes'][0]['legs'][0]['steps'][i]['html_instructions'] 
                    print(j)               
                    talk(j) 
            return

        elif 'how are you' in command:
                talk("I am fine, Thank you")
                talk("How are you?")

    
        elif 'fine' in command or "good" in command:
                talk("It's good to know that your fine")

        elif 'thank you' in command or 'good to know' in command:
            talk("its fine, check out me for more exciting things")

        elif 'time' in command:
                strTime = datetime.datetime.now().strftime("%m-%d-%Y %H:%M%p")    
                talk(f"the time is {strTime}")
        elif 'day' in command:
            now = datetime.datetime.now()
            today=now.strftime("%A")
            talk(f'today is{today}')  

        elif "who made you" in command or "who created you" in command: 
                talk("I have been created by the team VisionWalk.")

        elif 'joke' in command or 'jokes' in command:
                talk(pyjokes.get_joke(language="en", category="neutral"))

        elif "who i am" in command:
                talk("If you talk then definitely your human.")
    
        elif "why you came to world" in command:
                talk("Thanks to VISIONWALK. further It's a secret")

        elif 'is love' in command:
                talk("It is 7th sense that destroy all other senses")
    
        elif "who are you" in command:
                talk("I am Elaine, your virtual assistant created by caring team rooba, dineshkanna, abinaya and manikandan to assist visually impaired people in everyday life.")
    
        elif 'reason for you' in command:
                talk("I was created as a Minor project by the team VisionWalk ")

        elif "calculate" in command:
            print("Say what you want to calculate, example: 3 plus 3")
            talk("Say what you want to calculate, example: 3 plus 3")
            talk("if u want tocalculate power of something or cube or square say like 4 power 2")
            print("if u want tocalculate power of something or cube or square say like 4 power 2")

            my_string = myCommand()   
            if 'power'in my_string:
                a,b,c = my_string.split()
                a= int(a)
                c=int(c)
                answer = pow(a,c)
            else:    
                    app_id="37XXAW-U5Y43UTLQR"
                    client = wolframalpha.Client(app_id)
                    res = client.query(my_string)
                    answer = next(res.results).text
            print('answer :', answer)
            talk("The answer is ")
            talk(answer)  
            talk("may i help you with something else")       
            return 

        elif "write a note" in command or 'take notes' in  command or 'take a note' in command:
                talk("What should i write, ")
                content = myCommand()
                path=r"C:\Users\Rooba\mini-project\my notes"
                strTime = datetime.datetime.now().strftime("%m-%d-%Y %H:%M:%S%p")
                note = strTime.replace(" ","")
                note = note.replace(":","_")
                note = note.replace("-","_")
                notes = f"notes{note}.txt"
                file = open(os.path.join(path,notes), 'w')
                talk("Mam, Should i include date and time")
                snfm = myCommand()
                if 'yes' in snfm or 'sure' in snfm:
                    sTime = datetime.datetime.now()               
                    tm = datetime.datetime.now().time()
                    y = num2words(sTime.year, to = 'year')
                    d = sTime.strftime('%A')
                    dt=sTime.strftime("%d")              
                    m = sTime.strftime('%B')
                    ks = "at " + y  +" "+ m +" " + dt + " " + d+ " "+ str(tm) + " this note was taken, the content is:-\n"

                    file.write(ks)                
                    file.write(content)                
                    talk(content)
                else:
                    file.write(content)
                    talk(content)

                talk("your notes are taken. anything else you want me to do")
            
        elif "show note" in command or "show my notes" in command:            
                talk("Showing latest Notes")
                path=r"C:\Users\Rooba\mini-project\my notes"
                
                if not os.listdir(path):
                    print("Empty directory")
                    talk("the notes are empty")
                else:
                    files = os.listdir(path)
                    list_of_files = [os.path.join(path, basename) for basename in files]

                    fname=max(list_of_files,key = os.path.getmtime)
                    print(fname)
                    print()
                    print(os.listdir(path))                   
                    file = open(fname, "r")
                    content = file.read()
                    print(file.read())
                    talk(content)
                    talk("its done.. further you want my help in something else")
                
                    
        
        elif "don't listen" in command or "stop listening" in command:
                speak("for how much time you want to stop elaine from listening commands")
                a = int(myCommand())
                time.sleep(a)
                print(a)
                talk("im back.")
                talk("how can i help you ")

        elif "camera" in command or "take a photo" in command:
                videoCaptureObject = cv2.VideoCapture(0)
                time.sleep(10) 
                path=r"C:\Users\Rooba\mini-project\my pictures from elaine"
                result = True
                while(result):
                    ret,frame = videoCaptureObject.read()
                    now = datetime.datetime.now().strftime("%m-%d-%Y %H:%M:%S%p") 
                    
                    now=now.replace(" ", "")
                    now =now.replace(":","_")
                    now=now.replace("-","_")
                    print(now)
                    image=f"img{now}.jpg"  
                    print(image)

                    cv2.imwrite(os.path.join(path,image), frame)
                    result = False
                videoCaptureObject.release()
                cv2.destroyAllWindows()
                talk('your photo has taken and saved, anything else')
                return
        
        elif "video" in command or "take a video" in command:
            talk("how long you want to capture")
            lnth= myCommand()
            t_s, typ =lnth.split()
            t_s =int(t_s)
            if 'minutes' in typ:
                t_s = t_s * 60
            elif 'hours' in typ:
                t_s = t_s * 3600

            path=r"C:\Users\Rooba\mini-project\my pictures from elaine"
            now = datetime.datetime.now().strftime("%m-%d-%Y %H:%M:%S%p") 
                    
            now=now.replace(" ", "")
            now =now.replace(":","_")
            now=now.replace("-","_")
            print(now)
            video=f"video{now}.avi"  
            print(video)
            capture_duration = t_s

            cap = cv2.VideoCapture(0)

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(os.path.join(path,video),fourcc, 20.0, (640,480))

            start_time = time.time()
            while( int(time.time() - start_time) < capture_duration ):
                ret, frame = cap.read()
                if ret==True:
                    frame = cv2.flip(frame,0)
                    frame = cv2.flip(frame,0)
                    # write the flipped frame
                    out.write(frame)
                    cv2.imshow('frame',frame)        
                else:
                    break

            # Release everything if job is finished
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            talk("video has taken and saved. may  i help you with anything else")
            return

        elif 'news' in command or 'headlines' in command:            

            try:
                jsonObj = urlopen("https://newsapi.org/v2/top-headlines?country=in&apiKey=57dcbb51f9054001967f9988d48f82a4")
                data = json.load(jsonObj)
                print(data)
                i = 1
                 
                talk('here are some top news from the times of india')
                print('''=============== TIMES OF INDIA ============'''+ '\n')
                 
                for item in data['articles']:
                     
                    print(str(i) + '. ' + item['title'] + '\n')
                    print(item['description'] + '\n')
                    talk(str(i) + '. ' + item['title'] + '\n')
                    i += 1
            except Exception as e:                              
                print(str(e))
                talk("i got some error")

            talk("you got the details right? what else can be helpful to you?")
        
        elif 'play music' in command or "play song" in command:
                talk("Here you go with music")               
                path = r"C:\Users\Rooba\Music"                
                playlist = os.listdir(path)
                print(playlist)             
                pygame.init()
                pygame.mixer.init()

                for song in playlist:
                    if song.endswith(".mp3"):
                        file_path = rf"C:\Users\Rooba\Music\{song}"
                        pygame.mixer.music.load(str(file_path))
                        pygame.mixer.music.play()
                        print("Playing::::: " + song)
                        while pygame.mixer.music.get_busy() == True:
                            continue             


                        talk("do you want to continue or exit?")
                        ch = myCommand()
                        if 'exit' in  ch or 'stop' in ch:                               
                                break   
                            
                talk(" i hope you enjoyed the music.. anything else do u need my service")
                
        elif "what is" in command or "who is" in command or "tell me about" in command:
                question = command
                answer = computational_intelligence(question)
                talk(answer)
                talk("anything else you wnat to know")
        
        
        elif 'game' in command:
            talk("shall we start playing rock paper scissors")
            moves=["rock", "paper", "scissor"]
            choice = myCommand()    
            while(choice!= 'no'):
                talk("choose among rock paper or scissor")
                time.sleep(2)
                pmove= myCommand() 
                cmove=random.choice(moves)               
                talk("The computer chose " + cmove)
                
                print("You chose " + pmove)        
                if pmove==cmove:
                    talk("the match is draw")
                elif pmove== "rock" and cmove== "scissor":
                    talk("Player wins")
                elif pmove== "rock" and cmove== "paper":
                    talk("Computer wins")
                elif pmove== "paper" and cmove== "rock":
                    talk("Player wins")
                elif pmove== "paper" and cmove== "scissor":
                    talk("Computer wins")
                elif pmove== "scissor" and cmove== "paper":
                    talk("Player wins")
                elif pmove== "scissor" and cmove== "rock":
                    talk("Computer wins")
                talk("do you want to continue")
                choice = myCommand()
            

            talk("its fun to play with you.. anything else you need my help")   

        elif "where i am" in command or "current location" in command or "where am i" in command:
                try:
                    city, state, country = my_location()
                    print(city, state, country)
                    talk(
                        f"You are currently in {city} city which is in {state} state and country {country}")
                except Exception as e:
                    talk(
                        "Sorry sir, I coundn't fetch your current location. Please try again")

        elif 'battery status' in command or 'battery' in  command:
            battery = psutil.sensors_battery()
            talk("Your system is having " + str(battery.percent) + " percent battery")  
            print("Your system is having " + str(battery.percent) + " percent battery")    

        elif 'what can you do' in command or 'what are all the things you can do' in command:
            talk("i can able to fetch information you want by surfing google and update about news headlines, weather status, and i can play music and youtube to relax you and i help you to send and receive mails and i help you with any mathematical calculation")
            talk("you please  don't worry about anything!!")
            talk("im here to help you with anything")
      
        elif "bye" in command or 'quit' in command or 'exit' in command or 'go offline' in command: 
                talk("Bye. Check Out me for more exicting things.. Im going offline") 
                exit() 
            
        else:
            error = random.choice(errors)
            talk(error)

   

class MainThread(QThread): 

    def __init__(self):
        super(MainThread,self).__init__()

    def run(self): 
        talk("Initializing Elaine")         
        talk("Checking the internet connection")        
        talk("All drivers are up and running")
        talk("All systems have been activated")
        talk("Now I am online")
        talk('Elaine is ready!')
        wishMe()
    
        while True:         
            Elaine(myCommand())    

startExecution = MainThread()      


class Gui_Main(QMainWindow):
    def __init__(self):       
        super().__init__()
        self.elaine_ui = Ui_MainWindow()
        self.elaine_ui.setupUi(self)       
        self.startTask()

    def startTask(self):
       
        self.elaine_ui.movie = QtGui.QMovie(r"C:\Users\Rooba\mini-project\CG.gif")
        self.elaine_ui.label_3.setMovie(self.elaine_ui.movie)
        self.elaine_ui.movie.start()           

        self.elaine_ui.movies_label = QtGui.QMovie(r"C:\Users\Rooba\mini-project\bot.gif")
        self.elaine_ui.label.setMovie(self.elaine_ui.movies_label)
        self.elaine_ui.movies_label.start()   
        
        startExecution.start()
    

app = QApplication(sys.argv)
Gui_elaine = Gui_Main()
Gui_elaine.show()
exit(app.exec_())


    