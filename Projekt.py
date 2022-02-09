import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
dict_size = 128

def czyst(zawartosc):
    zawartosc1=''
    for i in range(0, len(zawartosc)):
        if zawartosc[i] != '\n' and zawartosc[i]!=' ':
            zawartosc1=zawartosc1+zawartosc[i]
    return zawartosc1

def odczyt_danych_z_pliku(sciezk,sciezk1):
    #przyjmująć konstrukcje podaną w przykładzie zapisu .xml
    #można była skorzytać z pandas.read_xml ale wydawało mi się że jest to pójście na łatwizne
    sciezka=sciezk+sciezk1
    scie = sciezk + '/images/'
    zawartosc_p2={}
    plik = open(sciezka, "r", encoding="utf-8")
    zawartosc_p1= plik.read()
    zawartosc_p1=czyst(zawartosc_p1)
    wyraz_tyczas = ''
    ile_object=0
    lacz=[]
    for i in range(0, len(zawartosc_p1)):
        co=zawartosc_p1[i]
        if zawartosc_p1[i] == '<':
            if zawartosc_p1[i+1] == '/':
                laczw =''
                for n in lacz:
                    laczw=laczw+n+'.'
                if wyraz_tyczas!='':
                    zawartosc_p2[laczw]=wyraz_tyczas
                wyraz_tyczas=''
                del lacz[-1]
        elif zawartosc_p1[i] == '>':
            if wyraz_tyczas[0]!='/' and wyraz_tyczas[0]!='':
                if wyraz_tyczas=='object':
                    lacz.append(wyraz_tyczas+str(ile_object))
                    ile_object+=1
                else:
                    lacz.append(wyraz_tyczas)
            wyraz_tyczas=''

        else:
            wyraz_tyczas = wyraz_tyczas + zawartosc_p1[i]
    zawartosc_p2['ile_object'] = ile_object
    if ile_object != 0:
        image = cv2.imread(scie+zawartosc_p2['annotation.filename.'])
        zawartosc_p2['image.array']=image
    for i in range(0, ile_object):
        if zawartosc_p2['annotation.object'+str(i)+'.name.']=="speedlimit":
            zawartosc_p2['new.annotation.object' + str(i)+'.name.']='1'
        else:
            zawartosc_p2['new.annotation.object' + str(i)+'.name.'] = '0'
    return zawartosc_p2

def odczyt_danych_z_folderu(folder):
    nazwa_f_z_inf="annotations"
    lista_plikow = os.listdir(folder+'/'+nazwa_f_z_inf)
    zawartosc=[]
    for n in lista_plikow:
        zawartosc.append(odczyt_danych_z_pliku(folder,'/'+nazwa_f_z_inf+'/'+n))
    return zawartosc

def uczenie(dane,sciezka):
    bow = cv2.BOWKMeansTrainer(dict_size)
    przesiew = cv2.SIFT_create()
    for przy in dane:
        kpts = przesiew.detect(przy['image.array'], None)
        kpts, desc = przesiew.compute(przy['image.array'], kpts)
        if desc is not None:
            bow.add(desc)
    slownik = bow.cluster()
    np.save(sciezka+'/slow.npy', slownik)

def wyodrebienie(dane,sciezka):

    przesiew = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(przesiew, flann)
    slownik = np.load(sciezka+'/slow.npy')
    bow.setVocabulary(slownik)
    for przy in dane:
        kpts = przesiew.detect(przy['image.array'], None)
        desc = bow.compute(przy['image.array'], kpts)
        if desc is not None:
            przy.update({'desc': desc})
        else:
            przy.update({'desc': np.zeros((1, dict_size))})
    return dane

def trenowanie(dane):
    clf = RandomForestClassifier(dict_size)
    x = np.empty((1, dict_size))
    y = []
    #do poprawy później
    for przy in dane:
        for i in range(0, przy['ile_object']):
            y.append(przy['new.annotation.object' + str(i)+'.name.'])
            x = np.vstack((x, przy['desc']))
    clf.fit(x[1:], y)
    return clf

def predykcja(rf, dane):
    #do poprawy
    for przy in dane:
        przy['label_pred']=rf.predict(przy['desc'])[0]
    return dane


#Przyjmująć że plik zanjduje się jak w przykładzie
os.chdir("..")
dane_z_plików=odczyt_danych_z_folderu("Test")
#zapisuje plik w folderze "Test"
uczenie(dane_z_plików,"Test")
wyodrebienie(dane_z_plików,"Test")
rf = trenowanie(dane_z_plików)
dane_z_plików = predykcja(rf, dane_z_plików)

