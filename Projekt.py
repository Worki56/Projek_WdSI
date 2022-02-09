import os
import cv2
import numpy as np

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
        if zawartosc_p1[i] == '<':
            if zawartosc_p1[i+1] == '/':
                if wyraz_tyczas!='':
                    laczw =''
                    for n in lacz:
                        laczw=laczw+n+'.'
                    zawartosc_p2[laczw]=wyraz_tyczas
                    wyraz_tyczas=''
                    del lacz[-1]
        elif zawartosc_p1[i] == '>':
            if wyraz_tyczas[0]!='/':
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
    return zawartosc_p2
def odczyt_danych_z_folderu(folder):
    nazwa_f_z_inf="annotations"
    lista_plikow = os.listdir(folder+'/'+nazwa_f_z_inf)
    zawartosc=[]
    for n in lista_plikow:
        zawartosc.append(odczyt_danych_z_pliku(folder,'/'+nazwa_f_z_inf+'/'+n))
    return zawartosc


def uczenie(dane,sciezka):

    dict_size = 128
    bow = cv2.BOWKMeansTrainer(dict_size)

    sift = cv2.SIFT_create()
    for sample in dane:
        kpts = sift.detect(sample['image.array'], None)
        kpts, desc = sift.compute(sample['image.array'], kpts)
        if desc is not None:
            bow.add(desc)
    slownik = bow.cluster()
    np.save(sciezka+'/slow.npy', slownik)

#Przyjmująć że plik zanjduje się jak w przykładzie
os.chdir("..")
dane_z_plików=odczyt_danych_z_folderu("Test")
#zapisuje plik w folderze "Test"
uczenie(dane_z_plików,"Test")
