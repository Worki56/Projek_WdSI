import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
dict_size = 128

okes=0
okes2=0

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
    for i in range(0, ile_object):
        if zawartosc_p2['annotation.object'+str(i)+'.name.']=="speedlimit":
            global okes
            okes+=1
            #Punkt 7 w zadaniach
            y=int(zawartosc_p2['annotation.object' + str(i) + '.bndbox.ymax.'])-int(zawartosc_p2['annotation.object' + str(i) + '.bndbox.ymin.'])
            x=int(zawartosc_p2['annotation.object' + str(i) + '.bndbox.xmax.'])-int(zawartosc_p2['annotation.object' + str(i) + '.bndbox.xmin.'])
            if int(zawartosc_p2['annotation.size.width.'])/10 <x and int(zawartosc_p2['annotation.size.height.'])/10 <y:
                zawartosc_p2['annotation.object' + str(i)+'.name.']='1'
            else:
                zawartosc_p2['annotation.object' + str(i) + '.name.'] = '0'
        else:
            zawartosc_p2['annotation.object' + str(i)+'.name.'] = '0'
        zawartosc_p2['annotation.object' + str(i) + '.image.array.']=image[(int(zawartosc_p2['annotation.object' + str(i) + '.bndbox.ymin.'])-1):(int(zawartosc_p2['annotation.object' + str(i) + '.bndbox.ymax.'])-1),(int(zawartosc_p2['annotation.object' + str(i) + '.bndbox.xmin.'])-1):(int(zawartosc_p2['annotation.object' + str(i) + '.bndbox.xmax.'])-1)]
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
        for i in range(0, przy['ile_object']):
            kpts = przesiew.detect(przy['annotation.object' + str(i) + '.image.array.'], None)
            kpts, desc = przesiew.compute(przy['annotation.object' + str(i) + '.image.array.'], kpts)
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
        for i in range(0, przy['ile_object']):
            kpts = przesiew.detect(przy['annotation.object' + str(i) + '.image.array.'], None)
            desc = bow.compute(przy['annotation.object' + str(i) + '.image.array.'], kpts)
            if desc is not None:
                przy['desc'+ str(i)]=desc
            else:
                przy['desc' + str(i)] = np.zeros((1, dict_size))
    return dane

def wyodrebienie3(dane,slownik):
    przesiew = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(przesiew, flann)
    bow.setVocabulary(slownik)
    kpts = przesiew.detect(dane['image'], None)
    desc = bow.compute(dane['image'], kpts)
    if desc is not None:
        dane['desc']=desc
    else:
        dane['desc'] = np.zeros((1, dict_size))
    return dane


def trenowanie(dane):
    clf = RandomForestClassifier(dict_size)
    x = np.empty((1, dict_size))
    y = []
    for przy in dane:
        for i in range(0, przy['ile_object']):
            y.append(przy['annotation.object' + str(i)+'.name.'])
            x = np.vstack((x, przy['desc'+ str(i)]))
    clf.fit(x[1:], y)
    return clf

def predykcja2(rf, dane):
    res=rf.predict(dane['desc'])[0]
    return res

def predykcja(rf, dane):
    for przy in dane:
        przy['label_pred']=predykcja2(rf,przy)
    return dane

def sprawdzanie(rf,sciezka,n,slownik):
    dane = {}
    dane['image'] = cv2.imread(sciezka + '/' + n)
    dane=wyodrebienie3(dane,slownik)
    wynik=predykcja2(rf,dane)
    if wynik=='1':
        global okes2
        okes2+=1
    return True

def wypisz(rf,sciezka):
    slownik = np.load('test/slow.npy')
    lista_plikow = os.listdir(sciezka)
    zawartosc=[]
    for n in lista_plikow:
        print(n)
        sprawdzanie(rf,sciezka,n,slownik)
    return True

def klasyfikacja(rf,scie):
    slownik = np.load('test/slow.npy')
    ile_zdj=int(input())
    for i in range(0, ile_zdj):
        nazwa=input()
        image = cv2.imread(scie + '/' + nazwa)
        ile_wycink=int(input())
        wycik=[]
        for i2 in range(0, ile_wycink):
            wycik.append(input())
        for i2 in range(0, ile_wycink):
            wycik2 = []
            wyraz_tyczas = ''
            for i3 in range(0, len(wycik[i2])):
                if wycik[i2][i3]==' ':
                    wycik2.append(int(wyraz_tyczas))
                else:
                    wyraz_tyczas =wyraz_tyczas +wycik[i2][i3]
            wycik2.append(int(wyraz_tyczas))
            dane={}
            dane['image'] = image[(wycik2[2] - 1):(wycik2[3] - 1), (wycik2[0] - 1):(wycik2[1] - 1)]
            dane=wyodrebienie3(dane,slownik)
            wynik=predykcja2(rf,dane)
            print(wynik)
    return True

def main():
    # Przyjmująć że plik zanjduje się jak w przykładzie
    gdzie="test"
    gdzie2 = "train/images"
    gdzie2 = "test/images"
    os.chdir("..")
    dane_z_plików = odczyt_danych_z_folderu(gdzie)
    # zapisuje plik w folderze "Test"
    uczenie(dane_z_plików, gdzie)
    dane_z_plików=wyodrebienie(dane_z_plików, gdzie)
    rf = trenowanie(dane_z_plików)
    del dane_z_plików
    while True:
        #print('Koniec')
        funkcja = input()
        if funkcja == "classify":
            klasyfikacja(rf,gdzie2)
        elif funkcja == "detect":
            wypisz(rf,gdzie2)
        else:
            print("Error")
        #print(okes)
        #print(okes2)

if __name__ == '__main__':
    main()

