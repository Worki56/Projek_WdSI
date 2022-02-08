import os
def czyst(zawartosc):
    zawartosc1=''
    for i in range(0, len(zawartosc)):
        if zawartosc[i] != '\n' and zawartosc[i]!=' ':
            zawartosc1=zawartosc1+zawartosc[i]
    return zawartosc1
def odczyt_danych_z_pliku(sciezka):
    #przyjmująć konstrukcje podaną w przykładzie zapisu .xml
    zawartosc_p2={}
    plik = open(sciezka, "r", encoding="utf-8")
    zawartosc_p1= plik.read()
    zawartosc_p1=czyst(zawartosc_p1)
    wyraz_tyczas = ''

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
                lacz.append(wyraz_tyczas)
            wyraz_tyczas=''
        else:
            wyraz_tyczas = wyraz_tyczas + zawartosc_p1[i]
    return zawartosc_p2
def odczyt_danych_z_folderu(folder):
    nazwa_f_z_inf="annotations"
    lista_plikow = os.listdir(folder+'/'+nazwa_f_z_inf)
    zawartosc=[]
    for n in lista_plikow:
        zawartosc.append(odczyt_danych_z_pliku(folder+'/'+nazwa_f_z_inf+'/'+n))

    return zawartosc

jk=odczyt_danych_z_folderu("Test")

