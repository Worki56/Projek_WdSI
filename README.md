# Projekt WdSI

## Opis projektu
Projekt polega na wykorzystaniu uczenia maszynowego do wykrycia na zdjęciach znaków ograniczenia prędkości oraz ich pozycji na zdjęciu
##Wymagana umiejscowienie plików
`
├── test
   ├── annotations
   └── images
├── train
   ├── annotations
   └── images
└── To_repozytorium
   └── main.py
`
## Przykładowy plik .xml w folderze "annotations"
`
<annotation>
   <folder>images</folder>
   <filename>road214.png</filename>
   <size>
       <width>300</width>
       <height>400</height>
       <depth>3</depth>
   </size>
   <segmented>0</segmented>
   <object>
       <name>speedlimit</name>
       <pose>Unspecified</pose>
       <truncated>0</truncated>
       <occluded>0</occluded>
       <difficult>0</difficult>
       <bndbox>
           <xmin>127</xmin>
           <ymin>166</ymin>
           <xmax>145</xmax>
           <ymax>185</ymax>
       </bndbox>
   </object>
</annotation>
`
## Wyjaśnienie poleceń
Po nauce klasyfikatora możliwość jest użycie następujących poleceń:
* classify sprawdzanie pojedynczych zbiorów zdjęć aby sprawdzić klase 
* detect sprawdzenie całego folderu zdjęć train/images wypisanie znajdujących się elementów oraz ich położenia 
