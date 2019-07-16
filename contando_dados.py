import cv2
import mahotas
import numpy

img = cv2.imread('dados.jpeg')

# Tons de cinza
imgCinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplicando filtro blur
blur = cv2.blur(imgCinza, (7, 7))

# Binarizando a imagem
imgBin = blur.copy()
imgBin[imgBin > mahotas.thresholding.otsu(blur)] = 255
imgBin[imgBin < 255] = 0
imgBin = cv2.bitwise_not(imgBin)

# Detecção de bordas
bordas = cv2.Canny(imgBin, 70, 150)

# Contagem dos contornos
(objetos, lx) = cv2.findContours(bordas.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.putText(imgCinza, 'Tons de cinza', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 0, cv2.LINE_AA)
cv2.putText(blur, 'Blur', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 0, cv2.LINE_AA)
cv2.putText(imgBin, 'Binarizacao', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 0, cv2.LINE_AA)
cv2.putText(bordas, 'Bordas', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 0, cv2.LINE_AA)

# Exibindo imagens
cv2.imshow("Quantidade de objetos: "+str(len(objetos)), numpy.vstack([numpy.hstack([imgCinza, blur]),numpy.hstack([imgBin, bordas])]))
cv2.waitKey(0)
imgC2 = img.copy()
cv2.imshow("Imagem Original", img)
cv2.putText(imgC2, str(len(objetos)) + ' objetos encontrados', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 0, cv2.LINE_AA)
cv2.drawContours(imgC2, objetos, -1, (255, 0, 0), 2)
cv2.imshow('Resultado', imgC2)
cv2.waitKey(0);