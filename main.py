from multiprocessing.dummy import Process
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from gui_acentuado import Ui_MainWindow
import os
import cv2
import numpy as np
from scipy import ndimage
from multiprocessing import Process

class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton_cargar_img.clicked.connect(self.seleccionar_img)
        # bool de control
        self.original_cargada = False
        self.cv_cargada = False

    # eventos
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.original_cargada:
            self.mostrar_grafico(self.ui.graphicsView_img_original, self.pixmap_original)
        
    def seleccionar_img(self):
        archivo = QFileDialog()
        archivo.setWindowTitle("Seleccionar imagen")
        archivo.setFileMode(QFileDialog.ExistingFile)
        if archivo.exec_():
            ruta = archivo.selectedFiles()
            ruta = ruta[0]
            ruta_absoluta = os.path.abspath(ruta)
            ruta_normalizada = os.path.normpath(ruta_absoluta)
            self.mostrar_img(ruta_normalizada)

    def mostrar_img(self, ruta):
        # graphicsview
        pixmap = QPixmap(ruta)
        self.pixmap_original = QGraphicsPixmapItem(pixmap)
        self.mostrar_grafico(self.ui.graphicsView_img_original, self.pixmap_original)
        self.original_cargada = True
        self.ui.label_datos_original.setText(f"Ancho: {pixmap.width()}px ~ Alto: {pixmap.height()}px")
        # opencv
        self.cargar_cv(ruta)

    def mostrar_grafico(self, objeto, pixmap):
        scene = QGraphicsScene()
        scene.addItem(pixmap)
        objeto.setScene(scene)
        objeto.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        objeto.show()

    def cargar_cv(self, ruta):
        self.img_original = cv2.imread(ruta)
        self.r, self.g, self.b = cv2.split()
        self.cv_cargada = True

    def imgcv2pixmap(self, img):
        altoimg, anchoimg, channels = img.shape
        bytes_linea = channels * anchoimg
        q_image = QImage(
            img.data.tobytes(), anchoimg, altoimg, bytes_linea, QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_image)
        return pixmap
    
    def filtro_roberts(self, canal):
        kernelx = np.array([[ 0, 1 ],[ -1, 0 ]])
        kernely = np.array([[1, 0 ],[0,-1 ]])
        x = ndimage.convolve( canal, kernelx )
        y = ndimage.convolve( canal, kernely )
        img_bordes = np.sqrt( np.square(x) + np.square(y))
        return img_bordes
    
    def aplicar_roberts(self,r,g,b):
        r = self.filtro_roberts(r)
        g = self.filtro_roberts(g)
        b = self.filtro_roberts(b)
        img = cv2.merge([r,g,b])
        pixmap = self.imgcv2pixmap(img)
        self.mostrar_grafico(self.ui.graphicsView_roberts, pixmap)

    def proceso_roberts(self,r,g,b):
        proceso = Process(target=self.aplicar_roberts(r,g,b))
        proceso.start()
        proceso.join()



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())