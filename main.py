from multiprocessing.dummy import Process
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from gui_acentuado import Ui_MainWindow
import os
import cv2
import numpy as np
from scipy import ndimage as nd
import matplotlib.pyplot as plt
from multiprocessing import Process


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton_cargar_img.clicked.connect(self.seleccionar_img)
        # 
        self.ui.pushButton_roberts.clicked.connect(self.aplicar_roberts)
        self.ui.pushButton_prewitt.clicked.connect(self.aplicar_prewitt)
        self.ui.pushButton_sobel.clicked.connect(self.aplicar_sobel)
        self.ui.pushButton_laplace.clicked.connect(self.aplicar_laplace)
        self.ui.pushButton_kirsch.clicked.connect(self.aplicar_kirsch)
        self.ui.pushButton_freichen.clicked.connect(self.aplicar_freichen)
        # 
        self.ui.pushButton_ver_roberts.clicked.connect(self.ver_roberts)
        self.ui.pushButton_ver_prewitt.clicked.connect(self.ver_roberts)
        # bool de control
        self.original_cargada = False
        self.cv_cargada = False
        self.roberts_cargada = False
        self.prewitt_cargada = False
        self.sobel_cargada = False
        self.laplace_cargada = False
        self.kirsch_cargada = False
        self.freichen_cargada = False


    # eventos
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.original_cargada:
            self.mostrar_grafico(self.ui.graphicsView_img_original, self.pixmap_original)
        if self.roberts_cargada:
            self.mostrar_grafico(self.ui.graphicsView_roberts, self.pixmap_roberts)
        if self.prewitt_cargada:
            self.mostrar_grafico(self.ui.graphicsView_prewitt, self.pixmap_prewitt)
        if self.sobel_cargada:
            self.mostrar_grafico(self.ui.graphicsView_sobel, self.pixmap_sobel)
        if self.laplace_cargada:
            self.mostrar_grafico(self.ui.graphicsView_laplace, self.pixmap_laplace)
        if self.kirsch_cargada:
            self.mostrar_grafico(self.ui.graphicsView_kirsch, self.pixmap_kirsch)
        if self.freichen_cargada:
            self.mostrar_grafico(self.ui.graphicsView_freichen, self.pixmap_freichen)

        
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
        self.r, self.g, self.b = cv2.split(self.img_original)
        self.cv_cargada = True

    def imgcv2pixmap(self, img):
        altoimg, anchoimg, channels = img.shape
        bytes_linea = channels * anchoimg
        q_image = QImage(
            img.data.tobytes(), anchoimg, altoimg, bytes_linea, QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_image)
        return pixmap
    
    def ver(self, img):
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    
    def filtro_roberts(self, canal):
        kernelx = np.array([[1, 0], [0, -1]])
        kernely = np.array([[0, 1], [-1, 0]])
        img_robertx = cv2.filter2D(canal, -1, kernelx)
        img_roberty = cv2.filter2D(canal, -1, kernely)
        result = cv2.addWeighted(img_robertx, 1, img_roberty, 0.8, 0)
        return result
    
    def aplicar_roberts(self):
        if self.cv_cargada:
            r = self.filtro_roberts(self.r)
            g = self.filtro_roberts(self.g)
            b = self.filtro_roberts(self.b)
            self.img_roberts = cv2.merge([r,g,b])
            pixmap = self.imgcv2pixmap(self.img_roberts)
            self.pixmap_roberts = QGraphicsPixmapItem(pixmap)
            self.mostrar_grafico(self.ui.graphicsView_roberts, self.pixmap_roberts)
            self.roberts_cargada = True

    def ver_roberts(self):
        if self.roberts_cargada:
            self.ver(self.img_roberts)
            

    def filtro_prewitt(self, canal):
        kernelx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernely = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        img_prewittx = cv2.filter2D(canal, -1, kernelx)
        img_prewitty = cv2.filter2D(canal, -1, kernely)
        edges = cv2.addWeighted(img_prewittx, 0.5, img_prewitty, 0.5, 0)
        return edges
    
    def aplicar_prewitt(self):
        if self.cv_cargada:
            r = self.filtro_prewitt(self.r)
            g = self.filtro_prewitt(self.g)
            b = self.filtro_prewitt(self.b)
            img = cv2.merge([r,g,b])
            pixmap = self.imgcv2pixmap(img)
            self.pixmap_prewitt = QGraphicsPixmapItem(pixmap)
            self.mostrar_grafico(self.ui.graphicsView_prewitt, self.pixmap_prewitt)
            self.prewitt_cargada = True
    
    def filtro_sobel(self, canal):
        sobel = cv2.Sobel(src=canal, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
        sobel = np.array(sobel, dtype=np.uint8)
        return sobel
    
    def aplicar_sobel(self):
        if self.cv_cargada:
            r = self.filtro_sobel(self.r)
            g = self.filtro_sobel(self.g)
            b = self.filtro_sobel(self.b)
            img = cv2.merge((r,g,b))
            pixmap = self.imgcv2pixmap(img)
            self.pixmap_sobel = QGraphicsPixmapItem(pixmap)
            self.mostrar_grafico(self.ui.graphicsView_sobel, self.pixmap_sobel)
            self.sobel_cargada = True

    def filtro_laplace_gaussiano(self, canal):
        imagen_laplace = cv2.Laplacian(canal, cv2.CV_64F)
        imagen_laplace_abs = cv2.convertScaleAbs(imagen_laplace)
        return imagen_laplace_abs
    
    def aplicar_laplace(self):
        if self.cv_cargada:
            r = self.filtro_laplace_gaussiano(self.r)
            g = self.filtro_laplace_gaussiano(self.g)
            b = self.filtro_laplace_gaussiano(self.b)
            img = cv2.merge((r,g,b))
            pixmap = self.imgcv2pixmap(img)
            self.pixmap_laplace = QGraphicsPixmapItem(pixmap)
            self.mostrar_grafico(self.ui.graphicsView_laplace, self.pixmap_laplace)
            self.laplace_cargada = True

    def filtro_kirsch(self, canal):
        kernels = [
            np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=np.float32),
            np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], dtype=np.float32),
            np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], dtype=np.float32),
            np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]], dtype=np.float32),
            np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], dtype=np.float32),
            np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]], dtype=np.float32),
            np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype=np.float32),
            np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], dtype=np.float32)
        ]
        resultados = []
        for kernel in kernels:
            resultado = cv2.filter2D(canal, -1, kernel)
            resultados.append(resultado)
        maximo_absoluto = np.max(np.abs(resultados), axis=0)
        resultado_final = cv2.convertScaleAbs(maximo_absoluto)
        return resultado_final

    def aplicar_kirsch(self):
        if self.cv_cargada:
            r = self.filtro_kirsch(self.r)
            g = self.filtro_kirsch(self.g)
            b = self.filtro_kirsch(self.b)
            img = cv2.merge((r,g,b))
            pixmap = self.imgcv2pixmap(img)
            self.pixmap_kirsch = QGraphicsPixmapItem(pixmap)
            self.mostrar_grafico(self.ui.graphicsView_kirsch, self.pixmap_kirsch)
            self.kirsch_cargada = True

    def filtro_frei_chen(self, canal):
            mask_x = np.array([[-1, 0, 1], [-np.sqrt(2), 0, np.sqrt(2)], [-1, 0, 1]], dtype=np.float32)
            mask_y = np.array([[-1, -np.sqrt(2), -1], [0, 0, 0], [1, np.sqrt(2), 1]], dtype=np.float32)
            gradiente_x = cv2.filter2D(canal, -1, mask_x)
            gradiente_y = cv2.filter2D(canal, -1, mask_y)
            magnitud_gradiente = np.sqrt(np.square(gradiente_x) + np.square(gradiente_y))
            magnitud_gradiente = (magnitud_gradiente / np.max(magnitud_gradiente) * 255).astype(np.uint8)
            return magnitud_gradiente
    
    def aplicar_freichen(self):
        if self.cv_cargada:
            r = self.filtro_frei_chen(self.r)
            g = self.filtro_frei_chen(self.g)
            b = self.filtro_frei_chen(self.b)
            img = cv2.merge((r,g,b))
            pixmap = self.imgcv2pixmap(img)
            self.pixmap_freichen = QGraphicsPixmapItem(pixmap)
            self.mostrar_grafico(self.ui.graphicsView_freichen, self.pixmap_freichen)
            self.freichen_cargada = True


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())