from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from gui_acentuado import Ui_MainWindow
import os
import cv2

class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton_cargar_img.clicked.connect(self.seleccionar_img)
        # bool de control
        self.original_cargada = False

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

    def imgcv2pixmap(self, img):
        altoimg, anchoimg, channels = img.shape
        bytes_linea = channels * anchoimg
        q_image = QImage(
            img.data.tobytes(), anchoimg, altoimg, bytes_linea, QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_image)
        return pixmap




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())