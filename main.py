import sys
import cv2
import time
import numpy as np
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QLabel,
    QWidget,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QApplication,
    QTableWidgetItem,
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QIcon, QCursor
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread

import sudoku as sol
import image_proc as ip
import digit_classify as dc

# Index of the solutions
index = 0

width = 1920
height = 1080
sudoku_size = width * 0.25 - 8
cell_size = int(sudoku_size / 9)


# Class for cv2 - Video capture
class VideoThread(QThread):
    send_solution = pyqtSignal(list, list)
    send_process_percent = pyqtSignal(float)
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, url):
        super().__init__()
        self.row = 0
        self.col = 0
        self.url = url
        self.done = False
        self.sol_list = []
        self._run_flag = True
        self.currentIndex = 0
        self.src_frame = None
        self.con_start = None
        self.discon_start = None
        self.process_percent = 0
        self.sol = [[0 for i in range(9)] for j in range(9)]
        self.matrix = [[0 for i in range(9)] for j in range(9)]
        self.pred = [[[0, 0] for i in range(9)] for j in range(9)]

    def run(self):
        cap = cv2.VideoCapture(self.url)
        while self._run_flag:
            if not self.done:
                if self.src_frame is not None:
                    if self.row < 9:
                        for _ in range(3):
                            img = self.src_frame[36 * self.row + 4:36 * self.row + 32, 36 * self.col + 4:36 * self.col + 32]
                            img = cv2.resize(img, (28, 28))
                            image = img.reshape(1, 28, 28, 1)
                            if image.sum() > 15000:
                                val, prob = dc.classify_image(image)
                                save_img = cv2.resize(img, (52, 52))
                                self.matrix[self.row][self.col] = val
                                self.pred[self.row][self.col] = [val, prob]
                                cv2.imwrite('./cells/' + str(self.row) + str(self.col) + '.jpg', cv2.bitwise_not(save_img))
                            self.col += 1
                        if self.col == 9:
                            self.col = 0
                            self.row += 1
                        self.process_percent += 3.35
                        if self.process_percent < 100:
                            self.send_process_percent.emit(self.process_percent)
                    else:
                        res = sol.sudoku_solver(self.matrix)
                        self.process_percent = 1000
                        self.sol_list = res
                        self.done = True
                        if len(self.sol_list) > 0:
                            self.sol = self.sol_list[globals()['index']]
                        self.send_process_percent.emit(self.process_percent)
                        self.send_solution.emit(self.sol_list, self.pred)

            ret, cv_img = cap.read()
            if ret:
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (7, 7), 0)
                thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                max_area = 0
                contour_grille = None

                for c in contours:
                    area = cv2.contourArea(c)
                    if area > 25000:
                        peri = cv2.arcLength(c, True)
                        polygone = cv2.approxPolyDP(c, 0.05 * peri, True)
                        if area > max_area and len(polygone) == 4:
                            contour_grille = polygone
                            max_area = area

                if contour_grille is not None:
                    if self.con_start is None:
                        self.con_start = time.time()

                    if self.done:
                        self.discon_start = None

                    if time.time() - self.con_start >= 2 or self.done:
                        if self.src_frame is None:
                            self.src_frame = ip.getSudoku(thresh, contour_grille)
                            cv2.imwrite('cc.jpg', self.src_frame)
                        if not(self.currentIndex == globals()['index']):
                            self.currentIndex = globals()['index']
                            if (self.currentIndex < len(self.sol_list)):
                                self.sol = self.sol_list[self.currentIndex]
                        final = ip.img_proc(cv_img, contour_grille, self.sol)
                        self.change_pixmap_signal.emit(final)
                        continue
                else:
                    self.con_start = None
                    if self.done:
                        if self.discon_start is None:
                            self.discon_start = time.time()
                        if time.time() - self.discon_start >= 3:
                            globals()['index'] = 0
                            self.row = 0
                            self.col = 0
                            self.done = False
                            self.sol_list = []
                            self.src_frame = None
                            self.discon_start = None
                            self.process_percent = 0
                            self.sol = [[0 for _ in range(9)] for _ in range(9)]
                            self.matrix = [[0 for _ in range(9)] for _ in range(9)]
                            self.pred = [[[0, 0] for i in range(9)] for j in range(9)]

                self.change_pixmap_signal.emit(cv_img)

        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


# Class for PyQt5 - GUI
class App(QMainWindow):
    def __init__(self):
        super().__init__()

        self.width = 1920
        self.height = 1080
        self.title = 'SUDOKU SCAN SOLVER'
        self.sudoku_size = int(self.width * 0.25 - 8)
        self.cell_size = int(self.sudoku_size / 9)
        self.display_width = self.width * 0.75 * 2 / 3 - 8
        self.display_height = self.display_width * 0.75
        #
        self.x = 500
        self.y = 200
        self.header = QWidget(self)
        self.logo = QLabel(self)
        self.textbox = QLineEdit(self)
        self.team = QLabel(self)
        self.button = QPushButton(self)
        self.image_label = QLabel(self)
        self.thread = VideoThread(0)
        self.sudoku_cotainer = QWidget(self)
        self.button_list = [[None for _ in range(9)] for _ in range(9)]
        self.tableWidget = QTableWidget(self)
        self.label = QLabel(self)
        self.button_left = QPushButton(self)
        self.button_right = QPushButton(self)
        self.textbox1 = QLineEdit(self)
        #
        self.process_percent = 0
        self.solution_list = []
        self.solution = [[0 for i in range(9)] for j in range(9)]
        self.predict = [[[0, 0] for i in range(9)] for j in range(9)]
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.x, self.y, self.width, self.height)
        self.setStyleSheet("background-color: #f9f9f9;")

        self.header.resize(self.width, 54)
        self.header.setStyleSheet("background-color: #ffffff;")

        self.logo.resize(203, 40)
        self.logo.move(self.width * 0.25 / 2, 7)
        pixmap = QPixmap('logo.jpg')
        self.logo.setPixmap(pixmap)

        self.team.resize(85, 40)
        self.team.move(self.width * 0.8825 - 85, 7)
        pixmap = QPixmap('team.jpg')
        self.team.setPixmap(pixmap)
        self.team.setStyleSheet("float: right;")

        self.textbox.resize(self.width / 3 - 64, 32)
        self.textbox.move(self.width / 3, 11)
        self.textbox.setStyleSheet("""
            background-color: #ffffff;
            font-size: 14px;
            padding: 0 8px;
        """)

        self.button.setText("✔")
        self.button.resize(64, 32)
        self.button.move(self.width * 2 / 3 - 66, 11)
        self.button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.button.setIcon(QIcon('assets/cam-icon.png'))
        self.button.setStyleSheet("""
            QPushButton {
                background-color: #f9f9f9;
                border: 1px solid #000000;
            }
            QPushButton::hover {
                background-color : #e7e7e7;
            }
            QPushButton::pressed {
                background-color : #b7b7b7;
            }
        """)
        self.button.clicked.connect(self.on_click)

        self.textbox1.resize(0, 0)
        self.textbox1.setFocus()

        self.image_label.move(self.width * 0.25 / 2, 84)
        self.image_label.resize(self.display_width, self.display_height)
        self.image_label.setStyleSheet("background-color: #000000;")

        self.init_video_thread(0)

        self.sudoku_cotainer.resize(self.sudoku_size + 12, self.sudoku_size + 12)
        self.sudoku_cotainer.move(self.width * 0.625 + 16, 84)
        self.sudoku_cotainer.setStyleSheet("background-color: #000000;")

        for row in range(9):
            for col in range(9):
                button = QPushButton(self)
                if self.solution[row][col] > 0:
                    button.setText(str(self.solution[row][col]))
                button.move(self.width * 0.625 + 18 + (self.cell_size + 1) * col + int(col / 3) * 2, 86 + (self.cell_size + 1) * row + int(row / 3) * 2)
                button.resize(self.cell_size, self.cell_size)
                button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
                if self.solution[row][col] < 0:
                    button.setStyleSheet("border: none; background-image : url(cells/" + str(row) + str(col) + ".jpg);")
                else:
                    button.setStyleSheet("border: none; background-color: white; color: #69A7F0")
                self.button_list[row][col] = button

        self.tableWidget.setRowCount(1)
        self.tableWidget.setColumnCount(2)
        self.tableWidget.resize(sudoku_size + 12, 64)
        self.tableWidget.move(self.width * 0.625 + 16, 84 + sudoku_size + 32)
        self.tableWidget.verticalHeader().hide()
        self.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.tableWidget.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.tableWidget.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.tableWidget.setHorizontalHeaderLabels(["Predict", "Probability"])

        self.label.resize(sudoku_size - 96, 48)
        self.label.move(self.width * 0.625 + 18, 84 + sudoku_size + 180)
        self.label.setText('Scan the sudoku')
        self.label.setStyleSheet("font-size: 32px;")

        self.button_left.resize(48, 48)
        self.button_left.move(self.width * 0.875 - 74, 84 + sudoku_size + 180)
        self.button_left.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.button_left.setText("🢀")
        self.button_left.setStyleSheet("""
            QPushButton {
                background-color: #f9f9f9;
                border: none;
                font-size: 24px;
                border-radius : 24px;
            }
            QPushButton::hover {
                background-color : #e7e7e7;
            }
            QPushButton::pressed {
                background-color : #b7b7b7;
            }
        """)

        self.button_right.resize(48, 48)
        self.button_right.move(self.width * 0.875 - 26, 84 + sudoku_size + 180)
        self.button_right.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.button_right.setText("🢂")
        self.button_right.setStyleSheet("""
            QPushButton {
                background-color: #f9f9f9;
                border: none;
                font-size: 24px;
                border-radius : 24px;
            }
            QPushButton::hover {
                background-color : #e7e7e7;
            }
            QPushButton::pressed {
                background-color : #b7b7b7;
            }
        """)

        self.setting_buttons_event()
        self.button_left.clicked.connect(lambda: self.changeIndex(-1))
        self.button_right.clicked.connect(lambda: self.changeIndex(1))

    def init_video_thread(self, url):
        self.thread.stop()
        self.thread = VideoThread(url)
        self.thread.send_solution.connect(self.receive_solution)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.send_process_percent.connect(self.receive_process_percent)
        self.thread.start()

    def setting_buttons_event(self):
        self.button_list[0][0].clicked.connect(lambda: self.clickme(0, 0))
        self.button_list[0][1].clicked.connect(lambda: self.clickme(0, 1))
        self.button_list[0][2].clicked.connect(lambda: self.clickme(0, 2))
        self.button_list[0][3].clicked.connect(lambda: self.clickme(0, 3))
        self.button_list[0][4].clicked.connect(lambda: self.clickme(0, 4))
        self.button_list[0][5].clicked.connect(lambda: self.clickme(0, 5))
        self.button_list[0][6].clicked.connect(lambda: self.clickme(0, 6))
        self.button_list[0][7].clicked.connect(lambda: self.clickme(0, 7))
        self.button_list[0][8].clicked.connect(lambda: self.clickme(0, 8))

        self.button_list[1][0].clicked.connect(lambda: self.clickme(1, 0))
        self.button_list[1][1].clicked.connect(lambda: self.clickme(1, 1))
        self.button_list[1][2].clicked.connect(lambda: self.clickme(1, 2))
        self.button_list[1][3].clicked.connect(lambda: self.clickme(1, 3))
        self.button_list[1][4].clicked.connect(lambda: self.clickme(1, 4))
        self.button_list[1][5].clicked.connect(lambda: self.clickme(1, 5))
        self.button_list[1][6].clicked.connect(lambda: self.clickme(1, 6))
        self.button_list[1][7].clicked.connect(lambda: self.clickme(1, 7))
        self.button_list[1][8].clicked.connect(lambda: self.clickme(1, 8))

        self.button_list[2][0].clicked.connect(lambda: self.clickme(2, 0))
        self.button_list[2][1].clicked.connect(lambda: self.clickme(2, 1))
        self.button_list[2][2].clicked.connect(lambda: self.clickme(2, 2))
        self.button_list[2][3].clicked.connect(lambda: self.clickme(2, 3))
        self.button_list[2][4].clicked.connect(lambda: self.clickme(2, 4))
        self.button_list[2][5].clicked.connect(lambda: self.clickme(2, 5))
        self.button_list[2][6].clicked.connect(lambda: self.clickme(2, 6))
        self.button_list[2][7].clicked.connect(lambda: self.clickme(2, 7))
        self.button_list[2][8].clicked.connect(lambda: self.clickme(2, 8))

        self.button_list[3][0].clicked.connect(lambda: self.clickme(3, 0))
        self.button_list[3][1].clicked.connect(lambda: self.clickme(3, 1))
        self.button_list[3][2].clicked.connect(lambda: self.clickme(3, 2))
        self.button_list[3][3].clicked.connect(lambda: self.clickme(3, 3))
        self.button_list[3][4].clicked.connect(lambda: self.clickme(3, 4))
        self.button_list[3][5].clicked.connect(lambda: self.clickme(3, 5))
        self.button_list[3][6].clicked.connect(lambda: self.clickme(3, 6))
        self.button_list[3][7].clicked.connect(lambda: self.clickme(3, 7))
        self.button_list[3][8].clicked.connect(lambda: self.clickme(3, 8))

        self.button_list[4][0].clicked.connect(lambda: self.clickme(4, 0))
        self.button_list[4][1].clicked.connect(lambda: self.clickme(4, 1))
        self.button_list[4][2].clicked.connect(lambda: self.clickme(4, 2))
        self.button_list[4][3].clicked.connect(lambda: self.clickme(4, 3))
        self.button_list[4][4].clicked.connect(lambda: self.clickme(4, 4))
        self.button_list[4][5].clicked.connect(lambda: self.clickme(4, 5))
        self.button_list[4][6].clicked.connect(lambda: self.clickme(4, 6))
        self.button_list[4][7].clicked.connect(lambda: self.clickme(4, 7))
        self.button_list[4][8].clicked.connect(lambda: self.clickme(4, 8))

        self.button_list[5][0].clicked.connect(lambda: self.clickme(5, 0))
        self.button_list[5][1].clicked.connect(lambda: self.clickme(5, 1))
        self.button_list[5][2].clicked.connect(lambda: self.clickme(5, 2))
        self.button_list[5][3].clicked.connect(lambda: self.clickme(5, 3))
        self.button_list[5][4].clicked.connect(lambda: self.clickme(5, 4))
        self.button_list[5][5].clicked.connect(lambda: self.clickme(5, 5))
        self.button_list[5][6].clicked.connect(lambda: self.clickme(5, 6))
        self.button_list[5][7].clicked.connect(lambda: self.clickme(5, 7))
        self.button_list[5][8].clicked.connect(lambda: self.clickme(5, 8))

        self.button_list[6][0].clicked.connect(lambda: self.clickme(6, 0))
        self.button_list[6][1].clicked.connect(lambda: self.clickme(6, 1))
        self.button_list[6][2].clicked.connect(lambda: self.clickme(6, 2))
        self.button_list[6][3].clicked.connect(lambda: self.clickme(6, 3))
        self.button_list[6][4].clicked.connect(lambda: self.clickme(6, 4))
        self.button_list[6][5].clicked.connect(lambda: self.clickme(6, 5))
        self.button_list[6][6].clicked.connect(lambda: self.clickme(6, 6))
        self.button_list[6][7].clicked.connect(lambda: self.clickme(6, 7))
        self.button_list[6][8].clicked.connect(lambda: self.clickme(6, 8))

        self.button_list[7][0].clicked.connect(lambda: self.clickme(7, 0))
        self.button_list[7][1].clicked.connect(lambda: self.clickme(7, 1))
        self.button_list[7][2].clicked.connect(lambda: self.clickme(7, 2))
        self.button_list[7][3].clicked.connect(lambda: self.clickme(7, 3))
        self.button_list[7][4].clicked.connect(lambda: self.clickme(7, 4))
        self.button_list[7][5].clicked.connect(lambda: self.clickme(7, 5))
        self.button_list[7][6].clicked.connect(lambda: self.clickme(7, 6))
        self.button_list[7][7].clicked.connect(lambda: self.clickme(7, 7))
        self.button_list[7][8].clicked.connect(lambda: self.clickme(7, 8))

        self.button_list[8][0].clicked.connect(lambda: self.clickme(8, 0))
        self.button_list[8][1].clicked.connect(lambda: self.clickme(8, 1))
        self.button_list[8][2].clicked.connect(lambda: self.clickme(8, 2))
        self.button_list[8][3].clicked.connect(lambda: self.clickme(8, 3))
        self.button_list[8][4].clicked.connect(lambda: self.clickme(8, 4))
        self.button_list[8][5].clicked.connect(lambda: self.clickme(8, 5))
        self.button_list[8][6].clicked.connect(lambda: self.clickme(8, 6))
        self.button_list[8][7].clicked.connect(lambda: self.clickme(8, 7))
        self.button_list[8][8].clicked.connect(lambda: self.clickme(8, 8))

    def keyPressEvent(self, qKeyEvent):
        if qKeyEvent.key() == QtCore.Qt.Key_Return:
            self.textbox1.setFocus()
            url = self.textbox.text()
            if url == "":
                self.init_video_thread(0)
            else:
                self.init_video_thread(url)
        else:
            super().keyPressEvent(qKeyEvent)

    def reset_buttons(self):
        for row in range(9):
            for col in range(9):
                if self.solution[row][col] < 0:
                    self.button_list[row][col].setText("")
                    self.button_list[row][col].setStyleSheet("border: none; background-image : url(cells/" + str(row) + str(col) + ".jpg);")
                else:
                    temp_str = str(self.solution[row][col])
                    if self.solution[row][col] == 0:
                        temp_str = ""
                    self.button_list[row][col].setText(temp_str)
                    self.button_list[row][col].setStyleSheet("""
                        border: none;
                        color: #69A7F0;
                        font-size: 38px;
                        font-weight: bold;
                        background-color: white;
                    """)

    @pyqtSlot()
    def on_click(self):
        self.textbox1.setFocus()
        url = self.textbox.text()
        if url == "":
            self.init_video_thread(0)
        else:
            self.init_video_thread(url)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    @pyqtSlot(list, list)
    def receive_solution(self, sol, pred):
        self.solution_list = sol
        temp_len = len(sol)
        if temp_len == 0 or temp_len == 1:
            self.label.setText("Found: " + str(temp_len) + " solution")
        else:
            self.label.setText("Found: " + str(temp_len) + " solutions")
        if len(self.solution_list) > 0:
            self.solution = self.solution_list[globals()['index']]
        self.predict = pred
        self.reset_buttons()
        if temp_len == 0:
            print(self.predict)
            for row in range(9):
                for col in range(9):
                    if self.predict[row][col][0] > 0:
                        self.button_list[row][col].setText("")
                        self.button_list[row][col].setStyleSheet(
                            "border: none; background-image : url(cells/" + str(row) + str(col) + ".jpg);")
                    else:
                        self.button_list[row][col].setText("")
                        self.button_list[row][col].setStyleSheet("""
                            border: none;
                            background-color: #ffffff;
                        """)

    @pyqtSlot(float)
    def receive_process_percent(self, process_percent):
        self.process_percent = process_percent
        self.label.setText("Solving: " + str(int(min(self.process_percent, 100))) + "%")


    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    @pyqtSlot(int)
    def clickme(self, row, col):
        str1 = ""
        str2 = ""
        if self.predict[row][col][0] > 0:
            str1 = str(self.predict[row][col][0])
            str2 = str(self.predict[row][col][1])

        item0 = QTableWidgetItem(str1)  # create the item
        item0.setTextAlignment(Qt.AlignHCenter)  # change the alignment
        self.tableWidget.setItem(0, 0, item0)

        item1 = QTableWidgetItem(str2)  # create the item
        item1.setTextAlignment(Qt.AlignHCenter)  # change the alignment
        self.tableWidget.setItem(0, 1, item1)

    @pyqtSlot(int)
    def changeIndex(self, num):
        len_cc = len(self.solution_list)
        if len_cc > 0:
            globals()['index'] = (globals()['index'] + num + len_cc) % len_cc
            self.solution = self.solution_list[globals()['index']]
            self.reset_buttons()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.showMaximized()
    sys.exit(app.exec_())