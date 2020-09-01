from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton, QTreeView, \
    QTableWidgetItem, QGraphicsScene, QAction
# from PyQt5.QtGui import QStandardItemModel, QImage
from PyQt5.QtGui import *
from PyQt5.QtCore import QTimer, Qt, QDate, QRectF

from ui import Ui_MainWindow
# from predict2 import var
from image_change import image_change_view
from server_connect import aws_connect,aws_connect2
import sys
import sqlite3
# import cv2



# class sql(QWidget):
class test(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.sqlConnect()
        self.setupUi(self)
        self.run()
        self.show()

    #DB 연결
    def sqlConnect(self):
        try:
            self.conn = sqlite3.connect('test.db')

        except:
            print("잘묏된 방식입니다.")
            exit(1)
        print("연결 성공")
        self.cur = self.conn.cursor()

    #DB table 확인
    def run(self):
        self.cmd="SELECT * FROM sqlite_master WHERE type='table';"
        self.cur.execute(self.cmd)
        self.conn.commit()
        print(self.cur.fetchall())
        # pass

    def closeEvent(self,QCloseEvent):
        self.conn.close()

    # 새로고침 선택 시 환자 목록을 확인 하는 기능, 오늘 날짜에 맞춰서 실행 된다.
    def startFound(self):
        self.patient_list_fun = 0
        b = self.info.rowCount()
        for i in range(b):
            self.info.removeRow(i)
        print("main")
        # 오늘 날짜를 불러오는 과정
        self.today = QDate.currentDate()
        self.day_year = self.today.year()
        self.day_month = self.today.month()
        self.day_day = self.today.day()
        # print(type(self.today_year))
        # print(self.today_month)
        # print(self.today_day)

        #오늘 날짜 값을 가지고 DB안에서 값 호출
        self.cmd = "select chart_index,patient_name,patient_birth_info FROM patient_list where patient_visit_day = '{}.{}.{}'".format(self.day_year,self.day_month,self.day_day)
        self.cur.execute(self.cmd)
        self.conn.commit()
        ar = self.cur.fetchall()

        #호출된 값들을 목록에 저장하는 과정
        for i in range(len(ar)):
            self.info.removeRow(len(ar))
            self.info.removeRow(i)
            self.info.insertRow(i)
            self.info.setData(self.info.index(i, 0), ar[i][0])
            self.info.setData(self.info.index(i, 1), ar[i][1])
            self.info.setData(self.info.index(i, 2), ar[i][2])

    #환자 선택 후, 환자의 사진 목록을 불러오는 과정
    def find_patient_image_list(self):
        # self.info_image.modelReset()
        # self.info_image.resetInternalData()
        #선택한 환자의 row index 값
        a = self.patient_list.currentIndex().row()
        # b = self.patient_list.currentIndex().column()

        #환자 사진 목록 초기화
        b = self.info_image.rowCount()
        for i in range(b) :
            self.info_image.removeRow(i)

        if self.patient_list_fun == 0 :


            self.cmd = "select chart_index,patient_name,patient_birth_info FROM patient_list where patient_visit_day = '{}.{}.{}'".format(
                self.day_year, self.day_month, self.day_day)
            self.cur.execute(self.cmd)
            self.conn.commit()
            ar = self.cur.fetchall()

            # print(ar)
            print(ar[a][1])

            self.patient_name=ar[a][1]
            self.chart_index =ar[a][0]

        self.cmd = "select test_day,image_name,etc FROM patient_image where patient_name = '" +self.patient_name+ "' and chart_index = '{}'".format(self.chart_index)
        self.cur.execute(self.cmd)
        self.conn.commit()
        ar1 = self.cur.fetchall()


        for i in range(len(ar1)):
            self.info_image.removeRow(i)
            self.info_image.insertRow(i)
            self.info_image.setData(self.info_image.index(i, 0), ar1[i][0])
            self.info_image.setData(self.info_image.index(i, 1), ar1[i][1])
            self.info_image.setData(self.info_image.index(i, 2), ar1[i][2])


        #환자 검색
    def search_patient_list(self):
        self.patient_list_fun = 0
        b = self.info.rowCount()
        for i in range(b):
            self.info.removeRow(i)

        self.patient_name = self.lineEdit.text()
        c = self.patient_list.currentIndex().row()


        self.cmd = "select * FROM patient_list where patient_name = '" +self.patient_name+ "'"
        self.cur.execute(self.cmd)
        self.conn.commit()
        ar = self.cur.fetchall()

        d =list(ar[c][3].split("."))
        self.day_year = d[0]
        self.day_month= d[1]
        self.day_day = d[2]
        self.chart_index = ar[c][0]



        for i in range(len(ar)):
            self.info.removeRow(len(ar))
            self.info.removeRow(i)
            self.info.insertRow(i)
            self.info.setData(self.info.index(i, 0), ar[i][0])
            self.info.setData(self.info.index(i, 1), ar[i][1])
            self.info.setData(self.info.index(i, 2), ar[i][2])



    def show_patient_image(self):
        self.cmd = "select test_day,image_name,etc FROM patient_image where patient_name = '" + self.patient_name + "'"
        self.cur.execute(self.cmd)
        self.conn.commit()
        ar = self.cur.fetchall()
        a = self.patient_image.currentIndex().row()
        self.test_day = ar[a][0]
        self.patient_image_file = ar[a][1]
        self.etc = ar[a][2]
        print(self.patient_image_file)

        image_change_view('./image/' + self.patient_image_file + '.png')
        self.qPixmapVar = QPixmap()
        # print("check1")
        self.qPixmapVar.load('./output.png')
        # print("check2")
        self.qPixmapVar_1 = self.qPixmapVar
        # self.qPixmapVar_size = self.qPixmapVar.size()
        # print(self.qPixmapVar_size.width())
        # print(self.qPixmapVar_size.height())

        # if self.qPixmapVar_size.width() > self.qPixmapVar_size.height():
        #     self.qPixmapVar = self.qPixmapVar.scaledToHeight(460)
        # else:
        #     self.qPixmapVar = self.qPixmapVar.scaledToWidth(460)

        # self.qPixmapVar = self.qPixmapVar.scaled(480,480)

        self.scene = QGraphicsScene()
        self.scene.addPixmap(self.qPixmapVar)

        self.graphicsView.setScene(self.scene)


    def image_size_up(self):
        self.qPixmapVar = self.qPixmapVar.scaled((self.qPixmapVar.size().width())*1.2,(self.qPixmapVar.size().height())*1.2)
        self.scene = QGraphicsScene()
        self.scene.addPixmap(self.qPixmapVar)
        self.graphicsView.setScene(self.scene)

    def image_size_down(self):
        self.qPixmapVar = self.qPixmapVar.scaled((self.qPixmapVar.size().width()) * 0.8,
                                                 (self.qPixmapVar.size().height()) * 0.8)
        self.scene = QGraphicsScene()
        self.scene.addPixmap(self.qPixmapVar)
        self.graphicsView.setScene(self.scene)

    def origin_image_size(self):
        # self.qPixmapVar.load('C:/Users/user/Desktop/QT/image/' + self.patient_image_file + '.png')
        self.scene = QGraphicsScene()
        self.scene.addPixmap(self.qPixmapVar_1)
        self.graphicsView.setScene(self.scene)


        #
        # qPixmapVar = QPixmap()
        # qPixmapVar.load('C:/Users/user/Desktop/QT/image/' + self.patient_image_file + '.png')
        # qPixmapVar= qPixmapVar.scaled(450, 450, QtCore.Qt.KeepAspectRatio)
        # self.image_view.setPixmap(qPixmapVar)
        # self.patient_image = QImage('C:/Users/user/Downloads/' + self.patient_image_file + '.png')
        # if self.patient_image.isNull():
        #     print("Error loading image")
        #     sys.exit(1)
        # self.image_view.setPixmap(QtGui.QPixmap('C:/Users/user/Desktop/QT/image/'+self.patient_image_file+'.png'))

    def calendar_date(self):
        self.patient_list_fun = 0
        b = self.info.rowCount()
        for i in range(b) :
            self.info.removeRow(i)

        self.select_day = self.calendarWidget.selectedDate()
        self.day_year = self.select_day.year()
        self.day_month = self.select_day.month()
        self.day_day = self.select_day.day()
        # print(type(self.day_year))
        # print(self.day_month)
        # print(self.day_day)

        self.cmd = "select chart_index,patient_name,patient_birth_info FROM patient_list where patient_visit_day = '{}.{}.{}'".format(
            self.day_year, self.day_month, self.day_day)
        self.cur.execute(self.cmd)
        self.conn.commit()
        ar = self.cur.fetchall()

        for i in range(len(ar)):
            self.info.removeRow(len(ar))
            self.info.removeRow(i)
            self.info.insertRow(i)
            self.info.setData(self.info.index(i, 0), ar[i][0])
            self.info.setData(self.info.index(i, 1), ar[i][1])
            self.info.setData(self.info.index(i, 2), ar[i][2])

    def cancer_grade_predict(self):
        self.path = './image/' + self.patient_image_file + '.png'
        # print(self.path)
        # self.isup_grade = var(self.path)
        self.isup_grade = aws_connect(self.path)
        # self.isup_grade =2
        print(self.isup_grade)
        self.lcdNumber.display(self.isup_grade)

    def cancer_grade_segment(self):
        self.path = './image/' + self.patient_image_file + '.png'
        # print(self.path)
        # self.isup_grade = var(self.path)
        self.isup_grade = aws_connect2(self.path)
        # self.isup_grade =2
        print(self.isup_grade)
        self.lcdNumber.display(self.isup_grade)

    def image_copy(self):
        self.qPixmapVar.save('./image/' + self.patient_image_file +'{}{}{}.png'.format(self.day_year,self.day_month,self.day_day))
        self.etc ="copy"
        # print(type(self.patient_image_file))
        # print(type(self.etc))
        # print(type(self.patient_name))
        # print(type(self.test_day))
        # print(type(self.chart_index))
        # print(type(self.day_year))
        # print(type(self.day_day))
        # print(type(self.day_month))

        self.cmd = f"insert into patient_image values ({self.test_day},'{self.patient_image_file}{self.day_year}{self.day_month}{self.day_day}','{self.etc}','{self.patient_name}',{self.chart_index})"
        self.cur.execute(self.cmd)
        self.conn.commit()

        print("check1")

        b = self.info_image.rowCount()
        for i in range(b):
            self.info_image.removeRow(i)
        print("check2")
        self.cmd = f"select test_day,image_name,etc FROM patient_image where patient_name = '{self.patient_name}' and chart_index = {self.chart_index}"

        self.cur.execute(self.cmd)
        self.conn.commit()
        ar1 = self.cur.fetchall()

        print(ar1)

        for i in range(len(ar1)):

            self.info_image.removeRow(i)
            self.info_image.insertRow(i)
            self.info_image.setData(self.info_image.index(i, 0), ar1[i][0])
            self.info_image.setData(self.info_image.index(i, 1), ar1[i][1])
            self.info_image.setData(self.info_image.index(i, 2), ar1[i][2])

        self.patient_image_file =f'{self.patient_image_file}{self.day_year}{self.day_month}{self.day_day}'
        print(type(self.patient_image_file))


app = QApplication(sys.argv)
w = test()
# k = search()
sys.exit(app.exec_())









