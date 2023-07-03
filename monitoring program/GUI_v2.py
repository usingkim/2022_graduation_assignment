import sys
import os
import os.path
from os import path
import csv
import MakeGraph
import data_preprocessing
import crop_image
from PIL.ImageQt import ImageQt
from PIL import Image, ImageEnhance
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtWidgets import QDialog
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, uic
from PyQt5.QtCore import QSize, Qt, QCoreApplication
from PyQt5.Qt import *

# UI파일 연결
# 단, UI파일은 Python 코드 파일과 같은 디렉토리에 위치해야한다.
form_class = uic.loadUiType("GUI.ui")[0]

# 화면을 띄우는데 사용되는 Class 선언
class WindowClass(QMainWindow,form_class) :
    dialog = None
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        self.scale_factor = 3
        
        # 버튼에 기능 연결
        self.open_btn.clicked.connect(self.open_folder)
        self.exec_btn.clicked.connect(self.execute_file)
        self.quit_btn.clicked.connect(QCoreApplication.instance().quit)
        
        # label style
        self.origin_img_view.setStyleSheet("color: #808080; border-style: solid; border-width: 2px; border-color: #808080; border-radius: 10px; ")
        self.pre_img_view.setStyleSheet("color: #808080; border-style: solid; border-width: 2px; border-color: #808080; border-radius: 10px; ")
        self.result_view.setStyleSheet("color: #808080; border-style: solid; border-width: 2px; border-color: #808080; border-radius: 10px; ")
        self.IMG_list_widget.setStyleSheet("color: #808080; border-style: solid; border-width: 3px; border-color: #808080; border-radius: 10px; ")
        self.defect_view_widget.setStyleSheet("color: #808080; border-style: solid; border-width: 3px; border-color: #808080; border-radius: 10px; ")
        
        # 원본 이미지 출력
        self.IMG_list_widget.itemClicked.connect(self.showOrigin_Img)
        self.execute_state = False
        
        # 실행하기 버튼을 누른 후 전처리 이미지가 출력된다.
        self.IMG_list_widget.itemClicked.connect(self.showPrefilter_Img)
        
        # 결함 목록 출력
        self.defect_view_widget.itemClicked.connect(self.new_window_open)
        self.defect_view_widget.itemClicked.connect(self.show_bounding)

    # 실행하기
    def execute_file(self):
        MakeGraph.main()
        self.show_result_graph()
        #crop_image.main()
        #data_preprocessing.main()
        self.execute_state = True
        
    
    # 파일 불러오기
    def open_folder(self):
        self.folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        self.show_files_in_folder(self.folder) # 폴더 내 파일 목록을 표시합니다.
    
    # 원본 이미지를 listview에 추가
    def show_files_in_folder(self, folder_path):
        # 폴더 내 파일 목록을 불러옵니다.
        self.img_path = str(folder_path) + '/train_images/'
        TRAIN_IMGS_PATH = os.listdir(self.img_path)
        TRAIN_IMGS_PATH.sort()
        # 기존에 표시된 리스트를 지웁니다.
        self.IMG_list_widget.clear()
        for img_name in TRAIN_IMGS_PATH:
            # QListWidgetItem 객체 생성
            item = QListWidgetItem(img_name)
            # QListWidget에 QListWidgetItem 객체를 추가
            self.IMG_list_widget.addItem(item) 
    
    # defect를 listview에 추가
    def defect_add(self, folder_path):
        Defect_img_path = os.listdir(folder_path)
        Defect_img_path.sort()
        self.defect_view_widget.clear()
        for defect in Defect_img_path:
            item = QListWidgetItem(defect)
            self.defect_view_widget.addItem(item)
    
    # 원본이미지를 화면에 불러오기
    def showOrigin_Img(self, item):
        self.origin_img_name = item.text()
        self.origin_img_path = str(self.folder) + '/train_images/' + self.origin_img_name
        self.origin_pixmap = QPixmap(self.origin_img_path)
        self.origin_label = self.origin_img_view
        self.origin_img_view.setPixmap(self.origin_pixmap.scaled(self.origin_label.width(),self.origin_label.height(),aspectRatioMode=Qt.KeepAspectRatio))
    
    # SteelDefect.py를 실행하여 전처리 이미지가 담긴 폴더를 저장하고 파일 경로에 연결하여 출력
    def showPrefilter_Img(self, item):
        if self.execute_state == True:
            self.pre_filter_img_name = item.text()
            self.pre_filter_img_path = str(self.folder) + '/boundingbox_images/' + self.pre_filter_img_name
            if path.exists(self.pre_filter_img_path):
                self.pre_pixmap = QPixmap(self.pre_filter_img_path)
                self.pre_label = self.pre_img_view
                self.pre_label.setAlignment(Qt.AlignCenter)
                
                # 전처리 된 이미지를 두번째 label에 출력
                self.pre_result_pixmap = self.pre_pixmap.scaled(self.pre_label.width(),self.pre_label.height(),aspectRatioMode=Qt.KeepAspectRatio)
                self.pre_img_view.setPixmap(self.pre_result_pixmap)
                
                # 결함 목록에 결함 이미지 추가
                self.defect_img_path = str(self.folder) + '/result/' + self.pre_filter_img_name[:-4]
                self.defect_add(self.defect_img_path)
            else:
                self.pre_img_view.clear()
                self.defect_view_widget.clear()
                self.pre_img_view.setText("No defect")
                font = self.pre_img_view.font()
                font.setPointSize(16)
                self.pre_img_view.setFont(font)
                self.defect_view_widget.addItem("No defect")
            
            
    # bounding box 표시
    # 현재 pixmap 크기 : width : 831, height: 132
    def show_bounding(self, item):
        if self.execute_state == True:
            # 새로운 pixmap 생성
            result_pixmap = QPixmap(self.pre_result_pixmap)
            
            # 기존의 bounding box 제거
            painter = QPainter(result_pixmap)
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.setPen(Qt.transparent)
            painter.drawRect(0, 0, self.pre_result_pixmap.width(), self.pre_result_pixmap.height())
            
            bounding_image = item.text().replace("png","jpg")
            cord_list = defect_class.get_cord(bounding_image)
            source_width = 1600
            source_height = 256
            x, y = cord_list[0], cord_list[1]
            width, height = cord_list[2], cord_list[3]
            
            # 좌표값을 비율에 따라 변환
            x_ratio = result_pixmap.width() / source_width
            y_ratio = result_pixmap.height() / source_height
            mapped_x = int(x * x_ratio)
            mapped_y = int(y * y_ratio)
            mapped_width = int(width * x_ratio)
            mapped_height = int(height * y_ratio)
            
            # 새로운 bounding box 그리기
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.setPen(QPen(Qt.blue,3))
            painter.drawRect(mapped_x, mapped_y, mapped_width, mapped_height)
            painter.end()
            
            # 업데이트된 pixmap으로 설정
            self.pre_img_view.setPixmap(result_pixmap)
    
    # 이미지 확대
    def enlarge_image(self, image):
        width = image.width() * self.scale_factor
        height = image.height() * self.scale_factor
        return image.scaled(width, height)  
    
    # 이미지 해상도 개선
    def improve_resolution(self, image):
        improved_image = QImage(image.size(), QImage.Format_ARGB32)
        painter = QPainter(improved_image)
        painter.drawImage(0, 0, image)
        painter.end()
        return improved_image

    # 이미지 크기 알맞게 조정
    def resize_image(self,img,target_width,target_height):
        width, height = img.size
        aspect_ratio = width / float(height)
        if target_height / float(target_height) > aspect_ratio:
            new_width = int(target_height * aspect_ratio)
            new_height = target_height 
        else:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        resized_image = img.resize((new_width, new_height), Image.LANCZOS)
        x = (target_width - new_width) // 2
        y = (target_height - new_height) // 2
        new_image = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))
        new_image.paste(resized_image, (x, y, x + new_width, y + new_height))
        return new_image
    
    # 새로운 창 열기    
    def new_window_open(self,item):
        self.dialog = QDialog()
        self.dialog.setWindowTitle("zoom_in window")
        self.dialog.setWindowModality(Qt.ApplicationModal)
        self.dialog.resize(400,400)
        self.dialog.setGeometry(self.width(), self.y(), 300, 200)

        if self.execute_state == True:
            self.defect_img_name = item.text()
            self.show_defect_img_path = self.defect_img_path + '/'+ self.defect_img_name
            
            # 해상도 조절, zoon-in
            defect_img = QImage(self.show_defect_img_path)
            improved_image = self.improve_resolution(defect_img)
            zoom_in_image = self.enlarge_image(improved_image)
            
            self.zoom_in_label = QLabel(self.dialog)
            self.zoom_in_Pixmap = QPixmap.fromImage(zoom_in_image)
            self.zoom_in_label.setPixmap(self.zoom_in_Pixmap)
            self.zoom_in_label.setContentsMargins(10, 10, 10, 10)
            self.zoom_in_label.resize(self.zoom_in_Pixmap.width(), self.zoom_in_Pixmap.height())
            self.zoom_in_label.setScaledContents(True)
            
            # 결함 종류 출력
            defect_type = int(self.defect_img_name.split("_")[1])
            defect_list = ["pitted", "inclusion", "scratches", "patches"]
            text_label = QLabel("결함 종류 : {}".format(defect_list[defect_type-1]),self.dialog)
            text_label.setAlignment(Qt.AlignCenter)
            
            layout = QVBoxLayout(self.dialog)
            layout.addWidget(self.zoom_in_label)
            layout.addWidget(text_label)
            self.dialog.show()
            
    # 결함 분포 출력
    def show_result_graph(self):
        self.graph_path = "graph.png"
        graph_image = QImage(self.graph_path)
        graph = self.improve_resolution(graph_image)
        graph_pixmap = QPixmap.fromImage(graph)
        self.result_view.setPixmap(graph_pixmap)
        self.result_view.setScaledContents(True)


     
class Defectcord:
#defect_dict[][0]: classId, [1]: class1Cord [2]: class2Cord [3]: class3Cord [4]: class4Cord    
    def __init__(self, filename = "bounding_box.csv"):
        self.__top100__ = self.__csv_to_dict__(filename)
        
    def __csv_to_dict__(self,filename):
        data = {}
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            headers = next(reader)  # 첫 번째 행을 헤더로 설정
            for row in reader:
                key = row[0]  # 첫 번째 열을 키로 설정
                data[key] = row[1:]  # 나머지 열들을 아이템으로 설정
        return data
    
    # 좌표값 리스트에 저장하여 반환
    def get_cord(self,jpg_name):
        select = jpg_name.split('_')
        defect_index = int(select[2][0])
        image = select[0]+'.jpg'
        if select[1] == '1':
            cordlist =  self.get_class1_cord(image)
            return cordlist[defect_index]
            
        elif select[1] == '2':
            cordlist =  self.get_class2_cord(image)
            return cordlist[defect_index]
           
        elif select[1] == '3':
            cordlist =  self.get_class3_cord(image)
            return cordlist[defect_index]
            
        elif select[1] == '4':
            cordlist =  self.get_class4_cord(image)
            return cordlist[defect_index]
            
    def get_class1_cord(self,jpg_name):
        cord1list = []
        class1cord = self.__top100__[jpg_name][1]
        class1cord = class1cord[1:-1]
        class1cord = class1cord.replace("\'","")
        for substring in class1cord.split(','):
            numbers = [int(num) for num in substring.strip().split()]
            cord1list.append(numbers)
        return cord1list
    
    def get_class2_cord(self,jpg_name):
        cord2list = []
        class2cord = self.__top100__[jpg_name][2]
        class2cord = class2cord[1:-1]
        class2cord = class2cord.replace("\'","")
        for substring in class2cord.split(','):
            numbers = [int(num) for num in substring.strip().split()]
            cord2list.append(numbers)
        return cord2list
    
    def get_class3_cord(self,jpg_name):
        cord3list = []
        class3cord = self.__top100__[jpg_name][3]
        class3cord = class3cord[1:-1]
        class3cord = class3cord.replace("\'","")
        for substring in class3cord.split(','):
            numbers = [int(num) for num in substring.strip().split()]
            cord3list.append(numbers)
        return cord3list
    
    def get_class4_cord(self,jpg_name):
        cord4list = []
        class4cord = self.__top100__[jpg_name][4]
        class4cord = class4cord[1:-1]
        class4cord = class4cord.replace("\'","")
        for substring in class4cord.split(','):
            numbers = [int(num) for num in substring.strip().split()]
            cord4list.append(numbers)
        return cord4list
        
        
if __name__ == "__main__" :
    # QApplication : 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv)
    # WindowClass의 인스턴스 생성
    myWindow = WindowClass() 
    # Defectcord의 인스턴스 생성
    defect_class = Defectcord()
    # 프로그램 화면을 보여주는 코드
    myWindow.show()
    # 프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
    app.exec_()