from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import*
from album import Ui_MainWindow as albumpage
import sys
import pandas as pd
import numpy as np
import time

isstateList = []
istimeList = []
f = open('/Users/rui/desktop/label.csv','r')
ff = open('/Users/rui/desktop/timeplace.csv','r')
res = f.readlines()[1:]
ress = ff.readlines()[1:]
for i in res:
    i_str = i.replace('\n','')
    ilist = i_str.split(',')
    isstateList.append(ilist)

for i in ress:
    i_str = i.replace('\n','')
    ilistt = i_str.split(',')
    istimeList.append(ilistt)
df = pd.DataFrame(istimeList)
df.columns = ['timestamp', 'time','label', 'score1', 'place1', 'score2', 'place2', 'score3', 'place3','score4','place4','score5','place5']
dftime = df['timestamp']
dft = df['time']
dflabel = df['label']
dftime_label = df[['timestamp','label']]
dft_label = np.array(dftime_label)
dft1 = np.array(dft)
listtime = dft1.tolist()
dfl = np.array(dflabel)
dt = np.array(dftime)
tstamplist = dt.tolist()
tstamplist1 = dt.tolist()
labeL = dfl.tolist()
time_label = dft_label.tolist()
dffultime = np.array(df)#np.ndarray()
timelist = dffultime.tolist()
dftimelist = dftime.tolist()
labeltimelist = dftime.tolist()

df1 = pd.DataFrame(isstateList)
df1.columns = ['labelnumber', 'labelname','statenumber']
df1['labelnumber'] = df1['labelnumber'].astype('int')
#df1 = df1.set_index('labelnumber')
#df1 = df1.sort_index(ascending= True)

#output star label find_data is series findlist is list
find_data = df1[df1['statenumber']=='2']['labelname']
find_num = df1[df1['statenumber']=='2']['labelnumber']
find_nordata = df1[df1['statenumber']!='0']['labelname']
finda_nornum = df1[df1['statenumber']!='0']['statenumber']


ElseList=[]
findlist = find_data.tolist()
findnum = find_num.tolist()
norlist = find_nordata.tolist()
numnorlist = finda_nornum.tolist()

#clickable label
class myLabel(QLabel):
    clicked = pyqtSignal()
    def mouseReleaseEvent(self, QMouseEvent):
        if QMouseEvent.button() == Qt.LeftButton:
            self.clicked.emit()

class MyAlbumpage(QMainWindow, albumpage):
    def __init__(self, parent=None):
        super(MyAlbumpage, self).__init__(parent)
        self.setupUi(self)
        #调day/week/month那个
     #   self.comboBox.currentIndexChanged.connect(self.selectionchange)
        self.dateTimeEdit.setDate(QDate.currentDate())
       # sBeginDate = self.dateEdit.date().toString(Qt.ISODate)
     #   print(sBeginDate)
      #  self.dateTimeEdit.dateChanged.connect(self.datechange)

        self.label_7.setText('DAY')
        self.label_5.setText(self.dateTimeEdit.date().toString(Qt.ISODate))
        self.label_6.setText(self.dateTimeEdit.date().addDays(1).toString(Qt.ISODate))
        self.pushButton_11.clicked.connect(lambda: self.datechange())

        self.timel = tstamplist
        r = 0
        #生成图片的按钮
        for item in tstamplist:
            self.timel[r] = QPushButton(self.centralwidget)
          #  self.timel[r].setFixedSize(150, 10)
            self.timel[r].setText(str(listtime[r]))
            self.gridLayout_2.addWidget(self.timel[r], r, 0, 1, 1)

            self.timel[r].clicked.connect(lambda: self.ts_button_click(self.sender().text()))
            r = r + 1

      #  self.pushButton_11.clicked.connect(self.submit())
    def ts_button_click(self,timestamp):
        print(timestamp)
        a = str(timestamp)
        timeArray = time.strptime(a, "%Y-%m-%d %H:%M:%S")
        tStamp = int(time.mktime(timeArray))
        path = '/Users/rui/PycharmProjects/test1/Final/video/TLC00007/' + a + '.jpg'
        path1 = '/Users/rui/PycharmProjects/test1/Final/video/Feature/' + str(tStamp) + '-0.jpg'
        print(path)
        print(path1)
        jpgtest = QtGui.QPixmap(path)
        jpgF = QtGui.QPixmap(path1)
        self.label.setPixmap(jpgtest)
        self.label_9.setPixmap(jpgF)
      #  fdata = df[df['timestamp'] == timestamp]['labelname','place1','place2','place3','place4','place5']
      #  ft_label = np.array(fdata)
      #  fulllist = ft_label.tolist()
      #  self.pushButton.setText(fulllist[1])



    def datechange(self):

        self.label_7.setText(self.comboBox.currentText())
        ptime = self.comboBox.currentIndex()
        if ptime ==0:
            pt = self.dateTimeEdit.date().addDays(1)
            ptt = self.dateTimeEdit.dateTime().addDays(1)
        elif ptime==1:
            pt = self.dateTimeEdit.date().addDays(7)
            ptt = self.dateTimeEdit.dateTime().addDays(7)
        elif ptime ==2:
            pt = self.dateTimeEdit.date().addMonths(1)
            ptt = self.dateTimeEdit.dateTime().addMonths(1)
        elif ptime ==3:
            pt = self.dateTimeEdit.date().addYears(1)
            ptt = self.dateTimeEdit.dateTime().addYears(1)

       # self.dateEdit.setDisplayFormat("yyyy.MM.dd")
        timeEdit = self.dateTimeEdit.date().toString(Qt.ISODate)
        time1 = self.dateTimeEdit.dateTime().toString(Qt.ISODate)
     #   timeArray = time.strptime(time1, "%Y-%m-%d %H:%M:%S")
        ts1 = self.dateTimeEdit.dateTime().toTime_t()
        timeStamp = int(ts1)
       # timeStamp = int(time.mktime(timeArray))


        timeHou = pt.toString(Qt.ISODate)
        time2 = ptt.toString(Qt.ISODate)
     #   timeArray2 = time.strptime(time2, "%Y-%m-%d %H:%M:%S")
      #  timeStamp2 = int(time.mktime(timeArray2))
        ts2 = ptt.toTime_t()
        timeStamp2 = int(ts2)


        print(timeEdit)
        self.label_5.setText(timeEdit)
        self.label_6.setText(timeHou)
        i = timeStamp
        '''
        while i in range(timeStamp, timeStamp2):
            for item in time_label:
                if i in range(int(item[0]),int(item[0])+10):
                    print("yes!")
                    i = i + 10
        '''
        print(tstamplist1)
        newlist = []
        while i in range(timeStamp,timeStamp2):
            newlist.append(i)
            ran = int((timeStamp2 - timeStamp) / 50)
            i = i + ran
        print(newlist)

        self.nl = newlist

        c = 0
        i = timeStamp
        while i in range(timeStamp, timeStamp2):
            self.nl[c] = QLabel(self.centralwidget)
            self.gridLayout_3.addWidget(self.nl[c], c, 0, 1, 1)

            ran= int((timeStamp2 - timeStamp)/50)
            print(ran)
            print(i)
            b = 0
            a = i
            for a in range(i,i+20):
                #print(a)

                if str(a) in tstamplist1:
                    timeArraya = time.localtime(a)
                    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArraya)
                    print("yes!")
                    pa = '/Users/rui/PycharmProjects/test1/Final/video/TLC00007/' + otherStyleTime + '.jpg'
                    jpga = QtGui.QPixmap(pa)
                    self.nl[c].setPixmap(jpga)
                else:
                    b = b+1
                a = a+1
            if b == 20:
                print("no!")
            i = i + ran
            c = c+1


        #sBeginDate = self.dateEdit.date().toString(Qt.ISODate)
       # print(str(date))

    #useless
    def selectionchange(self, i):

        # 标签用来显示选中的文本
        # currentText()：返回选中选项的文本
        #self.btn1.setText(self.cb.currentText())
        #print('Items in the list are:')
        # 输出选项集合中每个选项的索引与对应的内容
        # count()：返回选项集合中的数目

        for count in range(self.comboBox.count()):
            #print('Item' + str(count) + '=' + self.cb.itemText(count))
            print('current index', i, 'selection changed', self.comboBox.currentText())

            self.label_7.setText(self.comboBox.currentText())

        if i ==0:
            pt = self.dateTimeEdit.date().addDays(1)
            ptt = self.dateTimeEdit.dateTime().addDays(1)
        elif i==1:
            pt = self.dateTimeEdit.date().addDays(7)
            ptt = self.dateTimeEdit.dateTime().addDays(7)
        elif i ==2:
            pt = self.dateTimeEdit.date().addMonths(1)
            ptt = self.dateTimeEdit.dateTime().addMonths(1)
        elif i ==3:
            pt = self.dateTimeEdit.date().addYears(1)
            ptt = self.dateTimeEdit.dateTime().addYears(1)
        timeEdit = self.dateTimeEdit.date().toString(Qt.ISODate)
        time1 = self.dateTimeEdit.dateTime().toString(Qt.ISODate)
        ts1 = self.dateTimeEdit.dateTime().toTime_t()
      #  timeArray = time.strptime(time1, "%Y-%m-%d %H:%M:%S")
    #    timeStamp = int(time.mktime(timeArray))
        timeStamp = int(ts1)

        timeHou = pt.toString(Qt.ISODate)
        time2 = ptt.toString(Qt.ISODate)
      #  timeArray2 = time.strptime(time2, "%Y-%m-%d %H:%M:%S")
        ts2 = ptt.toTime_t()
        timeStamp2 =int(ts2)
      #  timeStamp2 = int(time.mktime(timeArray2))


        '''
        for i in range(timeStamp, timeStamp2):
            for item in time_label:
                if i ==int(item[0]):
                    print("yes!")
                #else:
                   # print("no...")
        '''



       # print(timeEdit)
        self.label_5.setText(timeEdit)
        self.label_6.setText(timeHou)



if __name__ == "__main__":
    #固定的，PyQt5程序都需要QApplication对象。sys.argv是命令行参数列表，确保程序可以双击运行
    app = QApplication(sys.argv)
    #初始化
    myWin = MyAlbumpage()
    #将窗口控件显示在屏幕上
    myWin.show()

    #程序运行，sys.exit方法确保程序完整退出。
    sys.exit(app.exec_())
