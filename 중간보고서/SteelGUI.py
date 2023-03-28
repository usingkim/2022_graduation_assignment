from tkinter import *
from tkinter import messagebox
import os
import tkinter
import tkinter.font
import SteelDefect


tk = Tk()
tk.title('SteelDefect GUI')
font = tkinter.font.Font(family = "맑은 고딕", size = 10,slant = "roman",weight="bold")

caution_label = Label(tk, text = "폴더이름을 입력해주세요.",font=font)
caution_label.grid(row=0,column=0,columnspan=3,pady=20)

address_label = Label(tk,text="폴더 이름",width=10,font=("나눔바른펜",15))
address_label.grid(row=1,column=0)

# #entry : input()느낌
address_ent = Entry(tk,width = 30)
address_ent.grid(row=1,column=1)

result_label = Label(tk,wraplength=300)
result_label.grid(column=0,row=3,columnspan=3)

def start_code():
    SteelDefect.defect_exec(address_ent.get() + '\\')
    
def event_input_btn():
    #address = ent.get()
    global btn_input_state
    read_address = address_ent.get()
    result_label.config(text = f"{read_address} entered.\n")
    if btn_input_state==False:
        btn_confirm = Button(tk,fg='white',bg='pink',width=10,text = '확인',command = event_btn1_btn1)
        btn_confirm.grid(column=1,row=2,pady=20)
        btn_input_state = True
   
def event_btn1_btn1():
    read_address = address_ent.get()
    if os.path.isdir(read_address):
        btn_execute = Button(tk,fg='white',bg='pink',width=10,text = '실행하기',command = start_code)
        btn_execute.grid(column=2,row=2,pady=20)
        messagebox.showinfo(title="파일 확인 완료", message= "파일을 확인하였습니다.")
        result_label.config(text = "실행버튼을 눌러주세요.\n")

    else:
        messagebox.showinfo(title="파일 오류", message= "잘못된 파일을 입력하였습니다.")
        address_ent.delete(0,END)
        result_label.config(text = "다시 한번 입력해주세요.\n")

btn_input = Button(tk,fg='white',bg='pink',width = 10,text = '파일 입력',command = event_input_btn)
btn_input.grid(column=0,row=2,pady=20)
btn_input_state = False

tk.mainloop()