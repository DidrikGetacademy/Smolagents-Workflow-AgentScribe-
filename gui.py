import tkinter as tk
from tkinter import StringVar, DISABLED,END,scrolledtext,filedialog
import platform
import threading
from App import (
    clean_get_gpu_memory,
    Clean_log_onRun,
    video_creation_worker,
    
)
import os
from customtkinter import (
    CTk,
    CTkButton,
    CTkFrame,
    CTkComboBox,
    CTkCheckBox,
    CTkSlider,
    CTkEntry,
    CTkFont,
    CTkImage,
    CTkLabel,
    CTkOptionMenu,
    CTkScrollableFrame,
    CTkToplevel,
    filedialog,
    CTkTextbox,
    set_appearance_mode,
    set_default_color_theme,
)
import torch


class AutoGensis():
    def __init__(self,master):
        self.master = master
        self.master.title("Video Automation Gensis")
        self.Uploaded_videos_tolist = []

        ##############FRAMES###############
        self.main_frame = tk.Frame(master)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.work_list_frame = tk.Frame(self.main_frame)
        self.work_list_frame.pack(pady=10,padx=10)

        self.AgentUpdate = tk.Frame(self.main_frame)
        self.AgentUpdate.pack(pady=10,padx=10)



        Welcome_label = tk.Label(self.main_frame,text="Welcome to Automation Gensis")
        Welcome_label.pack(pady=10, padx=10)
        GPU,CPU = self.Get_DeviceMap()
        Device_information = tk.Label(self.main_frame, text=f"GPU detected: {GPU} - CPU detected: {CPU}")
        


        ############### Add Videos to WORK#################
        self.Video_Amount = tk.Label(self.main_frame, text=f"VideoFiles added ({len(self.Uploaded_videos_tolist)})")
        self.Video_Amount.pack(pady=12, padx=10)
        self.add_video = tk.Button(self.main_frame, text="Add Video to list", command=self.Add_Video_tolist)
        self.add_video.pack(pady=10, padx=10)


        ############ START WORK/PROGRAM ################
        self.Start_engine = tk.Button(self.main_frame, text="Run Engine", command=self.run_engine)
        self.Start_engine.pack(pady=10,padx=10)

        self.Video_list = tk.Listbox(self.work_list_frame, width=50, height=15)
        self.Video_list.pack(pady=10)


        ############ LIST OF VIDEOS UNDER CREATION ###################
        self.worklist_label = tk.Label(self.work_list_frame, text="Videos in progress!")
        self.worklist_label.pack(pady=10,padx=5)
        self.Work_list = tk.Listbox(self.work_list_frame)
        self.Work_list.pack(fill=tk.BOTH, expand=False, padx=5, pady=10)



        #####AGENT PROGRESS LOG WINDOW##############
        self.agent_log_display = scrolledtext.ScrolledText(
            self.AgentUpdate,
            Wrap=tk.WORD,
            width=55,
            height=25,
            font=("Helvetica",12),
            bg="black",  
            fg="white",
            state="disabled",
            )
        self.agent_log_display.config(  
            insertbackground="yellow",
            selectbackground="#444444",
            selectforeground="white",
            borderwidth=2,
            relief="sunken"
        )
        self.agent_log_display.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.agent_log_display.yview(END)
        self.agent_log_display.columnconfigure(0, weight=1)
        self.agent_log_display.rowconfigure(1, weight=1)


    def Get_DeviceMap(self):
        if torch.cuda.is_available():
                GPU = torch.cuda.get_device_name(0)
        else:
            GPU = "NO device detected"
        
        CPU = platform.processor()
        return GPU, CPU

    def Log_to_GUI(self, Log_message: str):
        self.agent_log_display.configure(state="NORMAL")
        self.agent_log_display.insert(tk.END, Log_message)
        self.agent_log_display.configure(state="disabled")
        self.agent_log_display.update()
        print(f"Updated GUI with message: {Log_message} :)\n")
       



    def Add_Video_tolist(self):
        file_path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=(("Video files","*.mp4 *.avi *.mkv *.mov"),)
        )
        if file_path:
            file_name = os.path.basename(file_path)
            print(file_name)
            self.Video_list.insert(tk.END,file_name)
            self.Uploaded_videos_tolist.append(file_path)
            print(f"File added to list: {file_path}")
        else:
            print("No selected file")
        
    


    


    def run_engine(self):
        clean_get_gpu_memory()
        Clean_log_onRun()
        try:
            worker_thread = threading.Thread(target=video_creation_worker,name="Video_creation(THREAD)")
            worker_thread.start()
            self.Log_to_GUI("Worker Thread - Status: Online")
        except Exception as e:
            self.Log_to_GUI("Worker Thread - Status: Offline")
        return
    
    
    




if __name__ == "__main__":  
    root = tk.Tk()
    app = AutoGensis(root)
    root.mainloop()

