import tkinter as tk
from tkinter import StringVar, DISABLED,END,scrolledtext
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
    set_default_color_theme
)

class AutoGensis():
    def __init__(self,master):
        self.master = master
        self.master.title("Video Automation Gensis")

        ##############FRAMES###############
        self.main_frame = tk.Frame(master)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.work_list_frame = tk.Frame(self.main_frame)
        self.work_list_frame.pack(pady=10,padx=10)

        self.AgentUpdate = tk.Frame(self.main_frame)
        self.AgentUpdate.pack(pady=10,padx=10)



        label = tk.Label(self.main_frame,text="Welcome to Automation Gensis")
        label.pack(pady=10, padx=10)


        ############### Add Videos to WORK#################
        self.add_video = tk.Button(self.main_frame, text="Add Video to list", command=self.Add_Video)
        self.add_video.pack(pady=10, padx=10)


        ############ START WORK/PROGRAM################
        self.Start_engine = tk.Button(self.main_frame, text="Run Engine", command=self.run_engine)
        self.Start_engine.pack(pady=10,padx=10)


        ############ LIST OF VIDEOS UNDER CREATION ###################
        self.worklist_label = tk.Label(self.work_list_frame, text="Videos in progress!")
        self.worklist_label.pack(pady=10,padx=10)
        self.Work_list = tk.Listbox(self.work_list_frame)
        self.Work_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)



        #####AGENT PROGRESS LOG WINDOW##############
        self.agent_label = tk.Label(self.AgentUpdate, text="AgentRun [LOG]")
        self.agent_label.pack(pady=10,padx=10)
        self.agent_log = tk.Text(self.AgentUpdate, height=10, wrap=tk.WORD)
        self.agent_log.pack(fill="both",expand=True, padx=10, pady=5)
        self.agent_log.config(state=DISABLED)





    def Add_Video(self):
        return
    



    def run_engine(self):
        return
    
    
    




if __name__ == "__main__":  
    root = tk.Tk()
    app = AutoGensis(root)
    root.mainloop()

