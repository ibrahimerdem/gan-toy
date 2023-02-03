import os
import signal
import queue

from tkinter import *
from tkinter.scrolledtext import ScrolledText
from tkinter import filedialog as fd
from tkinter import messagebox as mb
from tkinter import ttk

from PIL import Image, ImageTk

import dcgan_model


class App:

    def __init__(self, root):
        self.root = root
        self.root.title("Image Work")
        self.root.iconbitmap(os.getcwd() + "\\new.ico")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        left_frame = ttk.Labelframe(self.root, text="Control", width=200, height=200)
        left_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        right_frame = ttk.Labelframe(self.root, text="Console", width=450, height=200)
        right_frame.grid(row=0, column=1, columnspan=3, padx=5, pady=5)

        self.scrolled_text = ScrolledText(right_frame, state="disabled", width=75, height=10.2)
        self.scrolled_text.pack(padx=5, pady=5)
        self.scrolled_text.configure(font="TkFixedFont")

        self.photo_frame = ttk.Labelframe(self.root, text="Input Sample", height=200)
        self.photo_frame.grid(row=1, column=0, columnspan=4, padx=5, pady=5, sticky="nsew")

        self.example_size = 4
        for r in range(self.example_size):
            ttk.Label(self.photo_frame, text="Input", name="input"+str(r)).pack(side="left", expand=1, padx=5, pady=5)

        self.sample_frame = ttk.Labelframe(self.root, text="Output Sample", height=200)
        self.sample_frame.grid(row=2, column=0, columnspan=4, padx=5, pady=5, sticky="nsew")

        for r in range(self.example_size):
            ttk.Label(self.sample_frame, text="Output", name="output"+str(r)).pack(side="left",
                                                                                   expand=1,
                                                                                   padx=5,
                                                                                   pady=5)

        self.info_frame = ttk.Labelframe(self.root, text="Info", height=30)
        self.info_frame.grid(row=3, column=0, columnspan=4, padx=5, pady=5, sticky="nsew")

        btn_select = ttk.Button(left_frame, text="Select Input", width=15, command=self.select_input)
        btn_select.pack()

        btn_start = ttk.Button(left_frame, text="Start", width=15, command=self.start_modeling)
        btn_start.pack()

        btn_stop = ttk.Button(left_frame, text="Stop", width=15, command=self.interrupt_modeling)
        btn_stop.pack()

        btn_resume = ttk.Button(left_frame, text="Resume", width=15, command=self.resume_model)
        btn_resume.pack(anchor="s")

        self.queue = queue.Queue()
        self.report = queue.Queue()
        self.dcgan = dcgan_model.Dcgan(None)

        self.info = StringVar()
        self.info.set("Not started")
        i = ttk.Label(self.info_frame, textvariable=self.info)
        i.grid(column=0, row=1, sticky="w")

        self.files = 0

        self.root.protocol('WM_DELETE_WINDOW', self.quit)
        self.root.bind('<Control-q>', self.quit)
        signal.signal(signal.SIGINT, self.quit)

    def select_input(self):
        self.root.after(100, self.process_queue)
        filetypes = (("image files", '*.png'), ("image files", '*.jpg'),)
        f = fd.askopenfilenames(title="Select Images",
                                filetypes=filetypes)
        try:
            assert f
            self.files = f
            self.info.set("Selected")
            self.queue.put("Selected Files | " + str(f))
            self.root.after(100, self.process_queue)
        except AssertionError:
            self.queue.put("Nothing Selected")

    def start_modeling(self):
        self.root.after(100, self.process_queue)
        try:
            assert self.files
            res = mb.askquestion(
                "Continue", "Max epoch: {}, Batch size: {} \nOld checkpoints will be deleted!".format("20.000", 16))
            if res == "yes":
                self.dcgan = dcgan_model.Dcgan(self.report)
                self.root.after(100, self.process_report)
                self.dcgan.make_dataset(self.files)
                self.dcgan.start()
                self.info.set("Training")
            else:
                pass
        except AssertionError:
            self.queue.put("Nothing Selected")

    def interrupt_modeling(self):
        if hasattr(self.dcgan, "reporter"):
            self.root.after(100, self.process_report)
            if not self.dcgan.pause:
                res = mb.askquestion(
                    "Continue", "Training will be terminated!")
                if res == "yes":
                    self.dcgan.pause = 1
                    self.info.set("Terminated")
                    return 1
                else:
                    return 0
            else:
                return 1
        else:
            return 1

    def resume_model(self):
        if hasattr(self.dcgan, "reporter"):
            self.dcgan.restore_model()
        else:
            self.dcgan = dcgan_model.Dcgan(self.report)
            self.dcgan.restore_model()

    def process_report(self):
        reporter = self.dcgan.reporter
        try:
            while True:
                rep = reporter.get_nowait()
                self.scrolled_text.configure(state="normal")
                self.scrolled_text.insert(END, rep["text"] + '\n')
                self.scrolled_text.configure(state="disabled")
                self.scrolled_text.yview(END)

                if rep.get("sample_arr"):
                    for e in range(min(self.example_size, len(rep.get("sample_arr")))):
                        img = Image.open(rep.get("sample_arr")[e])
                        img = img.resize((128, 128))
                        img = ImageTk.PhotoImage(img)
                        l_name = "input" + str(e)
                        self.photo_frame.children.get(l_name)["image"] = img
                        self.photo_frame.children.get(l_name).image = img

                example_outputs = rep.get("image_list")
                if rep.get("image_list"):
                    for e in range(self.example_size):
                        img = Image.fromarray(example_outputs[e])
                        img = img.resize((128, 128))
                        img = ImageTk.PhotoImage(img)
                        l_name = "output" + str(e)
                        self.sample_frame.children.get(l_name)["image"] = img
                        self.sample_frame.children.get(l_name).image = img
        except queue.Empty:
            self.root.after(100, self.process_report)

    def process_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                self.scrolled_text.configure(state="normal")
                self.scrolled_text.insert(END, msg + '\n')
                self.scrolled_text.configure(state="disabled")
                self.scrolled_text.yview(END)
        except queue.Empty:
            self.root.after(100, self.process_queue)

    def quit(self):
        res = self.interrupt_modeling()
        if res:
            self.root.destroy()


def gui():
    root = Tk()
    app = App(root)
    app.root.mainloop()
