import os

from datetime import datetime
main_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

class ModelFile:
    file = "models"

    def __init__(self, m_folder, timestamp=None):
        self.m_folder = m_folder
        self.timestamp = timestamp

    def try_latest_t_folder(self):
        force_mkdir(self.m_folder)
        timestamps = os.listdir(self.m_folder)
        if len(timestamps) == 0:
            return None
        latest_stamp = max(timestamps)
        return os.path.join(self.m_folder, latest_stamp)

    def t_folder(self):
        if self.timestamp is None:
            try_t_folder = self.try_latest_t_folder()
            if try_t_folder is not None:
                return try_t_folder
            self.timestamp = main_timestamp
        return os.path.join(self.m_folder, self.timestamp)

    def e_folder(self):
        return os.path.join(self.t_folder(),"by_epoch")

    def ckpt(self, b4epoch):
        e_folder = self.e_folder()
        force_mkdir(e_folder)
        ckpt = os.path.join(e_folder,"%i.ckpt"%b4epoch)
        return ckpt

    def latest_ckpt(self, max_epoch=None):
        e_folder = self.e_folder()
        force_mkdir(e_folder)
        files = os.listdir(e_folder)
        epochs = [file.split(".",1)[0] for file in files]
        epochs = [int(e) for e in epochs if e.isdigit()]
        if max_epoch is not None:
            epochs = [e for e in epochs if e<=max_epoch]
        if len(epochs) == 0:
            return None
        epoch = max(epochs)
        file = "%i.ckpt" % epoch
        file = os.path.join(self.e_folder(),file)
        return file, epoch

    def train_acc_file(self):
        return os.path.join(self.t_folder(),"train_acc.txt")

    def test_acc_file(self):
        return os.path.join(self.t_folder(),"test_acc.txt")

class ResultFile:
    folder = "result2"

    def __init__(self, r_folder):
        self.r_folder = r_folder
        force_mkdir(self.t_folder())

    def t_folder(self):
        return os.path.join(self.r_folder, main_timestamp)

    def file(self,filename):
        return os.path.join(self.t_folder(),filename)

def force_mkdir(folder):
    if os.path.isdir(folder) or folder == "":
        return
    head = os.path.dirname(folder)
    force_mkdir(head)
    os.mkdir(folder)