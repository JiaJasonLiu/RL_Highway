from torch.utils.tensorboard import SummaryWriter
import os
import datetime

class Metrics:
    def __init__(self, policy, result_file_name, use_metrics):
        self.use_metrics = use_metrics
        if not self.use_metrics:
            return

        self.create_folder(result_file_name)
        
        time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        new_num = str(len(os.listdir("./" +result_file_name)) + 1)
        file_name = f'{result_file_name}/{policy}_{new_num}_{time}'
        self.writer = SummaryWriter(log_dir=file_name, flush_secs=60)
            
    def add(self, type, y, x):
        if not self.use_metrics:
            return
        
        self.writer.add_scalar(type, y, x)
    
    def save_params(self, params):
        if not self.use_metrics:
            return

        self.writer.add_text(
            "Hyperparameter",
            "Param | Value |\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in params.items()])),
            )

    def close(self):
        if not self.use_metrics:
            return
        self.writer.close()
        
    def create_folder(self, directory_name):
        try:
            os.mkdir(directory_name)
            print(f"Directory '{directory_name}' created successfully.")
        except FileExistsError:
            return
        except PermissionError:
            print(f"Permission denied: Unable to create '{directory_name}'.")
        except Exception as e:
            print(f"An error occurred: {e}")