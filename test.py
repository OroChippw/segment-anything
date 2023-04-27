import torch
class FixAudioLength(object):
    def init (self , time = 2, sample_rate = 16000):
        self.target_len = time * sample_rate
    def call (self , data: torch.Tensor):
        cur_len = data.shape[1]
        if self.target_len <= cur_len:
            data = data[:, :self.target_len]
        else:
            data = torch.nn.functional.pad(data , (0,self.target len - cur len))#data=torch.nn.functional.pad(data.(self.target len - cur len, )return data