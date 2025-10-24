import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin
        self.use_day_index = configs.use_day_index
        self.use_hour_index = configs.use_hour_index
        self.scale = 0.02
        self.emb_len_hour = configs.hour_length
        self.emb_len_day = configs.day_length
        layers = [
                nn.Linear(self.seq_len, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.pred_len)
        ]
        # layers_hour = [
        #         nn.Linear(self.seq_len, self.d_model),
        #         nn.ReLU(),
        #         nn.Linear(self.d_model, self.pred_len)
        # ]
        # layers_day = [
        #         nn.Linear(self.seq_len, self.d_model),
        #         nn.ReLU(),
        #         nn.Linear(self.d_model, self.pred_len)
        # ]
        self.model = nn.Sequential(*layers)
        # self.model_hour = nn.Sequential(*layers_hour)
        # self.model_day = nn.Sequential(*layers_day)

        if self.use_day_index:
            self.emb_hour_re = nn.Parameter(torch.zeros(self.emb_len_day * self.emb_len_hour,
                                                       self.enc_in, self.seq_len // 2 + 1), requires_grad=True)
            self.emb_hour_im = nn.Parameter(torch.zeros(self.emb_len_day * self.emb_len_hour,
                                                       self.enc_in, self.seq_len // 2 + 1), requires_grad=True)
        else:
            self.emb_hour_re = nn.Parameter(torch.zeros(self.emb_len_hour, self.enc_in, self.seq_len // 2 + 1), requires_grad=True)
            self.emb_hour_im = nn.Parameter(torch.zeros(self.emb_len_hour, self.enc_in, self.seq_len // 2 + 1), requires_grad=True)

        # self.emb_hour_re = nn.Parameter(torch.zeros(self.emb_len_hour, self.enc_in, self.seq_len // 2 + 1), requires_grad=True)
        # self.emb_hour_im = nn.Parameter(torch.zeros(self.emb_len_hour, self.enc_in, self.seq_len // 2 + 1), requires_grad=True)
        # self.emb_day_re = nn.Parameter(torch.zeros(self.emb_len_day, self.enc_in, self.seq_len // 2 + 1), requires_grad=True)
        # self.emb_day_im = nn.Parameter(torch.zeros(self.emb_len_day, self.enc_in, self.seq_len // 2 + 1), requires_grad=True)
        self.w = nn.Parameter(self.scale * torch.randn(1, self.seq_len))

    def forward(self, x, hour_index, day_index = None):
        # x: (batch_size, seq_len, enc_in), hour_index: (batch_size,), day_index: (batch_size,)

        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        x = x.permute(0, 2, 1)
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        w = torch.fft.rfft(self.w, dim=1, norm='ortho')
        # x_freq_real = x.real
        # x_freq_imag = x.imag

        if self.use_day_index:
            # print(hour_index, day_index)
            emb_hour_re = self.emb_hour_re[((hour_index * 24 + day_index) % 168).long()]
            emb_hour_im = self.emb_hour_im[((hour_index * 24 + day_index) % 168).long()]
        else:
            emb_hour_re = self.emb_hour_re[(hour_index % self.emb_len_hour).long()]
            emb_hour_im = self.emb_hour_im[(hour_index % self.emb_len_hour).long()]
        emb_hour = torch.complex(emb_hour_re, emb_hour_im)
        x = x - emb_hour

        # if self.use_hour_index:
        #     emb_hour_re = self.emb_hour_re[(hour_index % self.emb_len_hour).long()]
        #     emb_hour_im = self.emb_hour_im[(hour_index % self.emb_len_hour).long()]
        #     emb_hour = torch.complex(emb_hour_re, emb_hour_im)
        #     x = x - emb_hour
        #
        # if self.use_day_index:
        #     emb_day_re = self.emb_day_re[(day_index % self.emb_len_day).long()]
        #     emb_day_im = self.emb_day_im[(day_index % self.emb_len_day).long()]
        #     emb_day = torch.complex(emb_day_re, emb_day_im)
        #     x = x - emb_day

        # x_freq_minus_emb = torch.complex(x_freq_real, x_freq_imag)
        # y = x * w
        # y_real = y.real
        # y_freq_imag = y.imag

        # if self.use_day_index:
        #     y_real = y_real + emb_day
        # if self.use_hour_index:
        #     y_real = y_real + emb_hour

        y_freq = x * w + emb_hour
        y = torch.fft.irfft(y_freq, n=self.seq_len, dim=2, norm="ortho")
        y = self.model(y).permute(0, 2, 1)

        # y_hour = torch.fft.irfft(emb_hour, n=self.seq_len, dim=2, norm="ortho")
        # y_hour = self.model_hour(y_hour).permute(0, 2, 1)
        # y = y + y_hour

        # if self.use_hour_index:
        #     y_hour = torch.fft.irfft(emb_hour, n=self.seq_len, dim=2, norm="ortho")
        #     y_hour = self.model_hour(y_hour).permute(0, 2, 1)
        #     y = y + y_hour
        #
        # if self.use_day_index:
        #     y_day = torch.fft.irfft(emb_day, n=self.seq_len, dim=2, norm="ortho")
        #     y_day = self.model_day(y_day).permute(0, 2, 1)
        #     y = y + y_day

        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y