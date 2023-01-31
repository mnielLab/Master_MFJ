# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 12:09:39 2022

@author: Mathias
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as N

class BiLSTM(nn.Module):
    def __init__(self, max_length = 24):
        super().__init__()
        self.BiLSTM = nn.LSTM(input_size = 20, 
                               hidden_size = max_length,
                               batch_first = True,
                               bidirectional = True)
        
    def forward(self, x):
        _, (out, _) = self.BiLSTM(x)
        return out

class CDR3b_BiLSTM(nn.Module):
    def __init__(self, max_length = 24, hidden_size = 64, dropout_rate = 0.5):
        super().__init__()
        self.BiLSTM = (BiLSTM())
        
        self.linear = nn.Linear(in_features = max_length*2*2,
                                out_features = hidden_size)
        
        self.out = nn.Linear(in_features = hidden_size,
                             out_features = 1)
        
        dropout_rate = float(dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        if dropout_rate != 0:
            self.dropout_flag = True
        else:
            self.dropout_flag = False

        
        
    def forward(self, pep, cdr3b):
        BiLSTM_pep = self.BiLSTM(pep)
        BiLSTM_cdr3b = self.BiLSTM(cdr3b)
    
        cat = torch.cat([BiLSTM_pep, 
                         BiLSTM_cdr3b], 2)
        
        #Concatenate forward and reverse LSTM
        cat = torch.cat([cat[0], cat[1]], 1)
        
        if self.dropout_flag:
            cat = self.dropout(cat)
        
        hid = torch.relu(self.linear(cat))
        #if self.dropout_flag:
        #    hid = self.dropout(hid)
        out = torch.sigmoid(self.out(hid))
        
        return out

class CDR3_BiLSTM(nn.Module):
    def __init__(self, max_length = 24, hidden_size = 64, dropout_rate = 0.5):
        super().__init__()
        self.BiLSTM = (BiLSTM())
        
        self.linear = nn.Linear(in_features = max_length*3*2,
                                out_features = hidden_size)
        
        self.out = nn.Linear(in_features = hidden_size,
                             out_features = 1)
        
        dropout_rate = float(dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        if dropout_rate != 0:
            self.dropout_flag = True
        else:
            self.dropout_flag = False

        
        
    def forward(self, pep, cdr3a, cdr3b):
        BiLSTM_pep = self.BiLSTM(pep)
        BiLSTM_cdr3a = self.BiLSTM(cdr3a)
        BiLSTM_cdr3b = self.BiLSTM(cdr3b)
    
        cat = torch.cat([BiLSTM_pep, 
                         BiLSTM_cdr3a, 
                         BiLSTM_cdr3b], 2)
        
        #Concatenate forward and reverse LSTM
        cat = torch.cat([cat[0], cat[1]], 1)
        
        if self.dropout_flag:
            cat = self.dropout(cat)
        
        hid = torch.relu(self.linear(cat))
        #if self.dropout_flag:
        #    hid = self.dropout(hid)
        out = torch.sigmoid(self.out(hid))
        
        return out
    
class CDR123_BiLSTM(nn.Module):
    def __init__(self, max_length = 24, hidden_size = 64, dropout_rate = 0.5):
        super().__init__()
        self.BiLSTM = (BiLSTM())
        
        self.linear = nn.Linear(in_features = max_length*7*2,
                                out_features = hidden_size)
        
        self.out = nn.Linear(in_features = hidden_size,
                             out_features = 1)
        
        dropout_rate = float(dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        if dropout_rate != 0:
            self.dropout_flag = True
        else:
            self.dropout_flag = False
  
    def forward(self, pep, cdr1a, cdr2a, cdr3a, cdr1b, cdr2b, cdr3b):
        BiLSTM_pep = self.BiLSTM(pep)
        BiLSTM_cdr1a = self.BiLSTM(cdr1a)
        BiLSTM_cdr2a = self.BiLSTM(cdr2a)
        BiLSTM_cdr3a = self.BiLSTM(cdr3a)
        BiLSTM_cdr1b = self.BiLSTM(cdr1b)
        BiLSTM_cdr2b = self.BiLSTM(cdr2b)
        BiLSTM_cdr3b = self.BiLSTM(cdr3b)
    
        cat = torch.cat([BiLSTM_pep, 
                         BiLSTM_cdr1a, BiLSTM_cdr2a, BiLSTM_cdr3a, 
                         BiLSTM_cdr1b, BiLSTM_cdr2b, BiLSTM_cdr3b], 2)
        
        #Concatenate forward and reverse LSTM
        cat = torch.cat([cat[0], cat[1]], 1)
        
        if self.dropout_flag:
            cat = self.dropout(cat)
        
        hid = torch.relu(self.linear(cat))
        #if self.dropout_flag:
        #    hid = self.dropout(hid)
        out = torch.sigmoid(self.out(hid))
        
        return out
