
# File contains network models that is based on Re3: Real-time recurrent regression
# network for visual tracking for generic objects.
# Some parts of the code were from DanielGordon's PyTorch implementation of the paper
# CaffeLSTMCell class is from https://github.com/danielgordon10/re3-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from skimage import io
from torchvision import transforms



class CaffeLSTMCell(nn.Module):

    def __init__(self, input_size, output_size):
        super(CaffeLSTMCell, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.block_input = nn.Linear(input_size + output_size, output_size)
        self.input_gate = nn.Linear(input_size + output_size * 2, output_size)
        self.forget_gate = nn.Linear(input_size + output_size * 2, output_size)
        self.output_gate = nn.Linear(input_size + output_size * 2, output_size)

    def forward(self, inputs, hx=None):
        if hx is None or (hx[0] is None and hx[1] is None):
            zeros = torch.zeros(inputs.size(0), self.output_size, dtype=inputs.dtype, device=inputs.device)
            hx = (zeros, zeros)

        cell_outputs_prev, cell_state_prev = hx

        lstm_concat = torch.cat([inputs, cell_outputs_prev], 1)
        peephole_concat = torch.cat([lstm_concat, cell_state_prev], 1)

        block_input = torch.tanh(self.block_input(lstm_concat))

        input_gate = torch.sigmoid(self.input_gate(peephole_concat))
        input_mult = input_gate * block_input

        forget_gate = torch.sigmoid(self.forget_gate(peephole_concat))
        forget_mult = forget_gate * cell_state_prev

        cell_state_new = input_mult + forget_mult
        cell_state_activated = torch.tanh(cell_state_new)

        output_concat = torch.cat([lstm_concat, cell_state_new], 1)
        output_gate = torch.sigmoid(self.output_gate(output_concat))
        cell_outputs_new = output_gate * cell_state_activated

        return cell_outputs_new, cell_state_new


class deadAliveNetBase(nn.Module):

    def __init__(self, device, args=None):
        super(deadAliveNetBase, self).__init__()
        self.device = device
        self.args = args
        self.learning_rate =  None
        self.optimizer = None
        self.outputs = None
        self.loss_function = nn.BCELoss()

    def loss(self, outputs, labels):
        return self.loss_function(outputs, labels)

    def setup_optimizer(self, learning_rate):
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.0005)

    def update_learning_rate(self, lr_new):
        if self.learning_rate != lr_new:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr_new
            self.learning_rate = lr_new

    def step(self, inputs, labels):
        self.optimizer.zero_grad()
        self.outputs = self(inputs)
        loss = self.loss(self.outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.data.cpu().numpy()[0]
       

class deadAliveNet(deadAliveNetBase):

    def __init__(self, device, lstm_size=1024, args=None):
        super(deadAliveNet, self).__init__(device, args)
        self.device = device
        self.lstm_size = lstm_size
        self.conv = nn.ModuleList([
            nn.Conv2d(1, 64, kernel_size=(11, 7), stride=(5, 3), padding=0),
            nn.Conv2d(64, 128, 5, padding=2, groups=2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.Conv2d(256, 256, 3, padding=1, groups=2),
            nn.Conv2d(256, 128, 3, padding=1, groups=2),
        ])

        self.lrn = nn.ModuleList([
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),
        ])

        self.conv_skip = nn.ModuleList([nn.Conv2d(64, 16, 1),
                                        nn.Conv2d(128, 32, 1),
                                        nn.Conv2d(128, 64, 1)
                                        ])
        
        self.prelu_skip = nn.ModuleList([torch.nn.PReLU(16),
                                         torch.nn.PReLU(32),
                                         torch.nn.PReLU(64),])
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # change based on what you see after concatenation
        self.fc6 = nn.Linear(72992, 2048)

        self.lstm1 = CaffeLSTMCell(2048, self.lstm_size)
        self.lstm2 = CaffeLSTMCell(2048 + self.lstm_size, self.lstm_size)

        self.lstm_state = None
        self.fc_output_out = nn.Linear(self.lstm_size, 6)

    def forward(self, input, lstm_state=None):
        # input = torch.from_numpy(input)
        batch_size = input.shape[0]
        # Conv1, pool1, lrn1
        conv1 = self.conv[0](input)
        pool1 = F.relu(F.max_pool2d(conv1, (3, 3), stride=(2, 1)))
        lrn1 = self.lrn[0](pool1)
        #print(f"lrn1 shape: {lrn1.shape}")

        # some skip connections
        conv1_skip = self.prelu_skip[0](self.conv_skip[0](lrn1))
        # flatten_skip, change the view
        conv1_skip_flatten = conv1_skip.view(batch_size, -1)

        #print(f"conv1_skip shape: {conv1_skip.shape}")
        #print(f"conv1_skip_flatten shape: {conv1_skip_flatten.shape}")

        # Conv2, pool1, lrn2
        conv2  = self.conv[1](lrn1)
        pool2 = F.relu(F.max_pool2d(conv2, (3, 3), stride=(2,1)))
        lrn2 = self.lrn[1](pool2)
        #print(f"lrn2 shape: {lrn2.shape}")

        # some skip connections
        conv2_skip = self.prelu_skip[1](self.conv_skip[1](lrn2))
        #flatten_skip, change the view
        conv2_skip_flatten = conv2_skip.view(batch_size, -1)

        #print(f"conv2_skip shape: {conv2_skip.shape}")
        #print(f"conv2_skip_flatten shape: {conv2_skip_flatten.shape}")

        # Conv3, Conv4, Conv5
        conv3 = F.relu(self.conv[2](lrn2))
        conv4 = F.relu(self.conv[3](conv3))
        conv5 = F.relu(self.conv[4](conv4))
        pool5 = F.relu(F.max_pool2d(conv5, (3, 3), stride=(2, 1)))

        #flatten pool5
        pool5_flat = pool5.view(batch_size, -1)
        #print(f"pool5_flat shape: {pool5_flat.shape}")
        
        # Conv5 skip
        conv5_skip = self.prelu_skip[2](self.conv_skip[2](conv5))
        conv5_skip_flatten = conv5_skip.view(batch_size, -1)
        #print(f"conv5_skip shape: {conv5_skip.shape}")
        #print(f"conv5_skip_flat shape: {conv5_skip_flatten.shape}")

        # concat, reshape, fc6
        skip_concat = torch.cat([conv1_skip_flatten, conv2_skip_flatten, conv5_skip_flatten, pool5_flat], 1)
        #print(f"--> All skips concatenated shape: {skip_concat.shape}")
        
        fc6 = F.relu(self.fc6(skip_concat))
        #lstm

        if lstm_state is None:
            outputs1, state1 = self.lstm1(fc6)
            outputs2, state2 = self.lstm2(torch.cat((fc6, outputs1), 1))
        else:
            outputs1, state1, outputs2, state2 = lstm_state
            outputs1, state1 = self.lstm1(fc6, (outputs1, state1))
            outputs2, state2 = self.lstm2(torch.cat((fc6, outputs1), 1), (outputs2, state2))
        
        self.lstm_state = (outputs1, state1, outputs2, state2)

        fc_output_out = self.fc_output_out(outputs2)
 
        # lstm output to probabilites
        return torch.sigmoid(fc_output_out)
        




class deadAliveNet80036(deadAliveNetBase):
    
    def __init__(self, device, lstm_size=1024, args=None):
        super(deadAliveNet, self).__init__(device, args)
        
        self.device = device
        self.lstm_size = lstm_size
        self.lstm_state = None
        
        self.conv = nn.ModuleList([
            nn.Conv2d(1, 8, kernel_size=(3, 3), stride=1, padding =1),
            nn.Conv2d(8, 16, kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1),          
        ])
        
        self.lrn = nn.ModuleList([
            nn.LocalResponseNorm(size=4, alpha=0.0001, beta=0.75),
            nn.LocalResponseNorm(size=4, alpha=0.0001, beta=0.75)
        ])
        
        self.conv_skip = nn.ModuleList([
            nn.Conv2d(8, 2, 1),
            nn.Conv2d(16, 4, 1),
            nn.Conv2d(32, 8, 1)
        ])
        
        self.prelu_skip = nn.ModuleList([
            nn.PReLU(2),
            nn.PReLU(4),
            nn.PReLU(8)
        ])
        
        self.fc6 = nn.Linear(48800, 1024)
        self.lstm1 = CaffeLSTMCell(1024, self.lstm_size)
        self.lstm2 = CaffeLSTMCell(1024 + self.lstm_size, self.lstm_size)
        
        self.fc_out = nn.Linear(self.lstm_size, 6)
        
    
    def forward(self, input, lstm_state=None):
        batch_size = input.shape[0]
        
        
        conv1 = self.conv[0](input)
        print(f"Conv1 shape: {conv1.shape}")
        pool1 = F.relu(F.max_pool2d(conv1, (2, 2)))
        print(f"Pool1 shape: {pool1.shape}")
        lrn1 = self.lrn[0](pool1)
        print(f"Lrn1 shape: {lrn1.shape}")
        
        # get the pool features into a vector for the final lstm at this spatial 
        # scale
        
        conv1_skip = self.prelu_skip[0](self.conv_skip[0](lrn1))
        print(f"Conv1_skip shape: {conv1_skip.shape}")
        # flatten to pool later
        conv1_skip_flatten = conv1_skip.view(batch_size, -1)
        print(f"Conv1_skip_flatten shape: {conv1_skip_flatten.shape}")
        
        
        conv2 = self.conv[1](lrn1)
        print(f"Conv2 shape: {conv2.shape}")
        pool2 = F.relu(F.max_pool2d(conv2, (2, 2)))
        print(f"Pool2 shape: {pool2.shape}")
        lrn2 = self.lrn[1](pool2)
        print(f"Lrn2 shape: {lrn2.shape}")
        
        
        conv2_skip = self.prelu_skip[1](self.conv_skip[1](lrn2))
        print(f"Conv2_skip shape: {conv2_skip.shape}")
        # flatten to pool later
        conv2_skip_flatten = conv2_skip.view(batch_size, -1)
        print(f"Conv2_skip_flatten shape: {conv2_skip_flatten.shape}")
        
        conv3 = F.relu(self.conv[2](lrn2))
        print(f"Conv3 shape: {conv3.shape}")
        conv4 = F.relu(self.conv[3](conv3))
        print(f"Conv4 shape: {conv4.shape}")
        
        conv4_skip = self.prelu_skip[2](self.conv_skip[2](conv4))
        print(f"Conv4_skip shape: {conv4_skip.shape}")
        conv4_skip_flatten = conv4_skip.view(batch_size, -1)
        print(f"Conv4_skip_flatten shape: {conv4_skip_flatten.shape}")
        
        pool4 = F.relu(F.max_pool2d(conv4, (2, 2)))
        print(f"Pool4 shape: {pool4.shape}")
        
        pool4_flat = pool4.view(batch_size, -1)
        print(f"Pool4_flat shape: {pool4_flat.shape}")
        
        skip_concat = torch.cat([conv1_skip_flatten, conv2_skip_flatten, conv4_skip_flatten, pool4_flat], 1)
        print(f"Skip concat shape: {skip_concat.shape}")
        
        fc6 = F.relu(self.fc6(skip_concat))
        print(f"FC6 shape: {fc6.shape}")
        
        
        if lstm_state is None:
            outputs1, state1 = self.lstm1(fc6)
            outputs2, state2 = self.lstm2(torch.cat((fc6, outputs1), 1))
        else:
            outputs1, state1, outputs2, state2 = lstm_state
            outputs1, state1 = self.lstm1(fc6, (outputs1, state1))
            outputs2, state2 = self.lstm2(torch.cat((fc6, outputs1), 1), (outputs2, state2))

        self.lstm_state  = (outputs1, state1, outputs2, state2)
        
        fc_out = self.fc_out(outputs2)
        
        print(f"FC_out shape: {fc_out.shape}")
        return torch.sigmoid(fc_out)
        